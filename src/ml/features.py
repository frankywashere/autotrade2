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
CHANNEL_PROJECTION_VERSION = "v2"  # v5.6: Removed fixed projections - now calculated at inference from learned duration
BREAKDOWN_CALC_VERSION = "v3"  # v5.8: Fixed window sizes for 1min input (was 5x too short in v2)
PARTIAL_BAR_VERSION = "v4"  # v5.6: Removed projected_high/low/center - projections calculated at inference
CONTINUATION_LABEL_VERSION = "v3.0"  # v3.0: Return-after-break tracking (first_break_bar, returned, bars_outside, final_duration)
FEATURE_VERSION = f"v5.9.1_vix{VIX_CALC_VERSION}_ev{EVENTS_CALC_VERSION}_proj{CHANNEL_PROJECTION_VERSION}_bd{BREAKDOWN_CALC_VERSION}_pb{PARTIAL_BAR_VERSION}_cont{CONTINUATION_LABEL_VERSION}"
# v5.9.1: Partial window support - TFs with <100 bars generate labels for windows that DO fit (restores 3month labels)

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


def _compute_partial_bar_channel_worker(task):
    """
    Worker function for parallel partial bar channel calculation.
    Must be at module level for joblib pickling.

    Args:
        task: Tuple of (df_data, df_columns, timestamps_ns, symbol, tf_name, tf_rule, windows)

    Returns:
        DataFrame with channel features for this symbol/timeframe combination
    """
    import pandas as pd
    import numpy as np
    import sys
    import time
    from .partial_channel_calc_vectorized import calculate_all_channel_features_vectorized

    df_data, df_columns, timestamps_ns, symbol, tf_name, tf_rule, windows = task

    # Debug: print start (flush immediately so it shows)
    start_time = time.time()
    print(f"      [Worker] Starting {symbol}_{tf_name}...", file=sys.stderr, flush=True)

    # Reconstruct DataFrame from numpy arrays
    df = pd.DataFrame(df_data, columns=df_columns)
    df.index = pd.to_datetime(timestamps_ns, unit='ns')

    # Enable debug for problematic TFs (15min and 30min)
    debug_mode = tf_name in ['15min', '30min']
    if debug_mode:
        print(f"      [Worker] {symbol}_{tf_name}: {len(df):,} bars, {len(windows)} windows", file=sys.stderr, flush=True)

    # Calculate channel features with partial bars
    result = calculate_all_channel_features_vectorized(
        df, symbol, tf_name, tf_rule,
        windows=windows, show_progress=False, debug=debug_mode
    )

    # Debug: print completion
    elapsed = time.time() - start_time
    print(f"      [Worker] Completed {symbol}_{tf_name} in {elapsed:.1f}s", file=sys.stderr, flush=True)

    return result


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

        # Multi-window channel features (v5.6: 31 features per window - removed projections)
        # Total: 14 windows × 11 TFs × 31 metrics × 2 symbols = 9,548 channel features
        # v5.6: Projections are now calculated at inference using learned duration predictions
        windows = config.CHANNEL_WINDOW_SIZES  # [100, 90, 80, ..., 10] (14 values)
        timeframes = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']

        # ALL 31 metrics from partial_channel_calc_vectorized.py (v5.6: removed projected_high/low/center)
        # Naming pattern: {symbol}_channel_{tf}_w{window}_{metric}
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
            # Ping-pongs - alternating touches (4 thresholds)
            'ping_pongs', 'ping_pongs_0_5pct', 'ping_pongs_1_0pct', 'ping_pongs_3_0pct',
            # Complete cycles - full round-trips (4 thresholds)
            'complete_cycles', 'complete_cycles_0_5pct', 'complete_cycles_1_0pct', 'complete_cycles_3_0pct',
            # Direction flags (3)
            'is_bull', 'is_bear', 'is_sideways',
            # Quality indicators (3)
            'quality_score', 'is_valid', 'insufficient_data',
            # Duration (1)
            'duration',
            # v5.6: Removed projected_high/low/center - now calculated at inference from learned duration
        ]

        for symbol in ['tsla', 'spy']:
            for tf in timeframes:
                for w in windows:
                    for m in metrics:
                        features.append(f'{symbol}_channel_{tf}_w{w}_{m}')

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
            'spy_is_volatile_now',       # v5.8: SPY volatility regime flag
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

    def extract_features(self, df: pd.DataFrame, use_cache: bool = True, use_gpu: str = 'auto', cache_suffix: str = None, events_handler=None, continuation: bool = False, continuation_mode: str = 'simple', use_chunking: bool = False, chunk_size_years: int = 1, shard_storage_path: str = None, vix_data: pd.DataFrame = None, skip_native_tf_generation: bool = False, skip_chunk_validation: bool = False, **kwargs) -> tuple:
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
            skip_chunk_validation: Skip validation of mmap chunk shards (default: False)
                - v5.9.2: Set True when native TF sequences exist but chunks were deleted
                - Allows training without keeping 60GB of chunk files
                - Use with use_cache=True when training layer is ready
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
        - 15 binary feature flags (is_monday, is_friday, is_volatile_now, spy_is_volatile_now, in_channel flags)
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
        # v5.9.1: Accept v5.9.0 channel shards as compatible (only label format changed)
        mmap_meta_files = list(unified_cache_dir.glob(f"features_mmap_meta_{FEATURE_VERSION}_*.json"))
        if not mmap_meta_files:
            # Backward compatibility: try v5.9.0 (same channel format)
            mmap_meta_files = list(unified_cache_dir.glob(f"features_mmap_meta_v5.9.0_*.json"))
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

                # v5.9.2: Skip chunk validation if flag is set (native TF sequences exist but chunks deleted)
                if skip_chunk_validation:
                    print(f"   ℹ️  Channel shards: Validation SKIPPED (using native TF sequences)")
                    print(f"       → Chunks not required for training when TF layer is ready")
                    channel_cache_valid = True
                    channel_cache_type = 'mmap'
                    self._mmap_meta_path = str(meta_file)
                    validated_meta_path = meta_file
                else:
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
                    # v5.9: Accept both old format (duration_bars) and new format (w50_duration)
                    has_old_format = 'duration_bars' in test_load.columns
                    has_new_format = 'w50_duration' in test_load.columns or 'w10_duration' in test_load.columns

                    if len(test_load) > 0 and (has_old_format or has_new_format):
                        format_type = "v5.6 (single-window)" if has_old_format else "v5.9 (multi-window)"
                        print(f"   ✓ Continuation labels (hierarchical): Valid ({len(HIERARCHICAL_TIMEFRAMES)} TFs cached, {format_type})")
                        cont_cache_valid = True
                    else:
                        print(f"   ⚠️  Continuation labels: Invalid format - will regenerate")
                except Exception as e:
                    print(f"   ⚠️  Continuation labels: Corrupted ({type(e).__name__}) - will regenerate")
            elif len(found_tfs) >= len(HIERARCHICAL_TIMEFRAMES) - 1:
                # v5.9.1: Accept 10/11 or 11/11 as valid (3month may have partial windows)
                if missing_tfs:
                    print(f"   ⚠️  Continuation labels: {len(found_tfs)}/{len(HIERARCHICAL_TIMEFRAMES)} TFs (missing: {missing_tfs} - tolerance triggered)")
                else:
                    print(f"   ✓ Continuation labels: Valid ({len(found_tfs)}/{len(HIERARCHICAL_TIMEFRAMES)} TFs)")
                cont_cache_valid = True
            elif len(found_tfs) > 0:
                print(f"   ⚠️  Continuation labels: Partial ({len(found_tfs)}/{len(HIERARCHICAL_TIMEFRAMES)} TFs) - will regenerate missing")
            else:
                print(f"   ❌ Continuation labels: Not found - will generate (~1 hour)")

        # Check 3: Non-channel features (Price, RSI, Correlation, Cycle, Volume, Time, Breakdown)
        non_channel_cache_path = unified_cache_dir / f"non_channel_features_{cache_key}.pkl"
        # v5.9.1: Try v5.9.0 cache if v5.9.1 not found (backward compatible)
        if not non_channel_cache_path.exists():
            cache_key_v590 = cache_key.replace('v5.9.1', 'v5.9.0').replace('contv2.1', 'contv2')
            non_channel_cache_path_v590 = unified_cache_dir / f"non_channel_features_{cache_key_v590}.pkl"
            if non_channel_cache_path_v590.exists():
                non_channel_cache_path = non_channel_cache_path_v590
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
            # v5.9.2: Accept 10/11 files (3month may not have enough data for transitions)
            all_continuation_cached = True
            all_transition_cached = True
            found_transition_tfs = []
            missing_transition_tfs = []
            if use_cache:
                for tf in HIERARCHICAL_TIMEFRAMES:
                    # Check continuation labels
                    tf_label_path = cache_dir / f"continuation_labels_{tf}_{cache_suffix}.pkl"
                    if not tf_label_path.exists():
                        all_continuation_cached = False
                    # Check transition labels (v5.2)
                    tf_transition_path = cache_dir / f"transition_labels_{tf}_{cache_suffix}.pkl"
                    if tf_transition_path.exists():
                        found_transition_tfs.append(tf)
                    else:
                        missing_transition_tfs.append(tf)
                # v5.9.2: Accept 10/11 transition files (3month often has <20 rows, can't generate transitions)
                all_transition_cached = len(found_transition_tfs) >= len(HIERARCHICAL_TIMEFRAMES) - 1
                if all_transition_cached and len(missing_transition_tfs) > 0:
                    print(f"   ⚠️  Transition labels: {len(found_transition_tfs)}/{len(HIERARCHICAL_TIMEFRAMES)} (missing: {missing_transition_tfs} - tolerance triggered)")
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
                    missing_info = f", missing: {missing_transition_tfs}" if missing_transition_tfs else ""
                    print(f"   📂 Found cached transition labels ({len(found_transition_tfs)}/{len(HIERARCHICAL_TIMEFRAMES)} TFs{missing_info})")
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

        # v5.4: Single-pass processing with partial bar channels
        # Calculate breakdown at 5min from 5min channel features, then resample to each TF
        print(f"   📊 Calculating breakdown features at 5min resolution (v5.4)...")

        # Calculate all breakdown features at 5min resolution
        breakdown_5min = self._calculate_all_breakdown_at_5min(
            features_df,
            raw_df=raw_df,
            events_handler=events_handler
        )

        # Add breakdown to features_df for resampling
        features_with_breakdown = pd.concat([features_df, breakdown_5min], axis=1, copy=False)

        print(f"   💾 Resampling and saving timeframe sequences...")

        for tf in tqdm(HIERARCHICAL_TIMEFRAMES, desc="   Saving TF sequences", leave=False, ncols=100, ascii=True):
            # Get columns for this TF: shared + TF-specific + all breakdown
            tf_cols = shared_cols + tf_specific_cols[tf] + list(breakdown_5min.columns)
            tf_features = features_with_breakdown[tf_cols].copy()

            # Resample to native TF resolution
            tf_rule = TIMEFRAME_RESAMPLE_RULES[tf]
            resampled = tf_features.resample(tf_rule).last().dropna()

            # Remove duplicate columns (if any)
            resampled = resampled.loc[:, ~resampled.columns.duplicated(keep='first')]

            # Update metadata with final column list
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
        Peak RAM: ~120GB during hstack/concat operations (with memory cleanup).
        Without cleanup: ~245GB peak. With del+gc.collect(): ~120GB peak.
        """
        import json
        import gc
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

        # Free chunk list - no longer needed after vstack (saves ~49 GB)
        del all_features, all_indices
        gc.collect()
        print(f"   🗑️  Freed chunk list memory")

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

                # Free monthly array - no longer needed after hstack (saves ~11 GB)
                del monthly_array
                gc.collect()
                print(f"   🗑️  Freed monthly shard memory")
            else:
                print(f"   ⚠️  Monthly shard not found: {monthly_path}")

        # Create DataFrame with timestamps as index and proper column names
        df = pd.DataFrame(full_features, columns=feature_columns)
        df.index = pd.to_datetime(full_indices, unit='ns')

        # Free numpy arrays - DataFrame now owns the data (saves ~60 GB)
        del full_features, full_indices
        gc.collect()
        print(f"   🗑️  Freed numpy array memory")

        print(f"   ✓ Loaded {len(df):,} rows × {len(df.columns)} channel features")

        # Load and merge non-channel features (contains tsla_close, spy_close, etc.)
        non_channel_path = cache_dir / f"non_channel_features_{mmap_meta['cache_key']}.pkl"
        if non_channel_path.exists():
            print(f"   📂 Loading non-channel features...")
            non_channel_df = pd.read_pickle(non_channel_path)
            # Ensure same index format
            if not isinstance(non_channel_df.index, pd.DatetimeIndex):
                non_channel_df.index = pd.to_datetime(non_channel_df.index)

            # v5.4: Remove old breakdown columns - we'll calculate fresh at 5min resolution
            breakdown_patterns = [
                'duration_ratio', 'alignment', 'time_in_channel', 'position_norm',
                'in_channel', 'rsi_divergence', 'volume_surge',
                'is_monday', 'is_tuesday', 'is_wednesday', 'is_thursday', 'is_friday',
                'is_first_hour', 'is_last_hour', 'is_volatile_now',
                'is_earnings_week', 'days_until_earnings', 'days_until_fomc', 'is_high_impact_event'
            ]
            old_breakdown_cols = [c for c in non_channel_df.columns
                                  if any(pattern in c for pattern in breakdown_patterns)]
            if old_breakdown_cols:
                non_channel_df = non_channel_df.drop(columns=old_breakdown_cols)
                print(f"   ✓ Removed {len(old_breakdown_cols)} old breakdown cols (will recalculate at 5min)")

            # Align indices and merge (non-channel first so tsla_close is accessible)
            common_idx = df.index.intersection(non_channel_df.index)
            if len(common_idx) > 0:
                df = pd.concat([non_channel_df.loc[common_idx], df.loc[common_idx]], axis=1)
                print(f"   ✓ Merged {len(non_channel_df.columns)} non-channel features ({len(df):,} aligned rows)")

                # Free non-channel DataFrame - no longer needed after concat (saves ~1.2 GB)
                del non_channel_df
                gc.collect()
            else:
                print(f"   ⚠️  No overlapping timestamps between channel and non-channel features")
                del non_channel_df
                gc.collect()
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

        # v5.4: Single-pass processing with breakdown at 5min resolution
        # Calculate breakdown from 5min channel features, then resample to each TF
        print(f"   📊 Calculating breakdown at 5min resolution (v5.4 single-pass)...")
        breakdown_5min = self._calculate_all_breakdown_at_5min(
            df,
            raw_df=None,          # Not available in chunked mode
            events_handler=None   # Not available in chunked mode
        )
        print(f"   ✓ Generated {len(breakdown_5min.columns)} breakdown features at 5min")

        # Save breakdown column names BEFORE deletion
        breakdown_cols = list(breakdown_5min.columns)

        # Merge breakdown with features at 5min
        df_with_breakdown = pd.concat([df, breakdown_5min], axis=1)

        # Free original df and breakdown - now combined in df_with_breakdown (saves ~62 GB)
        del df, breakdown_5min
        gc.collect()
        print(f"   🗑️  Freed pre-merge DataFrames")

        # Now resample to each TF and save
        print(f"   💾 Resampling to native TF resolutions and saving...")

        for tf in tqdm(HIERARCHICAL_TIMEFRAMES, desc="   Saving TF sequences", leave=False, ncols=100, ascii=True):
            # Get columns for this TF (shared + TF-specific + breakdown)
            tf_cols = shared_cols + tf_specific_cols[tf]
            # breakdown_cols already saved above

            # Select columns for this timeframe
            tf_features = df_with_breakdown[tf_cols + breakdown_cols].copy()
            tf_rule = TIMEFRAME_RESAMPLE_RULES[tf]

            # Use .last() to get value at end of each bar
            resampled = tf_features.resample(tf_rule).last().dropna()

            # Remove duplicate columns (can happen from same-TF concat)
            resampled = resampled.loc[:, ~resampled.columns.duplicated(keep='first')]

            # Update metadata with final column list
            meta['timeframe_columns'][tf] = list(resampled.columns)

            # Save as .npy file
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
        True streaming implementation: ~7-8 GB peak RAM instead of ~120 GB.

        Strategy:
        - Phase 1: Load only 48 breakdown-required columns (~1 GB peak)
        - Phase 2: For each TF, load only that TF's columns (~7 GB peak per TF)
        - Memory freed between each TF iteration

        Produces identical results to _generate_native_tf_full_load.
        """
        import json
        import gc
        import os
        import pandas as pd

        total_rows = mmap_meta['total_rows']
        cache_key = mmap_meta.get('cache_key', 'from_chunks')

        print(f"   🔄 True streaming mode: ~7-8 GB peak RAM (vs ~120 GB full load)")

        # ========================================================================
        # PHASE 0: Load timestamps and compute column mappings (~50 MB)
        # ========================================================================
        print(f"   📊 Phase 0: Loading timestamps and computing mappings...")

        # Load all timestamp indices
        all_indices = []
        chunk_row_ranges = []
        row_offset = 0

        for chunk in chunk_info:
            index_path = cache_dir / chunk['index_path']
            index_array = np.load(str(index_path), mmap_mode='r')[:]
            all_indices.append(index_array)
            chunk_row_ranges.append((row_offset, row_offset + len(index_array)))
            row_offset += len(index_array)

        full_indices = np.concatenate(all_indices)
        del all_indices
        gc.collect()

        datetime_index = pd.to_datetime(full_indices, unit='ns')

        # Compute column index mappings
        shared_chunk_col_indices = []
        tf_chunk_col_indices = {tf: [] for tf in HIERARCHICAL_TIMEFRAMES}

        for i, col in enumerate(feature_columns):
            is_tf_specific = False
            for tf in HIERARCHICAL_TIMEFRAMES:
                if f'_{tf}_' in col:
                    tf_chunk_col_indices[tf].append(i)
                    is_tf_specific = True
                    break
            if not is_tf_specific:
                shared_chunk_col_indices.append(i)

        # Monthly shard column mappings
        monthly_columns = []
        shared_monthly_col_indices = []
        tf_monthly_col_indices = {tf: [] for tf in HIERARCHICAL_TIMEFRAMES}

        if monthly_shard_info:
            monthly_columns = monthly_shard_info.get('columns', [])
            for i, col in enumerate(monthly_columns):
                is_tf_specific = False
                for tf in HIERARCHICAL_TIMEFRAMES:
                    if f'_{tf}_' in col:
                        tf_monthly_col_indices[tf].append(i)
                        is_tf_specific = True
                        break
                if not is_tf_specific:
                    shared_monthly_col_indices.append(i)

        print(f"   ✓ Chunk columns: {len(shared_chunk_col_indices)} shared + {sum(len(v) for v in tf_chunk_col_indices.values())} TF-specific")
        if monthly_columns:
            print(f"   ✓ Monthly columns: {len(shared_monthly_col_indices)} shared + {sum(len(v) for v in tf_monthly_col_indices.values())} TF-specific")

        # ========================================================================
        # PHASE 1: Compute breakdown from minimal columns (~1 GB peak)
        # ========================================================================
        print(f"   📊 Phase 1: Computing breakdown features...")

        # Find breakdown-required column indices (36 from chunks: stability/position for 9 TFs)
        breakdown_chunk_col_indices = []
        breakdown_chunk_col_names = []
        tfs_with_w50 = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly']

        for tf in tfs_with_w50:
            for col_pattern in [
                f'tsla_channel_{tf}_w50_stability',
                f'tsla_channel_{tf}_w50_position',
                f'spy_channel_{tf}_w50_stability',
                f'spy_channel_{tf}_w50_position',
            ]:
                if col_pattern in feature_columns:
                    idx = feature_columns.index(col_pattern)
                    breakdown_chunk_col_indices.append(idx)
                    breakdown_chunk_col_names.append(col_pattern)

        print(f"   📂 Loading {len(breakdown_chunk_col_indices)} breakdown cols from chunks...")

        # Load only breakdown columns from chunks (~0.24 GB)
        breakdown_chunk_data = np.zeros((total_rows, len(breakdown_chunk_col_indices)), dtype=np.float32)

        for chunk_idx, chunk in enumerate(chunk_info):
            chunk_path = cache_dir / chunk['path']
            chunk_mm = np.load(str(chunk_path), mmap_mode='r')

            start_row, end_row = chunk_row_ranges[chunk_idx]
            for col_out_idx, col_in_idx in enumerate(breakdown_chunk_col_indices):
                breakdown_chunk_data[start_row:end_row, col_out_idx] = chunk_mm[:, col_in_idx]

            del chunk_mm

        # Build breakdown input DataFrame
        breakdown_df = pd.DataFrame(breakdown_chunk_data, columns=breakdown_chunk_col_names, index=datetime_index)
        del breakdown_chunk_data
        gc.collect()

        # v5.7 fix: Also load w50 columns for monthly/3month from monthly shard
        # These TFs have their w50 columns in the monthly shard, not in chunks
        if monthly_shard_info and monthly_columns:
            monthly_path = cache_dir / monthly_shard_info['path']
            if monthly_path.exists():
                monthly_mm_temp = np.load(str(monthly_path), mmap_mode='r')
                monthly_breakdown_cols_added = 0

                for tf in ['monthly', '3month']:
                    for col_pattern in [
                        f'tsla_channel_{tf}_w50_stability',
                        f'tsla_channel_{tf}_w50_position',
                        f'spy_channel_{tf}_w50_stability',
                        f'spy_channel_{tf}_w50_position',
                    ]:
                        if col_pattern in monthly_columns:
                            col_idx = monthly_columns.index(col_pattern)
                            breakdown_df[col_pattern] = monthly_mm_temp[:total_rows, col_idx]
                            monthly_breakdown_cols_added += 1

                del monthly_mm_temp
                if monthly_breakdown_cols_added > 0:
                    print(f"   ✓ Added {monthly_breakdown_cols_added} monthly/3month w50 columns for breakdown")

        # Load non-channel features
        non_channel_path = cache_dir / f"non_channel_features_{cache_key}.pkl"

        # v5.9.2: Backward compatibility - try alternate versions if exact match not found
        if not non_channel_path.exists():
            # Try v5.9.0 if looking for v5.9.1
            cache_key_v590 = cache_key.replace('v5.9.1', 'v5.9.0').replace('contv2.1', 'contv2')
            alt_path = cache_dir / f"non_channel_features_{cache_key_v590}.pkl"
            if alt_path.exists():
                non_channel_path = alt_path
                print(f"   ℹ️  Using v5.9.0 non-channel (backward compat): {alt_path.name}")
            else:
                # Try v5.9.1 if looking for v5.9.0
                cache_key_v591 = cache_key.replace('v5.9.0', 'v5.9.1').replace('contv2', 'contv2.1')
                alt_path = cache_dir / f"non_channel_features_{cache_key_v591}.pkl"
                if alt_path.exists():
                    non_channel_path = alt_path
                    print(f"   ℹ️  Using v5.9.1 non-channel (forward compat): {alt_path.name}")
                else:
                    # Try to find ANY non-channel file and use it (last resort)
                    nc_files = list(cache_dir.glob("non_channel_features_*.pkl"))
                    if nc_files:
                        # Sort by size (larger = more data) and use largest
                        nc_files_sorted = sorted(nc_files, key=lambda p: p.stat().st_size, reverse=True)
                        non_channel_path = nc_files_sorted[0]
                        print(f"   ⚠️  Using fallback non-channel (date mismatch possible): {non_channel_path.name}")

        non_channel_df = None
        non_channel_columns = []

        if non_channel_path.exists():
            non_channel_df = pd.read_pickle(non_channel_path)
            if not isinstance(non_channel_df.index, pd.DatetimeIndex):
                non_channel_df.index = pd.to_datetime(non_channel_df.index)

            # Remove old breakdown columns
            breakdown_patterns = [
                'duration_ratio', 'alignment', 'time_in_channel', 'position_norm',
                'in_channel', 'rsi_divergence', 'volume_surge',
                'is_monday', 'is_tuesday', 'is_wednesday', 'is_thursday', 'is_friday',
                'is_first_hour', 'is_last_hour', 'is_volatile_now',
                'is_earnings_week', 'days_until_earnings', 'days_until_fomc', 'is_high_impact_event'
            ]
            old_breakdown_cols = [c for c in non_channel_df.columns
                                  if any(pattern in c for pattern in breakdown_patterns)]
            if old_breakdown_cols:
                non_channel_df = non_channel_df.drop(columns=old_breakdown_cols)
                print(f"   📂 Non-channel: {len(non_channel_df.columns)} cols (removed {len(old_breakdown_cols)} old breakdown)")

            non_channel_columns = list(non_channel_df.columns)

            # Add breakdown-required non-channel cols (RSI + tsla_volume_ratio + tsla_close)
            breakdown_nc_cols = []
            for tf in HIERARCHICAL_TIMEFRAMES:
                rsi_col = f'tsla_rsi_{tf}'
                if rsi_col in non_channel_df.columns:
                    breakdown_nc_cols.append(rsi_col)
            if 'tsla_volume_ratio' in non_channel_df.columns:
                breakdown_nc_cols.append('tsla_volume_ratio')
            if 'tsla_close' in non_channel_df.columns:
                breakdown_nc_cols.append('tsla_close')

            # Align indices
            common_idx = breakdown_df.index.intersection(non_channel_df.index)
            if len(common_idx) < len(breakdown_df):
                breakdown_df = breakdown_df.loc[common_idx]
                datetime_index = breakdown_df.index
                print(f"   ⚠️  Rows trimmed by alignment: {total_rows:,} → {len(common_idx):,}")
                total_rows = len(common_idx)

            for col in breakdown_nc_cols:
                if col in non_channel_df.columns:
                    breakdown_df[col] = non_channel_df.loc[breakdown_df.index, col].values

            print(f"   ✓ Added {len(breakdown_nc_cols)} non-channel cols for breakdown")
        else:
            print(f"   ⚠️  No non-channel features found: {non_channel_path.name}")

        # Calculate breakdown features
        print(f"   📊 Calculating breakdown at 5min resolution...")
        breakdown_result = self._calculate_all_breakdown_at_5min(
            breakdown_df,
            raw_df=None,
            events_handler=None
        )
        breakdown_columns = list(breakdown_result.columns)
        print(f"   ✓ Generated {len(breakdown_columns)} breakdown features")

        # Save breakdown to temp file
        breakdown_temp_path = output_cache_dir / f"_temp_breakdown_{cache_key}.npy"
        np.save(str(breakdown_temp_path), breakdown_result.values.astype(np.float32))

        del breakdown_df, breakdown_result
        gc.collect()

        # Memory-map breakdown for reading
        breakdown_mm = np.load(str(breakdown_temp_path), mmap_mode='r')

        # ========================================================================
        # PHASE 2: Generate each TF output (~7 GB peak per TF)
        # ========================================================================
        print(f"   💾 Phase 2: Generating TF outputs (one at a time)...")

        # Validate monthly shard if present
        monthly_mm = None
        if monthly_shard_info:
            monthly_path = cache_dir / monthly_shard_info['path']
            if monthly_path.exists():
                monthly_mm = np.load(str(monthly_path), mmap_mode='r')
                if monthly_mm.shape[0] != mmap_meta['total_rows']:
                    raise ValueError(f"Monthly shard row mismatch: {monthly_mm.shape[0]} vs {mmap_meta['total_rows']}")

        # Categorize non-channel columns into shared vs TF-specific (Bug #1 fix)
        # Must use SAME pattern as full_load: only _{tf}_ (underscore on both sides)
        # e.g., tsla_rsi_5min → SHARED, tsla_rsi_5min_oversold → TF-SPECIFIC
        nc_shared_cols = []
        nc_tf_cols = {tf: [] for tf in HIERARCHICAL_TIMEFRAMES}
        for col in non_channel_columns:
            is_tf_specific = False
            for tf in HIERARCHICAL_TIMEFRAMES:
                if f'_{tf}_' in col:  # Must match full_load exactly
                    nc_tf_cols[tf].append(col)
                    is_tf_specific = True
                    break
            if not is_tf_specific:
                nc_shared_cols.append(col)

        print(f"   ✓ Non-channel split: {len(nc_shared_cols)} shared + {sum(len(v) for v in nc_tf_cols.values())} TF-specific")

        # Build shared_cols list to match full_load order (nc_shared + chunk_shared + monthly_shared)
        all_shared_cols = nc_shared_cols + [feature_columns[i] for i in shared_chunk_col_indices]
        if monthly_mm is not None:
            all_shared_cols.extend([monthly_columns[i] for i in shared_monthly_col_indices])

        # Metadata for output
        meta = {
            'feature_version': FEATURE_VERSION,
            'cache_key': cache_key,
            'sequence_lengths': TIMEFRAME_SEQUENCE_LENGTHS,
            'shared_columns': all_shared_cols,
            'timeframe_columns': {},
            'timeframe_shapes': {},
            'total_rows_1min': total_rows,
        }

        for tf in tqdm(HIERARCHICAL_TIMEFRAMES, desc="   Generating TFs", leave=False, ncols=100, ascii=True):
            # Column order must match full_load: shared_cols + tf_specific_cols[tf] + breakdown_cols
            # Where shared = nc_shared + chunk_shared + monthly_shared
            # And tf_specific = nc_tf[tf] + chunk_tf[tf] + monthly_tf[tf]
            tf_col_names = []

            # 1. All shared columns first (matches full_load's shared_cols)
            tf_col_names.extend(nc_shared_cols)
            tf_col_names.extend([feature_columns[i] for i in shared_chunk_col_indices])
            if monthly_mm is not None and shared_monthly_col_indices:
                tf_col_names.extend([monthly_columns[i] for i in shared_monthly_col_indices])

            # 2. TF-specific columns (matches full_load's tf_specific_cols[tf])
            tf_col_names.extend(nc_tf_cols[tf])
            tf_col_names.extend([feature_columns[i] for i in tf_chunk_col_indices[tf]])
            if monthly_mm is not None and tf_monthly_col_indices[tf]:
                tf_col_names.extend([monthly_columns[i] for i in tf_monthly_col_indices[tf]])

            # 3. Breakdown columns
            tf_col_names.extend(breakdown_columns)

            # Allocate and fill data array (~7 GB for largest TFs)
            num_cols = len(tf_col_names)
            tf_data = np.zeros((total_rows, num_cols), dtype=np.float32)
            col_offset = 0

            # Data fill order must match column order exactly:
            # SHARED: nc_shared → chunk_shared → monthly_shared
            # TF-SPECIFIC: nc_tf[tf] → chunk_tf[tf] → monthly_tf[tf]
            # BREAKDOWN

            # --- SHARED SECTION ---
            # 1. nc_shared
            if non_channel_df is not None and nc_shared_cols:
                nc_shared_values = non_channel_df.loc[datetime_index, nc_shared_cols].values
                tf_data[:, col_offset:col_offset + len(nc_shared_cols)] = nc_shared_values
                col_offset += len(nc_shared_cols)

            # 2. chunk_shared
            if shared_chunk_col_indices:
                for chunk_idx, chunk in enumerate(chunk_info):
                    chunk_path = cache_dir / chunk['path']
                    chunk_mm = np.load(str(chunk_path), mmap_mode='r')
                    start_row, end_row = chunk_row_ranges[chunk_idx]
                    if end_row > total_rows:
                        end_row = total_rows
                    if start_row >= total_rows:
                        del chunk_mm
                        continue
                    for out_col, in_col in enumerate(shared_chunk_col_indices):
                        tf_data[start_row:end_row, col_offset + out_col] = chunk_mm[:end_row - start_row, in_col]
                    del chunk_mm
                col_offset += len(shared_chunk_col_indices)

            # 3. monthly_shared
            if monthly_mm is not None and shared_monthly_col_indices:
                for out_col, in_col in enumerate(shared_monthly_col_indices):
                    tf_data[:total_rows, col_offset + out_col] = monthly_mm[:total_rows, in_col]
                col_offset += len(shared_monthly_col_indices)

            # --- TF-SPECIFIC SECTION ---
            # 4. nc_tf[tf]
            if non_channel_df is not None and nc_tf_cols[tf]:
                nc_tf_values = non_channel_df.loc[datetime_index, nc_tf_cols[tf]].values
                tf_data[:, col_offset:col_offset + len(nc_tf_cols[tf])] = nc_tf_values
                col_offset += len(nc_tf_cols[tf])

            # 5. chunk_tf[tf]
            if tf_chunk_col_indices[tf]:
                for chunk_idx, chunk in enumerate(chunk_info):
                    chunk_path = cache_dir / chunk['path']
                    chunk_mm = np.load(str(chunk_path), mmap_mode='r')
                    start_row, end_row = chunk_row_ranges[chunk_idx]
                    if end_row > total_rows:
                        end_row = total_rows
                    if start_row >= total_rows:
                        del chunk_mm
                        continue
                    for out_col, in_col in enumerate(tf_chunk_col_indices[tf]):
                        tf_data[start_row:end_row, col_offset + out_col] = chunk_mm[:end_row - start_row, in_col]
                    del chunk_mm
                col_offset += len(tf_chunk_col_indices[tf])

            # 6. monthly_tf[tf]
            if monthly_mm is not None and tf_monthly_col_indices[tf]:
                for out_col, in_col in enumerate(tf_monthly_col_indices[tf]):
                    tf_data[:total_rows, col_offset + out_col] = monthly_mm[:total_rows, in_col]
                col_offset += len(tf_monthly_col_indices[tf])

            # --- BREAKDOWN ---
            # 7. breakdown
            tf_data[:, col_offset:col_offset + len(breakdown_columns)] = breakdown_mm[:total_rows, :]

            # Build DataFrame for resampling
            tf_df = pd.DataFrame(tf_data, columns=tf_col_names, index=datetime_index)
            del tf_data
            gc.collect()

            # Resample using .last()
            tf_rule = TIMEFRAME_RESAMPLE_RULES[tf]
            resampled = tf_df.resample(tf_rule).last().dropna()
            del tf_df
            gc.collect()

            # Remove duplicate columns
            resampled = resampled.loc[:, ~resampled.columns.duplicated(keep='first')]

            # Save outputs
            output_path = output_cache_dir / f"tf_sequence_{tf}_{cache_key}.npy"
            np.save(str(output_path), resampled.values.astype(np.float32))

            ts_path = output_cache_dir / f"tf_timestamps_{tf}_{cache_key}.npy"
            np.save(str(ts_path), resampled.index.view(np.int64))

            # Update metadata
            meta['timeframe_columns'][tf] = list(resampled.columns)
            meta['timeframe_shapes'][tf] = list(resampled.shape)

            # Log progress
            seq_len = TIMEFRAME_SEQUENCE_LENGTHS[tf]
            real_time = {
                '5min': '~17 hours', '15min': '~25 hours', '30min': '~40 hours',
                '1h': '1 week', '2h': '1 week', '3h': '1 week', '4h': '1 week',
                'daily': '30 days', 'weekly': '20 weeks', 'monthly': '12 months', '3month': '24 months'
            }
            print(f"      {tf}: {resampled.shape[0]:,} bars × {resampled.shape[1]} features (seq_len={seq_len}, {real_time.get(tf, '?')})")

            del resampled
            gc.collect()

        # ========================================================================
        # PHASE 3: Cleanup
        # ========================================================================
        del breakdown_mm
        if monthly_mm is not None:
            del monthly_mm
        if non_channel_df is not None:
            del non_channel_df
        gc.collect()

        # Delete temp breakdown file
        try:
            os.remove(str(breakdown_temp_path))
        except OSError:
            pass

        # Save metadata
        meta_path = output_cache_dir / f"tf_meta_{cache_key}.json"
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"   ✓ Generated {len(HIERARCHICAL_TIMEFRAMES)} timeframe sequences (true streaming)")
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

    def _extract_channel_features(self, df: pd.DataFrame, use_cache: bool = True, multi_res_data: dict = None, use_gpu: bool = False, cache_suffix: str = None, use_partial_bars: bool = True) -> pd.DataFrame:
        """
        Extract ROLLING linear regression channel features for multiple timeframes.

        CRITICAL: Channels are calculated at EACH timestamp using a rolling lookback window.
        This captures channel dynamics (formation, strength, breakdown) over time.

        v5.4: When use_partial_bars=True, channels for coarser TFs include in-progress data.
        At Monday 9:30am, weekly channel includes [last 49 weeks] + [Monday's partial data].

        NOW PROCESSES BOTH TSLA AND SPY (v3.4)
        Returns DataFrame with channel features at 5min resolution.

        Args:
            df: OHLCV DataFrame
            use_cache: If True, load from cache or save to cache (recommended)
            use_gpu: If True, use GPU acceleration (10-20x faster for large datasets)
            cache_suffix: Optional suffix for cache filename (for testing, e.g., 'GPU_TEST')
            use_partial_bars: If True, include partial bar data for coarser TFs (v5.4)
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
                        num_windows = len(config.CHANNEL_WINDOW_SIZES)  # 14 windows
                        features_per_window = 31  # v5.6: Removed projected_high/low/center (was 34)
                        timeframes = 11  # 5min, 15min, ..., monthly, 3month
                        stocks = 2  # TSLA, SPY
                        expected_cols = num_windows * features_per_window * timeframes * stocks  # = 9,548

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

        # v5.4: Use partial bar calculation for training mode (not live mode)
        if use_partial_bars and not is_live_mode:
            from .partial_channel_calc_vectorized import calculate_all_channel_features_vectorized

            print(f"   🔄 Calculating channels with PARTIAL BARS (v5.6 - includes in-progress TF data)...")
            print(f"   📊 Processing 11 timeframes × 2 stocks × {len(config.CHANNEL_WINDOW_SIZES)} windows × 31 features")

            all_timeframes = {
                '5min': '5min', '15min': '15min', '30min': '30min', '1h': '1h',
                '2h': '2h', '3h': '3h', '4h': '4h', 'daily': '1D',
                'weekly': '1W', 'monthly': '1ME', '3month': '3ME'
            }

            # Filter timeframes if chunking
            if cache_suffix and 'chunk' in str(cache_suffix):
                timeframes = {k: v for k, v in all_timeframes.items() if k not in ['monthly', '3month']}
                print(f"   ℹ️  Skipping monthly/3month (processed separately on full dataset)")
            else:
                timeframes = all_timeframes

            # Determine whether to use parallel processing
            # Note: GPU is for rolling stats (price/correlation), not channel regression
            # Channel extraction is CPU-bound - parallel CPU is still fastest
            n_cores = mp.cpu_count()
            parallel_enabled = config.PARALLEL_CHANNEL_CALC if hasattr(config, 'PARALLEL_CHANNEL_CALC') else True
            use_parallel = parallel_enabled

            if use_parallel:
                # Parallel processing for partial bar channels (using joblib like original)
                max_workers = config.MAX_PARALLEL_WORKERS if hasattr(config, 'MAX_PARALLEL_WORKERS') else 0
                n_jobs = max_workers if max_workers > 0 else -1
                cores_to_use = get_safe_worker_count(max_workers if max_workers > 0 else None)
                print(f"   🚀 Parallel processing: using {cores_to_use} of {n_cores} cores")
                print(f"   📋 Config: MAX_PARALLEL_WORKERS={max_workers}, n_jobs={n_jobs}")
                print(f"   ⏱️  Estimated time: ~{22 // max(cores_to_use, 1) + 2} minutes")

                # Build task list for parallel execution
                # Prepare data for workers (numpy arrays are picklable)
                df_data = df.values
                timestamps_ns = df.index.view(np.int64)
                df_columns = list(df.columns)
                windows = config.CHANNEL_WINDOW_SIZES

                tasks = []
                for symbol in ['tsla', 'spy']:
                    for tf_name, tf_rule in timeframes.items():
                        tasks.append((df_data, df_columns, timestamps_ns, symbol, tf_name, tf_rule, windows))

                # DEBUG: Test one task sequentially first to verify code works
                print(f"   🧪 Testing first task sequentially (tsla_5min)...")
                import time as _time
                _test_start = _time.time()
                _test_result = _compute_partial_bar_channel_worker(tasks[0])
                _test_elapsed = _time.time() - _test_start
                print(f"   ✅ Test task completed in {_test_elapsed:.1f}s - {_test_result.shape[1]} features")

                # Validate: check that results are not all zeros
                _nonzero_cols = (_test_result.abs().sum() > 0).sum()
                _sample_col = [c for c in _test_result.columns if 'position' in c][0]
                _sample_vals = _test_result[_sample_col].iloc[-100:]
                _nonzero_pct = (_sample_vals != 0.5).mean() * 100  # 0.5 is default
                print(f"   📊 Validation: {_nonzero_cols}/{_test_result.shape[1]} columns have data, "
                      f"position varies {_nonzero_pct:.0f}% of last 100 bars")
                if _nonzero_pct < 50:
                    print(f"   ⚠️  WARNING: Position values mostly default (0.5) - may indicate bug!")
                # Show sample values
                print(f"   📈 Sample {_sample_col}: min={_sample_vals.min():.3f}, "
                      f"max={_sample_vals.max():.3f}, mean={_sample_vals.mean():.3f}")

                # Use joblib for parallel processing (same as original channel code)
                print(f"   🚀 Launching {len(tasks)-1} remaining tasks in parallel...")
                all_results = [_test_result]  # Start with test result
                all_results.extend(list(
                    tqdm(
                        Parallel(
                            n_jobs=n_jobs,
                            backend='loky',
                            prefer="processes",
                            verbose=0,
                            return_as="generator"
                        )(delayed(_compute_partial_bar_channel_worker)(task) for task in tasks[1:]),
                        total=len(tasks)-1,
                        desc="   🔄 Partial bar channels",
                        unit="tf",
                        ncols=100,
                        mininterval=0.5,
                        bar_format="{l_bar}{bar:30}{r_bar}  {postfix}"
                    )
                ))
            else:
                # Sequential processing (fallback)
                print(f"   ℹ️  Sequential processing (parallel disabled in config)")
                print(f"   ⏱️  Estimated time: ~15-20 minutes")

                import gc as _gc
                all_results = []
                for symbol in ['tsla', 'spy']:
                    for tf_name, tf_rule in timeframes.items():
                        print(f"   Processing {symbol}_{tf_name}...")
                        result = calculate_all_channel_features_vectorized(
                            df, symbol, tf_name, tf_rule,
                            windows=config.CHANNEL_WINDOW_SIZES,
                            show_progress=True
                        )
                        all_results.append(result)
                        _gc.collect()

            channel_features = pd.concat(all_results, axis=1)
            del all_results
            import gc as _gc2
            _gc2.collect()

            # Save to cache
            if use_cache:
                print(f"   💾 Saving to cache: {cache_file.name}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(channel_features, f)

            return channel_features

        # Original approach (for live mode or use_partial_bars=False)
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
                                    # v5.6: 31 features per window (removed projected_high/low/center)
                                    for feat in ['position', 'upper_dist', 'lower_dist',
                                                'close_slope', 'close_slope_pct', 'high_slope', 'high_slope_pct',
                                                'low_slope', 'low_slope_pct', 'close_r_squared', 'high_r_squared',
                                                'low_r_squared', 'r_squared_avg', 'channel_width_pct',
                                                'slope_convergence', 'stability', 'ping_pongs',
                                                'ping_pongs_0_5pct', 'ping_pongs_1_0pct', 'ping_pongs_3_0pct',
                                                'complete_cycles', 'complete_cycles_0_5pct', 'complete_cycles_1_0pct', 'complete_cycles_3_0pct',
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

                                # v5.6: Removed projection storage - now calculated at inference from learned duration

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

        Memory: 108 monthly × 31 metrics × 14 windows × 2 symbols = ~94K values = 376 KB
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

    def _extract_rsi_features(self, df: pd.DataFrame, multi_res_data: dict = None,
                               use_partial_bars: bool = True) -> pd.DataFrame:
        """
        Extract RSI features for multiple timeframes.
        NOW PROCESSES BOTH TSLA AND SPY (v3.4)
        Returns DataFrame with 66 columns (33 TSLA + 33 SPY).
        Supports HYBRID mode for live predictions (uses multi-resolution data).

        v5.4: When use_partial_bars=True, RSI for coarser TFs includes in-progress data.
        At Monday 9:30am, weekly RSI includes [last N weeks] + [Monday's partial data].
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

        # v5.4: Import partial bars module if needed
        if use_partial_bars and not is_live_mode:
            from .partial_bars import compute_partial_bars, TIMEFRAME_PERIOD_RULES

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
                        # v5.4: Use partial bars for training mode on coarser TFs
                        if use_partial_bars and not is_live_mode and tf_name not in ['5min']:
                            # Calculate RSI with partial bars at 5min resolution
                            rsi_values = self._calculate_rsi_with_partial_bars(
                                symbol_df, resampled, tf_name, num_rows,
                                compute_partial_bars, TIMEFRAME_PERIOD_RULES
                            )
                        else:
                            # v5.3.3 approach: Calculate RSI on complete bars, ffill to 5min
                            rsi_series = self.rsi_calc.calculate_rsi(resampled)

                            # Map RSI values back to original df timestamps using ffill
                            rsi_aligned = rsi_series.reindex(df.index, method='ffill')
                            rsi_aligned = rsi_aligned.bfill().fillna(50.0)
                            rsi_values = rsi_aligned.values

                        rsi_features[f'{prefix}_{tf_name}'] = rsi_values
                        rsi_features[f'{prefix}_{tf_name}_oversold'] = (rsi_values < 30).astype(float)
                        rsi_features[f'{prefix}_{tf_name}_overbought'] = (rsi_values > 70).astype(float)

                    except Exception as e:
                        # Fill with defaults
                        rsi_features[f'{prefix}_{tf_name}'] = np.full(num_rows, 50.0)
                        rsi_features[f'{prefix}_{tf_name}_oversold'] = np.zeros(num_rows)
                        rsi_features[f'{prefix}_{tf_name}_overbought'] = np.zeros(num_rows)

                    pbar.update(1)

        return pd.DataFrame(rsi_features, index=df.index)

    def _calculate_rsi_with_partial_bars(self, symbol_df: pd.DataFrame, resampled: pd.DataFrame,
                                          tf_name: str, num_rows: int,
                                          compute_partial_bars, TIMEFRAME_PERIOD_RULES) -> np.ndarray:
        """
        Calculate RSI at 5min resolution including partial TF bar data.

        At each 5min bar, the RSI is computed using:
        - All complete TF bars before the current period
        - The partial bar (in-progress data for current TF period)

        This gives "live" RSI that evolves within each TF period.
        """
        rsi_period = self.rsi_calc.period  # Usually 14

        # Compute partial bar state at each 5min timestamp
        partial_state = compute_partial_bars(symbol_df, tf_name)

        # Complete bar closes and timestamps
        complete_closes = resampled['close'].values
        tf_timestamps = resampled.index
        n_complete = len(complete_closes)

        # Output array
        rsi_values = np.full(num_rows, 50.0, dtype=np.float32)

        # Get period mapping for 5min bars
        period_rule = TIMEFRAME_PERIOD_RULES.get(tf_name, tf_name)
        periods = symbol_df.index.to_period(period_rule)

        # OPTIMIZATION: Use pd.factorize to get sequential codes (0, 1, 2, ...) that map to unique periods
        # This is O(n) instead of O(n²) for the period mask lookup
        period_codes, unique_periods = pd.factorize(periods)

        # OPTIMIZATION: Precompute period-to-indices mapping ONCE (O(n) instead of O(n²))
        period_to_indices = {}
        for i, code in enumerate(period_codes):
            if code not in period_to_indices:
                period_to_indices[code] = []
            period_to_indices[code].append(i)
        # Convert lists to numpy arrays for efficiency
        period_to_indices = {k: np.array(v) for k, v in period_to_indices.items()}

        # Convert tf_timestamps to numpy for binary search
        tf_timestamps_ns = tf_timestamps.view(np.int64) if hasattr(tf_timestamps, 'view') else tf_timestamps.astype(np.int64)

        # Process by TF period using precomputed mapping
        for period_idx in range(len(unique_periods)):
            # Use precomputed mapping - O(1) lookup instead of O(n) mask
            bar_indices = period_to_indices.get(period_idx)
            if bar_indices is None or len(bar_indices) == 0:
                continue

            # Find complete TF bars before this period using binary search (O(log n) instead of O(n))
            period_start_ts = symbol_df.index[bar_indices[0]]
            period_start_ns = period_start_ts.value  # nanoseconds
            n_complete_before = np.searchsorted(tf_timestamps_ns, period_start_ns, side='left')

            # Need at least RSI period + 1 bars for valid calculation
            if n_complete_before < rsi_period + 1:
                # Not enough data - leave as default 50
                continue

            # Get historical closes (complete bars before this period)
            hist_closes = complete_closes[:n_complete_before]

            # Precompute gains/losses for historical bars
            hist_deltas = np.diff(hist_closes)
            hist_gains = np.maximum(hist_deltas, 0)
            hist_losses = np.maximum(-hist_deltas, 0)

            # Get the last (rsi_period-1) gains/losses from history
            recent_gains = hist_gains[-(rsi_period-1):] if len(hist_gains) >= rsi_period-1 else hist_gains
            recent_losses = hist_losses[-(rsi_period-1):] if len(hist_losses) >= rsi_period-1 else hist_losses

            # VECTORIZED: Compute RSI for all bars in this period at once
            partial_closes = partial_state.partial_close[bar_indices]
            last_complete_close = hist_closes[-1]

            # Delta from last complete bar to each partial bar
            delta_partials = partial_closes - last_complete_close
            gain_partials = np.maximum(delta_partials, 0)
            loss_partials = np.maximum(-delta_partials, 0)

            # Sum of recent gains/losses (constant for this period)
            sum_recent_gains = np.sum(recent_gains)
            sum_recent_losses = np.sum(recent_losses)

            # Average gains/losses for each bar (vectorized)
            avg_gains = (sum_recent_gains + gain_partials) / rsi_period
            avg_losses = (sum_recent_losses + loss_partials) / rsi_period

            # Calculate RSI (vectorized with safe division)
            # RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = np.where(avg_losses > 0, avg_gains / avg_losses, np.inf)
                rsi_vals = np.where(avg_losses == 0, 100.0,
                           np.where(avg_gains == 0, 0.0,
                           100 - (100 / (1 + rs))))

            rsi_values[bar_indices] = rsi_vals.astype(np.float32)

        return rsi_values

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

        # Volatility regime - TSLA (uses PAST volatility only - NO LEAKAGE) - Use features_df
        current_vol_10 = features_df['tsla_volatility_10']  # From base features
        historical_avg_vol = current_vol_10.rolling(200, min_periods=20).mean()  # Past 200 bars

        breakdown_features['is_volatile_now'] = (
            current_vol_10 > historical_avg_vol * 1.5
        ).fillna(0).astype(float)

        # Volatility regime - SPY (v5.8: distinguish TSLA-specific vs market-wide volatility)
        spy_vol_10 = features_df['spy_volatility_10']  # From base features
        spy_historical_avg_vol = spy_vol_10.rolling(200, min_periods=20).mean()

        breakdown_features['spy_is_volatile_now'] = (
            spy_vol_10 > spy_historical_avg_vol * 1.5
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
        # v5.3.3 fix: Use tsla_volume_ratio (which is available) instead of tsla_volume (which isn't)
        # tsla_volume_ratio = current_volume / 20-day rolling mean, so > 1 means above average
        if 'tsla_volume_ratio' in resampled_df.columns and len(resampled_df) >= 10:
            vol_ratio = resampled_df['tsla_volume_ratio']
            # Volume surge = how much current volume_ratio exceeds recent average ratio
            recent_avg_ratio = vol_ratio.rolling(5, min_periods=1).mean()
            historical_avg_ratio = vol_ratio.rolling(10, min_periods=3).mean().shift(3)
            # Surge is relative change in volume_ratio
            volume_surge = ((recent_avg_ratio - historical_avg_ratio) / (historical_avg_ratio + 1e-8)).fillna(0)
            breakdown_features['tsla_volume_surge'] = np.asarray(volume_surge)
        elif 'tsla_volume' in resampled_df.columns and len(resampled_df) >= 20:
            # Fallback to raw volume if available
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

        # 9. Volatility regime flag - TSLA
        if 'tsla_close' in resampled_df.columns and len(resampled_df) >= 50:
            tsla_close = resampled_df['tsla_close']
            current_vol = tsla_close.pct_change().rolling(10, min_periods=1).std()
            historical_vol = current_vol.rolling(50, min_periods=10).mean()
            is_volatile = (current_vol > 1.5 * historical_vol).fillna(False).astype(float)
            breakdown_features['is_volatile_now'] = np.asarray(is_volatile)
        else:
            breakdown_features['is_volatile_now'] = np.zeros(num_rows)

        # 9b. Volatility regime flag - SPY (v5.8)
        if 'spy_close' in resampled_df.columns and len(resampled_df) >= 50:
            spy_close = resampled_df['spy_close']
            spy_vol = spy_close.pct_change().rolling(10, min_periods=1).std()
            spy_hist_vol = spy_vol.rolling(50, min_periods=10).mean()
            spy_volatile = (spy_vol > 1.5 * spy_hist_vol).fillna(False).astype(float)
            breakdown_features['spy_is_volatile_now'] = np.asarray(spy_volatile)
        else:
            breakdown_features['spy_is_volatile_now'] = np.zeros(num_rows)

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

    def _calculate_all_breakdown_at_5min(
        self,
        features_df: pd.DataFrame,  # Full features at 1min resolution (name is legacy)
        raw_df: pd.DataFrame = None,
        events_handler = None
    ) -> pd.DataFrame:
        """
        Calculate breakdown features for ALL TFs (v5.4, fixed in v5.8).

        NOTE: Despite the name, input is at 1min resolution (name is legacy).
        v5.8 fix: Window sizes now correctly account for 1min input.

        Args:
            features_df: Full features DataFrame at 1min resolution
            raw_df: Original OHLCV DataFrame (for event features)
            events_handler: Optional event handler

        Returns:
            DataFrame with all breakdown features at 1min resolution
        """
        all_breakdown = {}
        num_rows = len(features_df)

        # Process each TF's breakdown from channel features (input is at 1min resolution)
        for tf in HIERARCHICAL_TIMEFRAMES:
            # Get adaptive window - convert from native TF bars to 1min bars
            # v5.8 FIX: Input data is at 1min resolution, not 5min!
            # Previous code assumed 5min input, making windows 5x too short.
            native_window = config.ADAPTIVE_WINDOW_BARS_NATIVE.get(tf, 100)
            # 1min bars per native TF bar (not 5min bars!)
            bars_per_tf_1min = {
                '5min': 5, '15min': 15, '30min': 30, '1h': 60,
                '2h': 120, '3h': 180, '4h': 240, 'daily': 390,  # 6.5 hrs * 60
                'weekly': 390*5, 'monthly': 390*22, '3month': 390*66
            }
            window = min(native_window * bars_per_tf_1min.get(tf, 5), num_rows // 4)
            window = max(window, 10)  # Minimum window

            # 1. Duration ratio (current stability vs rolling average)
            stability_col = f'tsla_channel_{tf}_w50_stability'
            if stability_col in features_df.columns:
                stability = features_df[stability_col]
                avg_stability = stability.rolling(window, min_periods=window//4).mean()
                duration_ratio = (stability / (avg_stability + 0.01)).fillna(1.0)
                all_breakdown[f'tsla_channel_duration_ratio_{tf}'] = np.asarray(duration_ratio)
            else:
                all_breakdown[f'tsla_channel_duration_ratio_{tf}'] = np.ones(num_rows)

            # 2. Channel alignment (SPY-TSLA)
            tsla_pos_col = f'tsla_channel_{tf}_w50_position'
            spy_pos_col = f'spy_channel_{tf}_w50_position'
            if tsla_pos_col in features_df.columns and spy_pos_col in features_df.columns:
                tsla_pos = features_df[tsla_pos_col] * 2 - 1
                spy_pos = features_df[spy_pos_col] * 2 - 1
                alignment = tsla_pos * spy_pos
                all_breakdown[f'channel_alignment_spy_tsla_{tf}'] = np.asarray(alignment)
            else:
                all_breakdown[f'channel_alignment_spy_tsla_{tf}'] = np.zeros(num_rows)

            # 3. Time in channel
            for symbol in ['tsla', 'spy']:
                stab_col = f'{symbol}_channel_{tf}_w50_stability'
                if stab_col in features_df.columns:
                    stab = features_df[stab_col]
                    time_in_channel = np.clip(stab * 100, 0, 100)
                    all_breakdown[f'{symbol}_time_in_channel_{tf}'] = np.asarray(time_in_channel)
                else:
                    all_breakdown[f'{symbol}_time_in_channel_{tf}'] = np.zeros(num_rows)

            # 4. Normalized position
            for symbol in ['tsla', 'spy']:
                pos_col = f'{symbol}_channel_{tf}_w50_position'
                if pos_col in features_df.columns:
                    position_norm = features_df[pos_col] * 2 - 1
                    all_breakdown[f'{symbol}_channel_position_norm_{tf}'] = np.asarray(position_norm)
                else:
                    all_breakdown[f'{symbol}_channel_position_norm_{tf}'] = np.zeros(num_rows)

            # 5. RSI divergence
            rsi_col = f'tsla_rsi_{tf}'
            pos_col = f'tsla_channel_{tf}_w50_position'
            if rsi_col in features_df.columns and pos_col in features_df.columns:
                rsi_normalized = features_df[rsi_col] / 100.0
                channel_pos = features_df[pos_col]
                divergence = rsi_normalized - channel_pos
                all_breakdown[f'tsla_rsi_divergence_{tf}'] = np.asarray(divergence)
            else:
                all_breakdown[f'tsla_rsi_divergence_{tf}'] = np.zeros(num_rows)

            # 6. In-channel binary flags
            for symbol in ['tsla', 'spy']:
                stab_col = f'{symbol}_channel_{tf}_w50_stability'
                if stab_col in features_df.columns:
                    stability = features_df[stab_col]
                    in_channel = (stability > 5.0).astype(float)
                    all_breakdown[f'{symbol}_in_channel_{tf}'] = np.asarray(in_channel)
                else:
                    all_breakdown[f'{symbol}_in_channel_{tf}'] = np.zeros(num_rows)

        # Non-TF-specific features (calculated once)
        # Volume surge
        if 'tsla_volume_ratio' in features_df.columns and num_rows >= 10:
            vol_ratio = features_df['tsla_volume_ratio']
            recent_avg = vol_ratio.rolling(5, min_periods=1).mean()
            hist_avg = vol_ratio.rolling(10, min_periods=3).mean().shift(3)
            surge = ((recent_avg - hist_avg) / (hist_avg + 1e-8)).fillna(0)
            all_breakdown['tsla_volume_surge'] = np.asarray(surge)
        else:
            all_breakdown['tsla_volume_surge'] = np.zeros(num_rows)

        # Day of week flags
        all_breakdown['is_monday'] = (features_df.index.dayofweek == 0).astype(float)
        all_breakdown['is_tuesday'] = (features_df.index.dayofweek == 1).astype(float)
        all_breakdown['is_wednesday'] = (features_df.index.dayofweek == 2).astype(float)
        all_breakdown['is_thursday'] = (features_df.index.dayofweek == 3).astype(float)
        all_breakdown['is_friday'] = (features_df.index.dayofweek == 4).astype(float)

        # Market timing flags
        if hasattr(features_df.index, 'hour'):
            hours = features_df.index.hour
            all_breakdown['is_first_hour'] = ((hours >= 9) & (hours < 11)).astype(float)
            all_breakdown['is_last_hour'] = ((hours >= 15) & (hours < 16)).astype(float)
        else:
            all_breakdown['is_first_hour'] = np.zeros(num_rows)
            all_breakdown['is_last_hour'] = np.zeros(num_rows)

        # Volatility regime - TSLA
        if 'tsla_close' in features_df.columns and num_rows >= 50:
            tsla_close = features_df['tsla_close']
            current_vol = tsla_close.pct_change().rolling(10, min_periods=1).std()
            historical_vol = current_vol.rolling(50, min_periods=10).mean()
            is_volatile = (current_vol > 1.5 * historical_vol).fillna(False).astype(float)
            all_breakdown['is_volatile_now'] = np.asarray(is_volatile)
        else:
            all_breakdown['is_volatile_now'] = np.zeros(num_rows)

        # Volatility regime - SPY (v5.8: distinguish TSLA-specific vs market-wide volatility)
        if 'spy_close' in features_df.columns and num_rows >= 50:
            spy_close = features_df['spy_close']
            spy_vol = spy_close.pct_change().rolling(10, min_periods=1).std()
            spy_hist_vol = spy_vol.rolling(50, min_periods=10).mean()
            spy_volatile = (spy_vol > 1.5 * spy_hist_vol).fillna(False).astype(float)
            all_breakdown['spy_is_volatile_now'] = np.asarray(spy_volatile)
        else:
            all_breakdown['spy_is_volatile_now'] = np.zeros(num_rows)

        # Event features (zeros for now - events not critical)
        all_breakdown['is_earnings_week'] = np.zeros(num_rows)
        all_breakdown['days_until_earnings'] = np.zeros(num_rows)
        all_breakdown['days_until_fomc'] = np.zeros(num_rows)
        all_breakdown['is_high_impact_event'] = np.zeros(num_rows)

        return pd.DataFrame(all_breakdown, index=features_df.index)

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

            # v5.9: Generate labels for ALL channel windows (w10-w100)
            # Previously used TIMEFRAME_SEQUENCE_LENGTHS (500 bars for 1h),
            # which didn't match feature windows. Now uses CHANNEL_WINDOW_SIZES
            # to ensure label windows match feature windows.

            # v5.9.1: Allow partial window sets for TFs with limited data
            # Worker function will skip windows that don't fit (line 5462)
            # This allows 3month (44 bars) to generate labels for w10-w30
            min_window = min(config.CHANNEL_WINDOW_SIZES)  # 10
            max_window = max(config.CHANNEL_WINDOW_SIZES)  # 100
            if len(tf_data) < min_window + 10:
                print(f"         ⚠️  Not enough data for {tf} (need {min_window}+ bars, have {len(tf_data)})")
                continue
            elif len(tf_data) < max_window:
                print(f"         ℹ️  {tf}: {len(tf_data)} bars (partial windows - larger windows will be skipped)")

            # Generate labels
            tf_labels = []

            # v5.9: Parallelize timestamp processing for faster label generation
            # Define worker function (must be at module level for pickling)
            def process_timestamp_for_labels(i, tf_data_values, tf_data_index, windows, current_price_col):
                """Process one timestamp across all windows - parallelizable."""
                try:
                    # Reconstruct needed data from arrays (avoid pickling large DataFrames)
                    current_price = tf_data_values[i, current_price_col]
                    current_timestamp = tf_data_index[i]

                    # Future data for this timestamp
                    future_values = tf_data_values[i:]
                    future_highs = future_values[:, 1]  # high column
                    future_lows = future_values[:, 2]   # low column
                    future_closes = future_values[:, 3]  # close column

                    if len(future_highs) > 0:
                        max_gain = max(abs((future_highs.max() - current_price) / current_price * 100),
                                     abs((future_lows.min() - current_price) / current_price * 100))
                    else:
                        max_gain = 0.0

                    label_row = {
                        'timestamp': current_timestamp,
                        'max_gain_pct': float(max_gain),
                    }

                    # Fit channel for each window
                    for window in windows:
                        if i < window:
                            # Not enough history for this window - mark as invalid
                            label_row[f'w{window}_valid'] = 0
                            continue

                        # Get lookback data
                        lookback_closes = tf_data_values[i - window:i, 3]  # close column

                        # Fit linear regression
                        from scipy import stats
                        x = np.arange(window)
                        slope, intercept, r_value, _, _ = stats.linregress(x, lookback_closes)
                        r_squared = r_value ** 2

                        # Calculate residuals
                        predicted = slope * x + intercept
                        residuals = lookback_closes - predicted
                        residual_std = np.std(residuals)

                        if residual_std < 1e-10:
                            label_row[f'w{window}_valid'] = 0
                            continue

                        # Cycle counting (simplified for parallelization)
                        upper_line = predicted + (2.0 * residual_std)
                        lower_line = predicted - (2.0 * residual_std)
                        touches_upper = lookback_closes >= (predicted + 1.5 * residual_std)
                        touches_lower = lookback_closes <= (predicted - 1.5 * residual_std)

                        complete_cycles = 0
                        last_touch = None
                        for j in range(len(lookback_closes)):
                            if touches_upper[j] and last_touch != 'upper':
                                if last_touch == 'lower':
                                    complete_cycles += 1
                                last_touch = 'upper'
                            elif touches_lower[j] and last_touch != 'lower':
                                if last_touch == 'upper':
                                    complete_cycles += 1
                                last_touch = 'lower'

                        is_valid = complete_cycles >= 2 and r_squared > 0.5

                        # v6.0: Enhanced break detection with return-after-break tracking
                        # Instead of stopping at first break, continue scanning to detect:
                        # - Temporary excursions (price returns inside channel)
                        # - True breaks (price stays outside)
                        first_break_bar = None
                        break_direction = 0  # -1=below, 0=none, 1=above
                        returned = False
                        bars_to_return = None
                        total_bars_outside = 0
                        max_consecutive_outside = 0
                        consecutive_outside = 0
                        consecutive_inside_after_break = 0
                        RETURN_THRESHOLD = 3  # Must stay inside for N bars to count as "returned"

                        scan_limit = min(len(future_closes), 500)  # Cap at 500 bars

                        for bar_idx in range(scan_limit):
                            x_pos = window + bar_idx
                            center = slope * x_pos + intercept
                            upper = center + (2.0 * residual_std)
                            lower = center - (2.0 * residual_std)

                            price = future_closes[bar_idx]
                            is_outside = (price > upper) or (price < lower)

                            if is_outside:
                                total_bars_outside += 1
                                consecutive_outside += 1
                                consecutive_inside_after_break = 0
                                max_consecutive_outside = max(max_consecutive_outside, consecutive_outside)

                                if first_break_bar is None:
                                    first_break_bar = bar_idx
                                    break_direction = 1 if price > upper else -1
                            else:
                                consecutive_outside = 0

                                if first_break_bar is not None and not returned:
                                    consecutive_inside_after_break += 1
                                    if consecutive_inside_after_break >= RETURN_THRESHOLD:
                                        returned = True
                                        bars_to_return = bar_idx - first_break_bar

                        # Calculate final duration (accounting for returns)
                        if first_break_bar is None:
                            final_duration = scan_limit  # Never broke - channel lasted entire scan
                        elif returned:
                            final_duration = scan_limit  # Broke but returned - channel still valid
                        else:
                            final_duration = first_break_bar  # Broke and never returned

                        # Backward compatibility: duration_bars = first break (old behavior)
                        duration_bars = first_break_bar if first_break_bar is not None else (scan_limit - 1)

                        # Confidence calculation
                        conf = r_squared * 0.7
                        cycle_bonus = min(complete_cycles / 5.0, 1.0) * 0.3
                        confidence = (conf + cycle_bonus) if is_valid else 0.1

                        # Track price behavior
                        future_bars = min(duration_bars, len(future_closes))
                        price_sequence = []
                        hit_upper = False
                        hit_midline = False
                        hit_lower = False
                        bars_until_hit_upper = future_bars
                        bars_until_hit_midline = future_bars
                        bars_until_hit_lower = future_bars
                        time_near_upper = 0.0
                        time_near_midline = 0.0
                        time_near_lower = 0.0

                        for bar_idx in range(future_bars):
                            if bar_idx >= len(future_closes):
                                break

                            actual_close = future_closes[bar_idx]
                            actual_pct = (actual_close - current_price) / current_price * 100
                            price_sequence.append(actual_pct)

                            # Project bounds
                            x_pos = window + bar_idx
                            center_at_bar = slope * x_pos + intercept
                            upper_at_bar = center_at_bar + (2.0 * residual_std)
                            lower_at_bar = center_at_bar - (2.0 * residual_std)

                            upper_pct = (upper_at_bar - current_price) / current_price * 100
                            midline_pct = (center_at_bar - current_price) / current_price * 100
                            lower_pct = (lower_at_bar - current_price) / current_price * 100

                            # Check proximity
                            range_pct = upper_pct - lower_pct
                            threshold = range_pct * 0.1

                            if abs(actual_pct - upper_pct) < threshold:
                                time_near_upper += 1
                                if not hit_upper:
                                    hit_upper = True
                                    bars_until_hit_upper = bar_idx

                            if abs(actual_pct - midline_pct) < threshold:
                                time_near_midline += 1
                                if not hit_midline:
                                    hit_midline = True
                                    bars_until_hit_midline = bar_idx

                            if abs(actual_pct - lower_pct) < threshold:
                                time_near_lower += 1
                                if not hit_lower:
                                    hit_lower = True
                                    bars_until_hit_lower = bar_idx

                        # Normalize
                        if future_bars > 0:
                            time_near_upper /= future_bars
                            time_near_midline /= future_bars
                            time_near_lower /= future_bars

                        # Save window data
                        label_row[f'w{window}_duration'] = float(duration_bars)
                        label_row[f'w{window}_price_sequence'] = price_sequence
                        label_row[f'w{window}_hit_upper'] = float(hit_upper)
                        label_row[f'w{window}_hit_midline'] = float(hit_midline)
                        label_row[f'w{window}_hit_lower'] = float(hit_lower)
                        label_row[f'w{window}_bars_until_hit_upper'] = float(bars_until_hit_upper)
                        label_row[f'w{window}_bars_until_hit_midline'] = float(bars_until_hit_midline)
                        label_row[f'w{window}_bars_until_hit_lower'] = float(bars_until_hit_lower)
                        label_row[f'w{window}_time_near_upper'] = float(time_near_upper)
                        label_row[f'w{window}_time_near_midline'] = float(time_near_midline)
                        label_row[f'w{window}_time_near_lower'] = float(time_near_lower)
                        label_row[f'w{window}_slope'] = float(slope)
                        label_row[f'w{window}_confidence'] = float(confidence)
                        label_row[f'w{window}_r_squared'] = float(r_squared)
                        label_row[f'w{window}_cycles'] = int(complete_cycles)
                        label_row[f'w{window}_valid'] = int(is_valid)
                        label_row[f'w{window}_width'] = float(residual_std * 4)

                        # v6.0: Return-after-break tracking labels
                        label_row[f'w{window}_first_break_bar'] = float(first_break_bar) if first_break_bar is not None else -1.0
                        label_row[f'w{window}_break_direction'] = int(break_direction)
                        label_row[f'w{window}_returned'] = int(returned)
                        label_row[f'w{window}_bars_to_return'] = float(bars_to_return) if bars_to_return is not None else -1.0
                        label_row[f'w{window}_bars_outside'] = float(total_bars_outside)
                        label_row[f'w{window}_max_consecutive_outside'] = int(max_consecutive_outside)
                        label_row[f'w{window}_final_duration'] = float(final_duration)

                    # Check if any window is valid
                    valid_windows = sum(1 for w in windows if label_row.get(f'w{w}_valid', 0) > 0)
                    if valid_windows > 0:
                        return label_row
                    return None

                except Exception:
                    return None

            # Get number of workers from config
            # v5.9: Label generation is MUCH lighter than channel extraction (500MB vs 15GB per worker)
            # So we can safely use more workers even on limited RAM systems
            max_workers = config.MAX_PARALLEL_WORKERS if hasattr(config, 'MAX_PARALLEL_WORKERS') else 0

            if max_workers > 0:
                # Adjust for label generation (30x lighter than channel extraction)
                # If menu recommended 1 worker for channels (16GB RAM), we can use ~6 for labels
                try:
                    import psutil
                    available_gb = psutil.virtual_memory().available / (1024**3)
                    # Each label worker uses ~500MB, be conservative with 1GB estimate
                    label_safe_workers = max(1, min(int(available_gb / 1.0), max_workers * 6))
                    # Also respect CPU core count
                    import os
                    cpu_cores = os.cpu_count() or 1
                    n_jobs = min(label_safe_workers, cpu_cores)

                    if n_jobs > max_workers:
                        print(f"         💡 Label generation using {n_jobs} workers (vs {max_workers} for channels - labels are lighter)")
                    else:
                        n_jobs = max_workers
                except:
                    n_jobs = max_workers
            else:
                n_jobs = 1

            # Prepare data for parallel processing (convert DataFrame to arrays)
            tf_data_values = tf_data[['tsla_open', 'tsla_high', 'tsla_low', 'tsla_close']].values
            tf_data_index = tf_data.index.values
            close_col_idx = 3  # Close is 4th column

            # v5.9.1: Start from min_window to allow partial windows for small TFs
            # Worker function skips windows that don't fit each timestamp
            timestamps_to_process = range(min_window, len(tf_data) - 1)

            if n_jobs > 1 and len(timestamps_to_process) > 100:
                # Parallel processing
                print(f"         🚀 Using {n_jobs} workers for parallel label generation...")
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
                    delayed(process_timestamp_for_labels)(
                        i, tf_data_values, tf_data_index, config.CHANNEL_WINDOW_SIZES, close_col_idx
                    )
                    for i in tqdm(timestamps_to_process, desc=f"         {tf} labels", leave=False, ncols=100, ascii=True)
                )

                # Filter out None results
                tf_labels = [r for r in results if r is not None]
            else:
                # Single-threaded fallback
                print(f"         💾 Using single-threaded processing (set feature_workers>1 for speedup)")
                tf_labels = []
                for i in tqdm(timestamps_to_process, desc=f"         {tf} labels", leave=False, ncols=100, ascii=True):
                    result = process_timestamp_for_labels(
                        i, tf_data_values, tf_data_index, config.CHANNEL_WINDOW_SIZES, close_col_idx
                    )
                    if result is not None:
                        tf_labels.append(result)

            # Continue with common code (parallel or single-threaded results now in tf_labels)
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

            # v5.9: Stats showing multi-window coverage
            num_timestamps = len(labels_df)
            valid_counts = {}
            for window in config.CHANNEL_WINDOW_SIZES:
                if f'w{window}_valid' in labels_df.columns:
                    valid_counts[window] = labels_df[f'w{window}_valid'].sum()
                else:
                    valid_counts[window] = 0

            total_valid = sum(valid_counts.values())
            avg_valid_per_timestamp = total_valid / num_timestamps if num_timestamps > 0 else 0

            print(f"         ✓ Saved {num_timestamps:,} timestamps")
            print(f"         ✓ Avg valid windows per timestamp: {avg_valid_per_timestamp:.1f}/14")
            print(f"         ✓ Window coverage: w10={valid_counts.get(10,0):,}, w50={valid_counts.get(50,0):,}, w100={valid_counts.get(100,0):,}")

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

    def generate_hierarchical_continuation_labels_5min(
        self,
        features_df: pd.DataFrame,  # 5min features with partial bar channels
        df: pd.DataFrame,           # Raw OHLCV for price lookups
        timeframes: list = None,
        output_dir: Path = None,
        cache_suffix: str = None
    ) -> Dict[str, Path]:
        """
        Generate continuation labels at 5min resolution (v5.4).

        With partial bars, each 5min bar has its own channel that includes
        in-progress TF data. Labels are generated at 5min resolution using
        these evolving channels.

        Key semantic change from TF-resolution labels:
        - OLD: "Will LAST WEEK's complete channel continue?"
        - NEW: "Will the channel I'm currently in (including today's data) continue?"

        Args:
            features_df: Full features at 5min with partial bar channels
            df: Raw OHLCV DataFrame for price lookups
            timeframes: List of timeframes (defaults to HIERARCHICAL_TIMEFRAMES)
            output_dir: Where to save label files
            cache_suffix: Cache key suffix for file naming

        Returns:
            Dict mapping timeframe -> file path of saved labels
        """
        if timeframes is None:
            timeframes = HIERARCHICAL_TIMEFRAMES
        if output_dir is None:
            output_dir = Path('data/feature_cache')
        if cache_suffix is None:
            cache_suffix = f"{FEATURE_VERSION}_{df.index[0].strftime('%Y%m%d')}_{df.index[-1].strftime('%Y%m%d')}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = {}

        print(f"\n   🔄 Generating 5min continuation labels for {len(timeframes)} timeframes (v5.4)...")
        print(f"      Using partial bar channels - labels evolve within each TF period")

        # Get TSLA prices at 5min resolution
        prices = df['tsla_close'].values if 'tsla_close' in df.columns else features_df.get('tsla_close', pd.Series(dtype=float)).values
        highs = df['tsla_high'].values if 'tsla_high' in df.columns else prices
        n_bars = len(features_df)

        for tf in timeframes:
            print(f"\n      Processing {tf} labels at 5min resolution...")

            # Get channel features for this TF
            pos_col = f'tsla_channel_{tf}_w50_position'
            upper_dist_col = f'tsla_channel_{tf}_w50_upper_dist'
            lower_dist_col = f'tsla_channel_{tf}_w50_lower_dist'
            r_squared_col = f'tsla_channel_{tf}_w50_close_r_squared'
            stability_col = f'tsla_channel_{tf}_w50_stability'
            valid_col = f'tsla_channel_{tf}_w50_is_valid'

            # Check if features exist
            if pos_col not in features_df.columns:
                print(f"         ⚠️  Missing channel features for {tf}")
                continue

            positions = features_df[pos_col].values
            upper_dists = features_df.get(upper_dist_col, pd.Series(np.zeros(n_bars))).values
            lower_dists = features_df.get(lower_dist_col, pd.Series(np.zeros(n_bars))).values
            r_squareds = features_df.get(r_squared_col, pd.Series(np.zeros(n_bars))).values
            stabilities = features_df.get(stability_col, pd.Series(np.zeros(n_bars))).values
            is_valids = features_df.get(valid_col, pd.Series(np.zeros(n_bars))).values

            # v5.4.1: Use insufficient_data flag from channel features instead of hardcoded warmup
            # This flag is set by partial_channel_calc_vectorized when there's not enough historical data
            insufficient_data_col = f'tsla_channel_{tf}_w50_insufficient_data'
            insufficient_data = features_df.get(insufficient_data_col, pd.Series(np.ones(n_bars))).values

            # Generate labels for each 5min bar
            # Skip bars where channel data is insufficient (warmup handled by feature extraction)
            tf_labels = []

            for i in tqdm(range(n_bars - 10),
                         desc=f"         {tf} 5min labels", leave=False, ncols=100, ascii=True):
                # Skip bars with insufficient channel data (this replaces hardcoded warmup)
                if insufficient_data[i] > 0.5:
                    continue

                # Get current channel bounds from features
                current_price = prices[i]
                if current_price <= 0 or np.isnan(current_price):
                    continue

                # Reconstruct channel bounds
                upper = current_price * (1 + upper_dists[i] / 100)
                lower = current_price * (1 - lower_dists[i] / 100)

                # Skip if invalid channel
                if upper <= lower or np.isnan(upper) or np.isnan(lower):
                    continue

                # v6.0: Enhanced break detection with return-after-break tracking
                first_break_bar = None
                break_direction = 0  # -1=below, 0=none, 1=above
                returned = False
                bars_to_return = None
                total_bars_outside = 0
                max_consecutive_outside = 0
                consecutive_outside = 0
                consecutive_inside_after_break = 0
                max_gain = 0.0
                RETURN_THRESHOLD = 3

                # Scan up to 500 bars ahead (to limit computation)
                scan_limit = min(500, n_bars - i - 1)

                for j in range(1, scan_limit):
                    future_price = prices[i + j]
                    future_high = highs[i + j]

                    if np.isnan(future_price):
                        continue

                    # Track max gain
                    gain = (future_high - current_price) / current_price * 100
                    max_gain = max(max_gain, abs(gain))

                    # Check if outside bounds
                    is_outside = (future_price > upper) or (future_price < lower)

                    if is_outside:
                        total_bars_outside += 1
                        consecutive_outside += 1
                        consecutive_inside_after_break = 0
                        max_consecutive_outside = max(max_consecutive_outside, consecutive_outside)

                        if first_break_bar is None:
                            first_break_bar = j
                            break_direction = 1 if future_price > upper else -1
                    else:
                        consecutive_outside = 0

                        if first_break_bar is not None and not returned:
                            consecutive_inside_after_break += 1
                            if consecutive_inside_after_break >= RETURN_THRESHOLD:
                                returned = True
                                bars_to_return = j - first_break_bar

                # Calculate final duration (accounting for returns)
                if first_break_bar is None:
                    final_duration = scan_limit
                elif returned:
                    final_duration = scan_limit
                else:
                    final_duration = first_break_bar

                # Backward compatibility
                duration = first_break_bar if first_break_bar is not None else scan_limit

                # Calculate confidence from channel quality
                r_sq = r_squareds[i]
                stability = stabilities[i] / 10.0  # Normalize
                is_valid = is_valids[i]
                confidence = r_sq * 0.6 + stability * 0.3 + is_valid * 0.1
                confidence = min(max(confidence, 0.1), 0.99)

                tf_labels.append({
                    'timestamp': features_df.index[i],
                    'duration_bars': float(duration),
                    'max_gain_pct': float(max_gain),
                    'confidence': float(confidence),
                    'channel_r_squared': float(r_sq),
                    'channel_valid': int(is_valid),
                    'position_at_entry': float(positions[i]),
                    # v6.0: Return-after-break tracking
                    'first_break_bar': float(first_break_bar) if first_break_bar is not None else -1.0,
                    'break_direction': int(break_direction),
                    'returned': int(returned),
                    'bars_to_return': float(bars_to_return) if bars_to_return is not None else -1.0,
                    'bars_outside': float(total_bars_outside),
                    'max_consecutive_outside': int(max_consecutive_outside),
                    'final_duration': float(final_duration),
                })

            if len(tf_labels) == 0:
                print(f"         ⚠️  No valid labels generated for {tf}")
                continue

            # Convert to DataFrame
            labels_df = pd.DataFrame(tf_labels)
            labels_df.set_index('timestamp', inplace=True)

            # Save to file (use _5min suffix to distinguish from TF-resolution labels)
            output_path = output_dir / f"continuation_labels_5min_{tf}_{cache_suffix}.pkl"
            labels_df.to_pickle(output_path)

            saved_files[tf] = output_path

            # Stats
            avg_duration = labels_df['duration_bars'].mean()
            avg_confidence = labels_df['confidence'].mean()
            print(f"         ✓ Saved {len(labels_df):,} 5min labels | avg duration: {avg_duration:.1f} bars | avg conf: {avg_confidence:.2f}")

        print(f"\n   ✓ Generated {len(saved_files)}/{len(timeframes)} timeframe 5min labels")
        return saved_files

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

                # v5.9: Multi-window format - pick best window's duration
                # Find window with highest r_squared
                best_window = None
                best_r2 = -1
                for window in config.CHANNEL_WINDOW_SIZES:
                    if f'w{window}_valid' in row and row[f'w{window}_valid'] > 0:
                        if f'w{window}_r_squared' in row:
                            r2 = row[f'w{window}_r_squared']
                            if r2 > best_r2:
                                best_r2 = r2
                                best_window = window

                # Use best window's duration, or skip if no valid windows
                if best_window is None:
                    continue  # No valid windows for this timestamp

                duration = int(row[f'w{best_window}_duration'])
                current_slope = row[f'w{best_window}_slope']
                current_r_squared = row[f'w{best_window}_r_squared']
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

                # v5.9: Get future window data (use same best window logic)
                future_best_window = None
                future_best_r2 = -1
                for window in config.CHANNEL_WINDOW_SIZES:
                    if f'w{window}_valid' in future_row and future_row[f'w{window}_valid'] > 0:
                        if f'w{window}_r_squared' in future_row:
                            r2 = future_row[f'w{window}_r_squared']
                            if r2 > future_best_r2:
                                future_best_r2 = r2
                                future_best_window = window

                if future_best_window is None:
                    # Future has no valid windows - assume continuation
                    future_slope = current_slope
                    future_direction = current_direction
                    future_r_squared = current_r_squared
                else:
                    future_slope = future_row[f'w{future_best_window}_slope']
                    future_direction = slope_to_direction(future_slope)
                    future_r_squared = future_row[f'w{future_best_window}_r_squared']

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
