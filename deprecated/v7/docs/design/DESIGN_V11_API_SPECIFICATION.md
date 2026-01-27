# v11.0.0 Multi-Window Cache API Specification

## Overview

This document provides detailed API specifications, data flow diagrams, and implementation examples for the v11.0.0 multi-window cache system.

Related: See `DESIGN_V11_MULTI_WINDOW_CACHE.md` for the high-level design rationale.

## Table of Contents

1. [Data Structures](#data-structures)
2. [Feature Extraction API](#feature-extraction-api)
3. [Dataset API](#dataset-api)
4. [Cache Management API](#cache-management-api)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [Usage Examples](#usage-examples)
7. [Performance Considerations](#performance-considerations)

---

## Data Structures

### PerWindowFeatures

```python
@dataclass
class PerWindowFeatures:
    """
    Features that depend on the channel detection window size.

    These features are extracted independently for each window (10, 20, 30, ..., 80).
    Each window produces different channel boundaries, positions, and derived metrics.

    Attributes:
        timestamp: Bar timestamp where features were extracted
        window: Window size used for channel detection (10, 20, 30, 40, 50, 60, 70, 80)
        tsla: TSLA channel features for all 11 timeframes
              Each TF has 35 features (18 base + 10 exit + 2 break + 5 return)
        spy: SPY channel features for all 11 timeframes
             Each TF has 11 features
        cross_containment: TSLA-in-SPY containment for all 11 timeframes
                          Each TF has 10 features

    Feature Dimensions:
        Per TF: 35 (TSLA) + 11 (SPY) + 10 (cross) = 56 features
        Total: 56 × 11 TFs = 616 features

    Example:
        # Features extracted with window=20
        per_win_20 = PerWindowFeatures(
            timestamp=pd.Timestamp('2024-01-15 15:55:00'),
            window=20,
            tsla={'5min': TSLAChannelFeatures(...), '10min': ...},
            spy={'5min': SPYFeatures(...), ...},
            cross_containment={'5min': CrossAssetContainment(...), ...}
        )
    """
    timestamp: pd.Timestamp
    window: int

    # Per-timeframe features (11 TFs each)
    tsla: Dict[str, TSLAChannelFeatures]  # 35 features per TF
    spy: Dict[str, SPYFeatures]  # 11 features per TF
    cross_containment: Dict[str, CrossAssetContainment]  # 10 features per TF


@dataclass
class SharedFeatures:
    """
    Features that are independent of the channel detection window size.

    These features are extracted once per sample and shared across all windows.
    They represent market regime, historical patterns, and event timing.

    Attributes:
        timestamp: Bar timestamp where features were extracted
        vix: VIX regime features (6 features)
        tsla_history: Historical TSLA channel patterns (25 features)
        spy_history: Historical SPY channel patterns (25 features)
        tsla_spy_direction_match: Boolean - are TSLA and SPY moving same direction?
        both_near_upper: Boolean - both TSLA and SPY near upper channel bounds?
        both_near_lower: Boolean - both TSLA and SPY near lower channel bounds?
        events: Optional event calendar features (46 features, zeros if None)
        tsla_window_scores: Multi-window quality scores (8 windows × 5 metrics = 40 features)

    Feature Dimensions:
        Total: 6 + 25 + 25 + 3 + 46 + 40 = 145 features

    Example:
        shared = SharedFeatures(
            timestamp=pd.Timestamp('2024-01-15 15:55:00'),
            vix=VIXFeatures(level=15.3, regime=1, ...),
            tsla_history=ChannelHistoryFeatures(...),
            spy_history=ChannelHistoryFeatures(...),
            tsla_spy_direction_match=True,
            both_near_upper=False,
            both_near_lower=False,
            events=EventFeatures(...),
            tsla_window_scores=np.array([[5, 0.87, 0.82, 0.9, 2.3], ...])
        )
    """
    timestamp: pd.Timestamp

    # Market regime
    vix: VIXFeatures  # 6 features

    # Historical patterns
    tsla_history: ChannelHistoryFeatures  # 25 features
    spy_history: ChannelHistoryFeatures  # 25 features

    # Cross-asset alignment (3 features)
    tsla_spy_direction_match: bool
    both_near_upper: bool
    both_near_lower: bool

    # Event calendar
    events: Optional[EventFeatures]  # 46 features (zeros if None)

    # Multi-window quality
    tsla_window_scores: Optional[np.ndarray]  # (8, 5) = 40 features
```

### Extended ChannelSample

```python
@dataclass
class ChannelSample:
    """
    Complete training sample with multi-window support (v11.0.0).

    This dataclass represents a single point in time where:
    1. Valid channels were detected at multiple window sizes
    2. Features were extracted for each window + shared features
    3. Labels were generated for each window

    v11.0.0 Extensions:
    - per_window_features: Features extracted at each window (10-80)
    - shared_features: Window-independent features (VIX, history, events)

    Backward Compatibility (v10.0.0):
    - features: Still available, constructed from per_window_features[best_window]
    - channel: Best channel (= channels[best_window])
    - labels: Best window labels (= labels_per_window[best_window])

    Attributes:
        timestamp: When this channel ends (last bar of detection window)
        channel_end_idx: Index in full dataset where channel ends

        # Best window results (backward compatible with v10)
        channel: The best detected Channel object
        features: FullFeatures for best window
        labels: Labels for best window (Dict[tf_name -> ChannelLabels])

        # Multi-window channels (v10.0.0+)
        channels: All detected channels {window_size -> Channel}
        best_window: Selected best window size
        labels_per_window: Labels for each window {window -> {tf -> ChannelLabels}}

        # Multi-window features (v11.0.0+)
        per_window_features: Per-window feature extraction {window -> PerWindowFeatures}
        shared_features: Shared feature extraction (single instance)

    Memory Layout (approximate):
        - channels: ~1 KB per window × 8 = 8 KB
        - labels_per_window: ~2 KB per window × 8 = 16 KB
        - per_window_features: ~2.5 KB per window × 8 = 20 KB
        - shared_features: ~1 KB
        Total: ~45 KB per sample

    Example:
        sample = ChannelSample(
            timestamp=pd.Timestamp('2024-01-15 15:55:00'),
            channel_end_idx=12345,
            channel=channels[20],  # Best window was 20
            features=FullFeatures.from_split_features(per_window_features[20], shared),
            labels=labels_per_window[20],
            channels={10: Channel(...), 20: Channel(...), ...},
            best_window=20,
            labels_per_window={10: {...}, 20: {...}, ...},
            per_window_features={10: PerWindowFeatures(...), 20: ..., ...},
            shared_features=SharedFeatures(...)
        )
    """
    # Core identification
    timestamp: pd.Timestamp
    channel_end_idx: int

    # Best window results (v10.0.0 backward compatible)
    channel: Channel
    features: FullFeatures
    labels: Dict[str, ChannelLabels]

    # Multi-window detection (v10.0.0)
    channels: Dict[int, Channel] = None
    best_window: int = None
    labels_per_window: Dict[int, Dict[str, ChannelLabels]] = None

    # Multi-window features (v11.0.0)
    per_window_features: Dict[int, PerWindowFeatures] = None
    shared_features: SharedFeatures = None

    # Helper methods
    def get_features_for_window(self, window: int) -> Optional[FullFeatures]:
        """
        Reconstruct complete FullFeatures for a specific window.

        This combines per_window_features[window] + shared_features to create
        a backward-compatible FullFeatures object.

        Args:
            window: Window size (must be in STANDARD_WINDOWS)

        Returns:
            FullFeatures object, or None if window not available

        Example:
            # Get features for window=30
            features_30 = sample.get_features_for_window(30)
            if features_30 is not None:
                print(f"TSLA 5min position: {features_30.tsla['5min'].position}")
        """
        if self.per_window_features is None or window not in self.per_window_features:
            return None

        return FullFeatures.from_split_features(
            self.per_window_features[window],
            self.shared_features
        )

    def get_all_window_features(self) -> Dict[int, FullFeatures]:
        """
        Get complete FullFeatures for all available windows.

        Returns:
            Dict mapping window_size -> FullFeatures

        Example:
            all_features = sample.get_all_window_features()
            for window, features in all_features.items():
                print(f"Window {window}: position = {features.tsla['5min'].position}")
        """
        if self.per_window_features is None:
            return {}

        return {
            window: FullFeatures.from_split_features(per_win, self.shared_features)
            for window, per_win in self.per_window_features.items()
        }

    def get_window_count(self) -> int:
        """Return number of windows with extracted features."""
        return len(self.per_window_features) if self.per_window_features else 0

    def has_multi_window_features(self) -> bool:
        """Check if this sample has v11 multi-window features."""
        return self.per_window_features is not None and len(self.per_window_features) > 1
```

---

## Feature Extraction API

### extract_full_features_multi_window()

```python
def extract_full_features_multi_window(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    windows: List[int] = STANDARD_WINDOWS,
    include_history: bool = True,
    lookforward_bars: int = 200,
    events_handler: Optional[EventsHandler] = None,
    cache_shared: bool = True
) -> Tuple[Dict[int, PerWindowFeatures], SharedFeatures]:
    """
    Extract features for all windows + shared features.

    This is the main v11.0.0 feature extraction function, replacing
    extract_full_features() for multi-window cache generation.

    Args:
        tsla_df: TSLA 5min OHLCV data (must have columns: open, high, low, close, volume)
        spy_df: SPY 5min OHLCV data (same schema as tsla_df)
        vix_df: VIX daily OHLCV data (columns: open, high, low, close)
        windows: List of window sizes to extract (default: STANDARD_WINDOWS)
        include_history: If True, scan historical channels (slower but more features)
        lookforward_bars: Bars to look forward for exit tracking features
        events_handler: Optional EventsHandler for calendar features
        cache_shared: If True, cache shared feature extraction within this call

    Returns:
        Tuple of (per_window_features, shared_features)
        - per_window_features: Dict[int, PerWindowFeatures] mapping window -> features
        - shared_features: SharedFeatures instance (single, shared across all windows)

    Raises:
        ValueError: If input dataframes are invalid or insufficient
        IndexError: If insufficient data for smallest window

    Performance:
        - With 8 windows, ~8x slower than single-window extraction
        - Shared features cached per call (extracted once, reused 8 times)
        - Per-window extraction parallelizable (future optimization)

    Example:
        # Extract features for all standard windows
        per_win, shared = extract_full_features_multi_window(
            tsla_df, spy_df, vix_df,
            windows=STANDARD_WINDOWS,
            include_history=True,
            events_handler=my_events
        )

        # Access features for window=20
        win20_tsla_5min = per_win[20].tsla['5min']
        print(f"Position: {win20_tsla_5min.position}")

        # Access shared VIX features
        print(f"VIX regime: {shared.vix.regime}")
    """
    timestamp = tsla_df.index[-1]

    # Phase 1: Extract shared features once
    shared = extract_shared_features(
        tsla_df, spy_df, vix_df,
        include_history=include_history,
        events_handler=events_handler
    )

    # Phase 2: Extract per-window features for each window
    per_window = {}

    for window in windows:
        try:
            per_win = extract_per_window_features(
                tsla_df, spy_df, window, timestamp,
                lookforward_bars=lookforward_bars
            )
            per_window[window] = per_win
        except (ValueError, IndexError) as e:
            # Skip windows with insufficient data
            # Log first occurrence
            if len(per_window) == 0:
                import warnings
                warnings.warn(f"Window {window} extraction failed: {e}")
            continue

    if not per_window:
        raise ValueError("No valid windows could be extracted")

    return per_window, shared


def extract_shared_features(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    include_history: bool = True,
    events_handler: Optional[EventsHandler] = None
) -> SharedFeatures:
    """
    Extract window-independent features.

    These features are the same regardless of channel detection window size:
    - VIX regime (market volatility state)
    - Historical channel patterns (past behavior)
    - Event calendar (upcoming/recent events)
    - Multi-window quality scores (all 8 windows on TSLA)

    Args:
        tsla_df: TSLA 5min OHLCV data
        spy_df: SPY 5min OHLCV data
        vix_df: VIX daily OHLCV data
        include_history: If True, scan historical channels (adds 50 features)
        events_handler: Optional EventsHandler for calendar features (adds 46 features)

    Returns:
        SharedFeatures instance

    Performance:
        - Without history: ~5ms
        - With history: ~50ms (due to historical channel scanning)

    Example:
        shared = extract_shared_features(tsla_df, spy_df, vix_df, include_history=True)
        print(f"VIX level: {shared.vix.level}")
        print(f"Last 5 TSLA directions: {shared.tsla_history.last_n_directions}")
    """
    timestamp = tsla_df.index[-1]

    # VIX regime features
    cross_asset = extract_all_cross_asset_features(tsla_df, spy_df, vix_df, window=20)
    vix_features = cross_asset['vix']

    # Historical channel patterns
    if include_history:
        tsla_records = scan_channel_history(tsla_df, window=20, max_channels=10)
        tsla_history = extract_history_features(tsla_records)

        spy_records = scan_channel_history(spy_df, window=20, max_channels=10)
        spy_history = extract_history_features(spy_records)
    else:
        # Use defaults if history disabled
        tsla_history = ChannelHistoryFeatures(
            last_n_directions=[1] * 5,
            last_n_durations=[50] * 5,
            last_n_break_dirs=[1] * 5,
            avg_duration=50.0,
            direction_streak=0,
            bear_count_last_5=0,
            bull_count_last_5=0,
            sideways_count_last_5=0,
            avg_rsi_at_upper_bounce=50.0,
            avg_rsi_at_lower_bounce=50.0,
            rsi_at_last_break=50.0,
            break_up_after_bear_pct=0.5,
            break_down_after_bull_pct=0.5,
        )
        spy_history = tsla_history  # Same defaults

    # Event calendar features
    event_features = None
    if events_handler is not None:
        try:
            event_features = extract_event_features(timestamp, events_handler, tsla_df)
        except Exception as e:
            import warnings
            warnings.warn(f"Event extraction failed: {e}")

    # Multi-window quality scores (detect at all 8 windows)
    try:
        multi_window_channels = detect_channels_multi_window(tsla_df, windows=STANDARD_WINDOWS)
        window_scores = extract_multi_window_scores(multi_window_channels)
    except (ValueError, IndexError):
        window_scores = np.zeros((len(STANDARD_WINDOWS), NUM_WINDOW_METRICS), dtype=np.float32)

    # Alignment features (use primary 5min TF with standard window=20)
    # Note: These are *technically* window-dependent but we standardize on window=20
    # for simplicity. The impact is minimal since alignment is a high-level metric.
    primary_channel_tsla = detect_channel(tsla_df, window=20)
    primary_channel_spy = detect_channel(spy_df, window=20)

    tsla_dir = int(primary_channel_tsla.direction) if primary_channel_tsla.valid else 1
    spy_dir = int(primary_channel_spy.direction) if primary_channel_spy.valid else 1
    tsla_pos = primary_channel_tsla.position_at() if primary_channel_tsla.valid else 0.5
    spy_pos = primary_channel_spy.position_at() if primary_channel_spy.valid else 0.5

    return SharedFeatures(
        timestamp=timestamp,
        vix=vix_features,
        tsla_history=tsla_history,
        spy_history=spy_history,
        tsla_spy_direction_match=(tsla_dir == spy_dir),
        both_near_upper=(tsla_pos > 0.8 and spy_pos > 0.8),
        both_near_lower=(tsla_pos < 0.2 and spy_pos < 0.2),
        events=event_features,
        tsla_window_scores=window_scores,
    )


def extract_per_window_features(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    window: int,
    timestamp: pd.Timestamp,
    lookforward_bars: int = 200
) -> PerWindowFeatures:
    """
    Extract window-dependent features for a specific window size.

    This extracts per-TF channel features (TSLA, SPY, cross-containment) using
    the specified window size for channel detection.

    Args:
        tsla_df: TSLA 5min OHLCV data
        spy_df: SPY 5min OHLCV data
        window: Window size for channel detection
        timestamp: Current timestamp
        lookforward_bars: Bars to look forward for exit tracking

    Returns:
        PerWindowFeatures instance

    Performance:
        - ~40ms per window (for all 11 timeframes)

    Example:
        # Extract features using window=20
        per_win = extract_per_window_features(tsla_df, spy_df, window=20, timestamp=now)
        print(f"TSLA 5min position: {per_win.tsla['5min'].position}")
    """
    # Resample cache to avoid redundant resampling
    resample_cache = {}

    def get_resampled(df, tf):
        if tf not in resample_cache:
            resample_cache[tf] = resample_ohlc(df, tf) if tf != '5min' else df
        return resample_cache[tf]

    # Extract TSLA features at all timeframes using this window
    tsla_features = {}
    tsla_channels_dict = {}

    # First pass: detect channels at all TFs
    for tf in TIMEFRAMES:
        df_tf = get_resampled(tsla_df, tf)
        try:
            tsla_channels_dict[tf] = detect_channel(df_tf, window=window)
        except (ValueError, IndexError):
            pass

    # Second pass: extract features with longer TF context
    for tf in TIMEFRAMES:
        df_tf = get_resampled(tsla_df, tf)
        try:
            longer_tfs = get_longer_timeframes(tf)
            longer_channels = {ltf: tsla_channels_dict.get(ltf) for ltf in longer_tfs if ltf in tsla_channels_dict}

            tsla_features[tf] = extract_tsla_channel_features(
                df_tf, tf, window,
                longer_tf_channels=longer_channels,
                lookforward_bars=lookforward_bars
            )
        except (ValueError, IndexError):
            pass

    # Extract SPY features at all timeframes
    spy_features = {}
    for tf in TIMEFRAMES:
        df_tf = get_resampled(spy_df, tf)
        try:
            spy_features[tf] = extract_spy_features(df_tf, window, tf)
        except (ValueError, IndexError):
            pass

    # Extract cross-asset containment
    cross_asset = extract_all_cross_asset_features(tsla_df, spy_df, None, window)

    return PerWindowFeatures(
        timestamp=timestamp,
        window=window,
        tsla=tsla_features,
        spy=spy_features,
        cross_containment=cross_asset['cross_containment'],
    )
```

---

## Dataset API

### ChannelDataset Updates

```python
class ChannelDataset(Dataset):
    """
    PyTorch Dataset with v11.0.0 multi-window support.

    Changes in v11.0.0:
    - __getitem__ returns multi-window features: [num_windows, num_features]
    - Backward compatible with v10 caches (single window only)
    - Window selection can be learned by model (all windows available)
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a training sample.

        Returns:
            Tuple of (features_tensor, labels_dict)

            features_tensor shape:
            - v11 (multi-window): [num_windows, num_features] = [8, 761]
            - v10 (single-window): [1, num_features] = [1, 761]

            labels_dict keys:
            - 'duration': [11] - duration in native TF bars
            - 'direction': [11] - break direction per TF
            - 'next_channel': [11] - next channel direction per TF
            - 'trigger_tf': [11] - which TF triggered break
            - 'duration_valid': [11] - validity mask
            - 'direction_valid': [11] - validity mask
            - 'next_channel_valid': [11] - validity mask
            - 'trigger_tf_valid': [11] - validity mask
            - 'window_scores': [8, 3] - per-window quality scores
            - 'window_valid': [8] - which windows are valid
            - 'best_window': scalar - index of best window
        """
        sample = self.samples[idx]

        # v11.0.0: Multi-window features
        if sample.per_window_features is not None and len(sample.per_window_features) > 1:
            features_tensor = self._get_multi_window_features(sample)
        else:
            # v10.0.0 backward compatibility: single window
            features_tensor = self._get_single_window_features(sample)

        # Labels (unchanged from v10)
        labels_dict = self._extract_labels(sample)

        return features_tensor, labels_dict

    def _get_multi_window_features(self, sample: ChannelSample) -> torch.Tensor:
        """
        Convert multi-window features to model input tensor.

        Process:
        1. For each window in STANDARD_WINDOWS:
           a. Combine per_window_features[window] + shared_features
           b. Convert to features_dict via features_to_tensor_dict()
           c. Concatenate to single array via concatenate_features_dict()
        2. Stack all windows into [8, 761] tensor
        3. Fill missing windows with zeros

        Returns:
            Tensor of shape [num_windows, num_features] = [8, 761]

        Example output shape:
            tensor([
                [0.5, 0.3, ...],  # Window 10 features (761 total)
                [0.6, 0.4, ...],  # Window 20 features
                ...
                [0.7, 0.5, ...],  # Window 80 features
            ])
        """
        window_tensors = []

        for window in STANDARD_WINDOWS:
            if window in sample.per_window_features:
                # Reconstruct FullFeatures for this window
                full_features = FullFeatures.from_split_features(
                    sample.per_window_features[window],
                    sample.shared_features
                )

                # Convert to dict of arrays
                features_dict = features_to_tensor_dict(full_features)

                # Concatenate in canonical order
                features_array = concatenate_features_dict(features_dict)
                window_tensors.append(features_array)
            else:
                # Missing window - use zeros
                window_tensors.append(np.zeros(TOTAL_FEATURES, dtype=np.float32))

        # Stack into [8, 761] tensor
        features_tensor = torch.tensor(np.stack(window_tensors), dtype=torch.float32)

        return features_tensor

    def _get_single_window_features(self, sample: ChannelSample) -> torch.Tensor:
        """
        Convert v10 single-window features to model input.

        For backward compatibility with v10 caches and v10-migrated v11 caches.

        Returns:
            Tensor of shape [1, num_features] = [1, 761]
            (Wrapped in extra dimension to match v11 multi-window shape)
        """
        features_dict = features_to_tensor_dict(sample.features)
        features_array = concatenate_features_dict(features_dict)

        # Wrap in [1, ...] to match multi-window shape
        features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)

        return features_tensor


def concatenate_features_dict(features_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate feature dict into single array using canonical ordering.

    Uses FEATURE_ORDER from feature_ordering.py to ensure consistent ordering.

    Args:
        features_dict: Output from features_to_tensor_dict()
                      Keys: feature group names (tsla_5min, spy_5min, vix, etc.)
                      Values: numpy arrays of feature values

    Returns:
        np.ndarray of shape (TOTAL_FEATURES,) = (761,)
        Features are in canonical order defined by FEATURE_ORDER

    Raises:
        ValueError: If required feature keys are missing

    Example:
        features = FullFeatures(...)
        features_dict = features_to_tensor_dict(features)
        # features_dict = {'tsla_5min': array([...]), 'spy_5min': array([...]), ...}

        features_array = concatenate_features_dict(features_dict)
        # features_array.shape = (761,)
    """
    from v7.features.feature_ordering import FEATURE_ORDER, TOTAL_FEATURES

    arrays = []
    for key in FEATURE_ORDER:
        if key not in features_dict:
            raise ValueError(f"Missing required feature key: {key}")
        arrays.append(features_dict[key])

    result = np.concatenate(arrays)

    # Validate total size
    if len(result) != TOTAL_FEATURES:
        raise ValueError(
            f"Feature array size mismatch: expected {TOTAL_FEATURES}, got {len(result)}"
        )

    return result
```

---

## Cache Management API

### Version Detection and Migration

```python
def load_cache_with_auto_migration(
    cache_path: Path,
    auto_migrate: bool = True,
    strict_version: bool = False
) -> List[ChannelSample]:
    """
    Load cache with automatic version detection and migration.

    This is the recommended way to load caches in v11.0.0 code.

    Args:
        cache_path: Path to cache file (.pkl)
        auto_migrate: If True, automatically migrate v10 -> v11
        strict_version: If True, only load exact v11.0.0 caches

    Returns:
        List of ChannelSample objects (v11 format)

    Raises:
        ValueError: If cache version incompatible or migration fails

    Behavior:
        - v11.0.0 cache: Load directly
        - v10.0.0 cache + auto_migrate: Migrate to v11, load migrated version
        - v10.0.0 cache + no auto_migrate: Load as-is (limited features)
        - v9.0.0 or older: Raise error (rebuild required)

    Example:
        # Auto-migrate v10 caches to v11
        samples = load_cache_with_auto_migration(
            Path('data/channels_v10.pkl'),
            auto_migrate=True
        )

        # Check if multi-window features available
        if samples[0].has_multi_window_features():
            print("Full v11 multi-window features available")
        else:
            print("Single-window only (migrated from v10)")
    """
    metadata = get_cache_metadata(cache_path)

    if metadata is None:
        raise ValueError(f"No metadata found for cache: {cache_path}")

    cached_version = metadata.get('cache_version', 'unknown')

    # v11.0.0 - load directly
    if cached_version == 'v11.0.0':
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # v10.0.0 - migrate or load as-is
    if cached_version == 'v10.0.0':
        if strict_version:
            raise ValueError("Strict version mode: only v11.0.0 caches allowed")

        if auto_migrate:
            print(f"Auto-migrating v10.0.0 -> v11.0.0...")
            v11_path = cache_path.with_name(cache_path.stem + '_v11.pkl')

            if v11_path.exists():
                print(f"  Found existing migrated cache: {v11_path}")
            else:
                print(f"  Creating migrated cache: {v11_path}")
                migrate_cache_v10_to_v11(cache_path, v11_path, progress=True)

            with open(v11_path, 'rb') as f:
                return pickle.load(f)
        else:
            print("WARNING: Loading v10.0.0 cache without migration")
            print("         Per-window features not available")
            print("         Set auto_migrate=True to upgrade")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

    # Older versions - require rebuild
    raise ValueError(
        f"Cache version {cached_version} not supported in v11.0.0\\n"
        f"Please rebuild cache using scan_valid_channels() with v11.0.0 code"
    )


def migrate_cache_v10_to_v11(
    v10_cache_path: Path,
    v11_cache_path: Path,
    progress: bool = True
) -> None:
    """
    Migrate v10.0.0 cache to v11.0.0 format.

    IMPORTANT: Migrated caches have LIMITED multi-window features.
    - Only per_window_features[best_window] is populated
    - Other windows are not available
    - For full v11 benefits, rebuild cache from scratch

    Args:
        v10_cache_path: Path to v10 cache file
        v11_cache_path: Path to save v11 cache
        progress: Show progress bar

    Example:
        migrate_cache_v10_to_v11(
            Path('data/channels_v10.pkl'),
            Path('data/channels_v11_migrated.pkl'),
            progress=True
        )
    """
    # Load v10 cache
    print(f"Loading v10 cache from {v10_cache_path}...")
    with open(v10_cache_path, 'rb') as f:
        v10_samples = pickle.load(f)

    print(f"Loaded {len(v10_samples)} samples")
    print("Migrating to v11 format...")

    v11_samples = []
    iterator = tqdm(v10_samples, desc="Migrating samples") if progress else v10_samples

    for sample in iterator:
        # Split existing features into per-window and shared
        per_window_features = {
            sample.best_window: PerWindowFeatures(
                timestamp=sample.features.timestamp,
                window=sample.best_window,
                tsla=sample.features.tsla,
                spy=sample.features.spy,
                cross_containment=sample.features.cross_containment,
            )
        }

        shared_features = SharedFeatures(
            timestamp=sample.features.timestamp,
            vix=sample.features.vix,
            tsla_history=sample.features.tsla_history,
            spy_history=sample.features.spy_history,
            tsla_spy_direction_match=sample.features.tsla_spy_direction_match,
            both_near_upper=sample.features.both_near_upper,
            both_near_lower=sample.features.both_near_lower,
            events=sample.features.events,
            tsla_window_scores=sample.features.tsla_window_scores,
        )

        # Create v11 sample
        v11_sample = ChannelSample(
            timestamp=sample.timestamp,
            channel_end_idx=sample.channel_end_idx,
            channel=sample.channel,
            features=sample.features,
            labels=sample.labels,
            channels=sample.channels,
            best_window=sample.best_window,
            labels_per_window=sample.labels_per_window,
            per_window_features=per_window_features,
            shared_features=shared_features,
        )

        v11_samples.append(v11_sample)

    # Save v11 cache
    print(f"Saving v11 cache to {v11_cache_path}...")
    with open(v11_cache_path, 'wb') as f:
        pickle.dump(v11_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata
    metadata = {
        'cache_version': 'v11.0.0',
        'num_samples': len(v11_samples),
        'migrated_from': 'v10.0.0',
        'full_multi_window': False,  # Only best_window available
        'migration_date': datetime.now().isoformat(),
        'note': 'Migrated cache has limited per-window features. Rebuild for full v11 benefits.',
    }

    meta_path = v11_cache_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Migration complete!")
    print(f"  Samples: {len(v11_samples)}")
    print(f"  Cache: {v11_cache_path}")
    print(f"  Metadata: {meta_path}")
    print()
    print("NOTE: Migrated cache has limited multi-window features.")
    print("      Only best_window features available.")
    print("      For full v11 benefits, rebuild cache from scratch.")
```

---

## Data Flow Diagrams

### v11.0.0 Feature Extraction Flow

```
Input Data (TSLA, SPY, VIX DataFrames)
    |
    v
+-------------------+
| Shared Features   |
| (extract once)    |
+-------------------+
    | - VIX regime
    | - TSLA history
    | - SPY history
    | - Event calendar
    | - Multi-window scores
    | - Alignment
    |
    v
SharedFeatures (145 features)
    |
    |
    |    +------------------+
    +--->| Window 10        |
    |    | (per-window      |
    |    |  extraction)     |
    |    +------------------+
    |         | - TSLA per TF (35×11)
    |         | - SPY per TF (11×11)
    |         | - Cross per TF (10×11)
    |         v
    |    PerWindowFeatures[10] (616 features)
    |
    |    +------------------+
    +--->| Window 20        |
    |    +------------------+
    |         v
    |    PerWindowFeatures[20] (616 features)
    |
    |    ... (windows 30-70)
    |
    |    +------------------+
    +--->| Window 80        |
         +------------------+
              v
         PerWindowFeatures[80] (616 features)

Final Output:
- per_window_features: Dict[int, PerWindowFeatures]  # 8 entries
- shared_features: SharedFeatures  # 1 entry
```

### v11.0.0 Dataset Loading Flow

```
ChannelSample (v11 format)
    |
    | .per_window_features (Dict[int, PerWindowFeatures])
    | .shared_features (SharedFeatures)
    |
    v
For each window in STANDARD_WINDOWS:
    |
    v
+-----------------------------------+
| Combine per-window + shared       |
|                                   |
| FullFeatures.from_split_features( |
|   per_window_features[window],    |
|   shared_features                 |
| )                                 |
+-----------------------------------+
    |
    v
FullFeatures (761 features)
    |
    v
features_to_tensor_dict()
    |
    v
Dict[str, np.ndarray]
    | - 'tsla_5min': [35]
    | - 'spy_5min': [11]
    | - 'cross_5min': [10]
    | - ... (all 11 TFs)
    | - 'vix': [6]
    | - 'tsla_history': [25]
    | - 'spy_history': [25]
    | - 'alignment': [3]
    | - 'events': [46]
    | - 'window_scores': [40]
    |
    v
concatenate_features_dict()
    |
    v
np.ndarray (761,)
    |
    v
Repeat for all 8 windows
    |
    v
Stack windows
    |
    v
torch.Tensor [8, 761]
    |
    v
Return to DataLoader
```

---

## Usage Examples

### Example 1: Generate v11 Cache

```python
from pathlib import Path
import pandas as pd
from v7.training.scanning import scan_valid_channels

# Load data
tsla_df = pd.read_parquet('data/tsla_5min.parquet')
spy_df = pd.read_parquet('data/spy_5min.parquet')
vix_df = pd.read_parquet('data/vix_daily.parquet')

# Scan with v11.0.0 (multi-window extraction)
samples = scan_valid_channels(
    tsla_df, spy_df, vix_df,
    window=20,  # Used for label quality ranking, not feature extraction
    step=10,
    min_cycles=1,
    include_history=True,
    lookforward_bars=200,
    progress=True,
    use_parallel=True,
    num_workers=8
)

print(f"Found {len(samples)} valid samples")
print(f"First sample has {samples[0].get_window_count()} windows")

# Save cache
cache_path = Path('data/channels_v11.pkl')
with open(cache_path, 'wb') as f:
    pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

# Save metadata
metadata = {
    'cache_version': 'v11.0.0',
    'num_samples': len(samples),
    'full_multi_window': True,
    'created_at': datetime.now().isoformat(),
}
with open(cache_path.with_suffix('.json'), 'w') as f:
    json.dump(metadata, f, indent=2)
```

### Example 2: Load and Inspect v11 Cache

```python
from v7.training.dataset import load_cache_with_auto_migration

# Load cache (auto-migrates v10 if needed)
samples = load_cache_with_auto_migration(
    Path('data/channels_v11.pkl'),
    auto_migrate=True
)

# Inspect first sample
sample = samples[0]
print(f"Timestamp: {sample.timestamp}")
print(f"Best window: {sample.best_window}")
print(f"Available windows: {list(sample.per_window_features.keys())}")

# Access features for different windows
for window in [10, 20, 30, 40]:
    features = sample.get_features_for_window(window)
    if features:
        pos = features.tsla['5min'].position
        print(f"Window {window}: TSLA 5min position = {pos:.3f}")

# Access shared features
print(f"VIX regime: {sample.shared_features.vix.regime}")
print(f"TSLA-SPY aligned: {sample.shared_features.tsla_spy_direction_match}")
```

### Example 3: Train Model with v11 Dataset

```python
from torch.utils.data import DataLoader
from v7.training.dataset import ChannelDataset

# Create dataset
dataset = ChannelDataset(
    samples=samples,
    augment=True
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# Training loop
for features, labels in dataloader:
    # features.shape = [batch_size, num_windows, num_features]
    #                = [32, 8, 761]

    # Model can process all windows
    # Option 1: Window selection head
    window_scores = model.window_selector(features)  # [32, 8]
    selected_window = window_scores.argmax(dim=1)    # [32]

    # Option 2: Attention over windows
    attended_features = model.window_attention(features)  # [32, 761]

    # Option 3: Process all windows independently
    all_predictions = model.per_window_predictor(features)  # [32, 8, num_targets]
```

---

## Performance Considerations

### Storage

**Per Sample (v11 full multi-window):**
- Channels: ~1 KB × 8 = 8 KB
- Labels: ~2 KB × 8 = 16 KB
- Features: ~2.5 KB × 8 = 20 KB
- Shared: ~1 KB
- **Total: ~45 KB per sample**

**Full Dataset:**
- 50k samples: ~2.2 GB
- 100k samples: ~4.5 GB
- 200k samples: ~9.0 GB

**v10 Comparison:**
- v10: ~7 KB per sample
- v11: ~45 KB per sample
- **Increase: ~6.4x**

### Memory (Training)

**Batch Size Impact:**
- v10 batch_size=128: ~3.8 MB features
- v11 batch_size=128: ~24 MB features
- **Recommendation: Reduce batch size by ~2x**

### Extraction Time

**Per Sample:**
- Shared features: ~5 ms (without history), ~50 ms (with history)
- Per-window features: ~40 ms per window
- **Total: 50ms + 40ms × 8 = ~370 ms per sample**

**v10 Comparison:**
- v10: ~50 ms per sample
- v11: ~370 ms per sample
- **Increase: ~7.4x**

**Mitigation:**
- Parallelize per-window extraction (future)
- Cache shared features across scanning
- Use faster storage (SSD)

### Loading Time

**Sequential Loading:**
- v10: ~1 second for 100k samples
- v11: ~7 seconds for 100k samples

**Mitigation:**
- Use memory-mapped files (HDF5/zarr)
- Lazy loading (load windows on-demand)
- Compress cache files

---

## Appendix: Feature Dimension Reference

| Component | Per Window | Shared | Total |
|-----------|-----------|--------|-------|
| TSLA (11 TFs × 35) | 385 | - | 385 |
| SPY (11 TFs × 11) | 121 | - | 121 |
| Cross (11 TFs × 10) | 110 | - | 110 |
| VIX | - | 6 | 6 |
| TSLA History | - | 25 | 25 |
| SPY History | - | 25 | 25 |
| Alignment | - | 3 | 3 |
| Events | - | 46 | 46 |
| Window Scores | - | 40 | 40 |
| **Subtotal** | **616** | **145** | **761** |
| **× 8 windows** | **4,928** | **145** | **5,073** |

---

## Next Steps

1. Review this API specification
2. Implement data structures (Phase 1)
3. Unit test feature extraction
4. Integration test with small dataset
5. Performance benchmark
6. Roll out to production pipeline
