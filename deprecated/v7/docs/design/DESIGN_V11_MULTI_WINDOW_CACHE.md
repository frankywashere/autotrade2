# v11.0.0 Multi-Window Cache Design

## Executive Summary

This document defines the data structure design for v11.0.0, which extends the cache to store features from all 8 window sizes (10, 20, 30, 40, 50, 60, 70, 80) for each sample. This enables the model to learn which window size to use based on complete feature context, rather than selecting a window before feature extraction.

**Key Design Decision:** Store complete `FullFeatures` objects per window with per-TF features extracted independently, achieving maximum flexibility with acceptable storage overhead.

## Current v10.0.0 Structure

### ChannelSample (v10.0.0)
```python
@dataclass
class ChannelSample:
    timestamp: pd.Timestamp
    channel_end_idx: int
    channel: Channel  # Best channel (channels[best_window])
    features: FullFeatures  # Features extracted using best_window
    labels: Dict[str, ChannelLabels]  # Labels for best_window
    channels: Dict[int, Channel]  # All detected channels {10: Channel, 20: Channel, ...}
    best_window: int  # Selected window (e.g., 20)
    labels_per_window: Dict[int, Dict[str, ChannelLabels]]  # {window: {tf: ChannelLabels}}
```

### FullFeatures (v10.0.0)
```python
@dataclass
class FullFeatures:
    timestamp: pd.Timestamp

    # Per-timeframe features (11 TFs: 5min, 10min, ..., daily, weekly, monthly)
    tsla: Dict[str, TSLAChannelFeatures]  # 35 features per TF
    spy: Dict[str, SPYFeatures]  # 11 features per TF
    cross_containment: Dict[str, CrossAssetContainment]  # 10 features per TF

    # Shared features (same across all windows)
    vix: VIXFeatures  # 6 features
    tsla_history: ChannelHistoryFeatures  # 25 features
    spy_history: ChannelHistoryFeatures  # 25 features
    tsla_spy_direction_match: bool
    both_near_upper: bool
    both_near_lower: bool
    events: Optional[EventFeatures]  # 46 features
    tsla_window_scores: Optional[np.ndarray]  # (8, 5) - per window scores
```

**Storage:** ~761 features/sample, single window's features only

## Problem Statement

### Current Limitation
In v10.0.0, window selection happens **before** feature extraction:
1. Detect channels at all 8 windows
2. Select best window based on channel quality metrics
3. Extract features using only the best window
4. Generate labels for all windows

This means the model never sees features from alternative windows, limiting its ability to learn optimal window selection from feature patterns.

### v11.0.0 Goal
Enable the model to:
1. See features from **all 8 windows simultaneously**
2. Learn which window provides the best signal for current market conditions
3. Make window selection part of the prediction task (alongside duration, direction, etc.)

## Design Options Analysis

### Option 1: Store 8 Complete FullFeatures Objects (RECOMMENDED)

```python
@dataclass
class ChannelSample:
    # ... existing fields ...
    features_per_window: Dict[int, FullFeatures]  # NEW: {10: FullFeatures, 20: FullFeatures, ...}
```

**Pros:**
- Maximum flexibility - complete feature context per window
- Clean separation - each window's features are independent
- Easy to extend - adding new per-window features is straightforward
- Backward compatible - can still use `features` for single-window workflows

**Cons:**
- ~8x storage for per-TF features (but see optimization below)
- Some feature duplication (shared features like VIX, history)

**Storage Analysis:**
- Per-TF features per window: (35 + 11 + 10) × 11 = 616 features
- Shared features (1 copy): 6 + 25 + 25 + 3 + 46 + 40 = 145 features
- Total per window: 616 + 145 = 761 features
- Total for 8 windows: 616 × 8 + 145 = 5,073 features (~6.7x increase)
- With 100k samples: ~2.0 GB (assuming float32)

**Optimization:** Store shared features once, reference from all windows
```python
@dataclass
class ChannelSample:
    # ... existing fields ...
    features_per_window: Dict[int, PerWindowFeatures]  # Per-window per-TF features
    shared_features: SharedFeatures  # Single copy of VIX, history, events, etc.
```

### Option 2: Store features_dict Per Window (Compact Arrays)

```python
@dataclass
class ChannelSample:
    # ... existing fields ...
    features_dict_per_window: Dict[int, Dict[str, np.ndarray]]  # {window: features_to_tensor_dict()}
```

**Pros:**
- More compact than FullFeatures objects
- Direct numpy arrays ready for model input
- Less object overhead

**Cons:**
- Less semantic - harder to inspect/debug
- Must convert to tensor_dict earlier in pipeline
- Loses structure/type information
- Harder to extend with new feature types

**Storage:** Similar to Option 1 (~5,000 features/sample)

### Option 3: Store Only Per-TF Features Per Window

```python
@dataclass
class PerWindowTFFeatures:
    """Only the features that depend on window size."""
    tsla: Dict[str, TSLAChannelFeatures]  # 35 features per TF
    spy: Dict[str, SPYFeatures]  # 11 features per TF
    cross_containment: Dict[str, CrossAssetContainment]  # 10 features per TF
    # Exclude: VIX, history, events (shared across windows)

@dataclass
class ChannelSample:
    # ... existing fields ...
    per_window_features: Dict[int, PerWindowTFFeatures]  # {10: features, 20: features, ...}
    shared_features: SharedFeatures  # Single copy
```

**Pros:**
- Minimal storage - only store what varies per window
- Clear separation of window-dependent vs shared features
- Most efficient storage (~4,928 + 145 = 5,073 features, same as Option 1 optimized)

**Cons:**
- More complex data structure
- Requires careful handling when converting to model input
- Slight increase in code complexity

## Recommended Design: Option 1 (Optimized)

### Rationale
1. **Clarity:** Complete FullFeatures objects are easier to work with
2. **Flexibility:** Can easily add new per-window features
3. **Backward Compatibility:** Existing code using `features` continues to work
4. **Storage Efficient:** With shared feature optimization, storage is comparable to Option 3
5. **Development Speed:** Minimal refactoring needed

### Implementation Strategy

#### Phase 1: Split FullFeatures into Window-Dependent and Shared

```python
@dataclass
class PerWindowFeatures:
    """Features that depend on window size (extracted per window)."""
    timestamp: pd.Timestamp
    window: int  # Which window was used (10, 20, 30, ...)

    # Per-timeframe features - these vary by window
    tsla: Dict[str, TSLAChannelFeatures]  # 35 features per TF
    spy: Dict[str, SPYFeatures]  # 11 features per TF
    cross_containment: Dict[str, CrossAssetContainment]  # 10 features per TF

    # Per-TF feature count: 56 per TF × 11 TFs = 616 features


@dataclass
class SharedFeatures:
    """Features that are window-independent (same for all windows)."""
    timestamp: pd.Timestamp

    # Market regime features
    vix: VIXFeatures  # 6 features

    # Historical pattern features (based on past channel breaks)
    tsla_history: ChannelHistoryFeatures  # 25 features
    spy_history: ChannelHistoryFeatures  # 25 features

    # Cross-asset alignment
    tsla_spy_direction_match: bool
    both_near_upper: bool
    both_near_lower: bool

    # Event calendar features
    events: Optional[EventFeatures]  # 46 features

    # Multi-window channel quality scores (from all 8 windows)
    tsla_window_scores: Optional[np.ndarray]  # (8, 5) = 40 features

    # Total: 145 features


@dataclass
class FullFeatures:
    """
    Complete feature set for a single bar (BACKWARD COMPATIBLE).

    In v11.0.0, this is constructed from per_window + shared.
    """
    timestamp: pd.Timestamp
    window: int  # NEW in v11: Which window these features use

    # Per-timeframe features
    tsla: Dict[str, TSLAChannelFeatures]
    spy: Dict[str, SPYFeatures]
    cross_containment: Dict[str, CrossAssetContainment]

    # Shared features
    vix: VIXFeatures
    tsla_history: ChannelHistoryFeatures
    spy_history: ChannelHistoryFeatures
    tsla_spy_direction_match: bool
    both_near_upper: bool
    both_near_lower: bool
    events: Optional[EventFeatures]
    tsla_window_scores: Optional[np.ndarray]

    @classmethod
    def from_split_features(
        cls,
        per_window: PerWindowFeatures,
        shared: SharedFeatures
    ) -> 'FullFeatures':
        """
        Construct FullFeatures from split components.
        Enables backward compatibility with v10 code.
        """
        return cls(
            timestamp=per_window.timestamp,
            window=per_window.window,
            tsla=per_window.tsla,
            spy=per_window.spy,
            cross_containment=per_window.cross_containment,
            vix=shared.vix,
            tsla_history=shared.tsla_history,
            spy_history=shared.spy_history,
            tsla_spy_direction_match=shared.tsla_spy_direction_match,
            both_near_upper=shared.both_near_upper,
            both_near_lower=shared.both_near_lower,
            events=shared.events,
            tsla_window_scores=shared.tsla_window_scores,
        )
```

#### Phase 2: Extend ChannelSample for v11.0.0

```python
@dataclass
class ChannelSample:
    """
    A single training sample with multi-window support.

    v11.0.0 Changes:
    - Added per_window_features: Dict[int, PerWindowFeatures]
    - Added shared_features: SharedFeatures
    - features now constructed from per_window_features[best_window] + shared_features

    Backward Compatibility:
    - v10.0.0 caches can be loaded (features, channel, labels present)
    - v11.0.0 caches have additional per_window_features and shared_features
    - Old code using just `features` continues to work
    """
    # Core identification
    timestamp: pd.Timestamp
    channel_end_idx: int

    # Best window results (backward compatible)
    channel: Channel  # Best channel (= channels[best_window])
    features: FullFeatures  # Best window features (= per_window_features[best_window] + shared)
    labels: Dict[str, ChannelLabels]  # Best window labels (= labels_per_window[best_window])

    # Multi-window channel detection (v10.0.0+)
    channels: Dict[int, Channel] = None  # {10: Channel, 20: Channel, ..., 80: Channel}
    best_window: int = None  # Selected best window (e.g., 20)
    labels_per_window: Dict[int, Dict[str, ChannelLabels]] = None

    # NEW in v11.0.0: Multi-window features
    per_window_features: Dict[int, PerWindowFeatures] = None  # {10: features, 20: features, ...}
    shared_features: SharedFeatures = None  # Single copy of window-independent features

    def get_features_for_window(self, window: int) -> Optional[FullFeatures]:
        """
        Get complete FullFeatures for a specific window.

        Combines per_window_features[window] + shared_features.
        Returns None if window not available.
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

        Returns dict mapping window -> FullFeatures.
        """
        if self.per_window_features is None:
            return {}

        return {
            window: FullFeatures.from_split_features(per_win, self.shared_features)
            for window, per_win in self.per_window_features.items()
        }
```

#### Phase 3: Update Feature Extraction

```python
def extract_full_features_multi_window(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    windows: List[int] = STANDARD_WINDOWS,
    include_history: bool = True,
    lookforward_bars: int = 200,
    events_handler: Optional[EventsHandler] = None
) -> Tuple[Dict[int, PerWindowFeatures], SharedFeatures]:
    """
    Extract features for all windows + shared features.

    This replaces extract_full_features() for v11.0.0 cache generation.

    Returns:
        Tuple of (per_window_features, shared_features)
        - per_window_features: Dict[int, PerWindowFeatures] - one per window
        - shared_features: SharedFeatures - single shared instance
    """
    timestamp = tsla_df.index[-1]

    # Extract shared features once (window-independent)
    shared = extract_shared_features(
        tsla_df, spy_df, vix_df,
        include_history=include_history,
        events_handler=events_handler
    )

    # Extract per-window features for each window
    per_window = {}
    for window in windows:
        try:
            per_win = extract_per_window_features(
                tsla_df, spy_df, window, timestamp,
                lookforward_bars=lookforward_bars
            )
            per_window[window] = per_win
        except (ValueError, IndexError):
            # Skip windows with insufficient data
            continue

    return per_window, shared


def extract_shared_features(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    include_history: bool = True,
    events_handler: Optional[EventsHandler] = None
) -> SharedFeatures:
    """Extract window-independent features."""
    # VIX regime
    vix_features = extract_vix_features(vix_df)

    # Channel history (if enabled)
    if include_history:
        tsla_history = extract_history_features(
            scan_channel_history(tsla_df, window=20, max_channels=10)
        )
        spy_history = extract_history_features(
            scan_channel_history(spy_df, window=20, max_channels=10)
        )
    else:
        tsla_history = ChannelHistoryFeatures(...)  # Defaults
        spy_history = ChannelHistoryFeatures(...)

    # Event features
    events = None
    if events_handler is not None:
        events = extract_event_features(tsla_df.index[-1], events_handler, tsla_df)

    # Multi-window scores (detect at all 8 windows on TSLA)
    multi_window_channels = detect_channels_multi_window(tsla_df, windows=STANDARD_WINDOWS)
    window_scores = extract_multi_window_scores(multi_window_channels)

    # Alignment features (extract from primary 5min TF with default window=20)
    # These are technically window-dependent, but we use a standard window for simplicity
    primary_channel_tsla = detect_channel(tsla_df, window=20)
    primary_channel_spy = detect_channel(spy_df, window=20)

    tsla_dir = int(primary_channel_tsla.direction) if primary_channel_tsla.valid else 1
    spy_dir = int(primary_channel_spy.direction) if primary_channel_spy.valid else 1
    tsla_pos = primary_channel_tsla.position_at() if primary_channel_tsla.valid else 0.5
    spy_pos = primary_channel_spy.position_at() if primary_channel_spy.valid else 0.5

    return SharedFeatures(
        timestamp=tsla_df.index[-1],
        vix=vix_features,
        tsla_history=tsla_history,
        spy_history=spy_history,
        tsla_spy_direction_match=(tsla_dir == spy_dir),
        both_near_upper=(tsla_pos > 0.8 and spy_pos > 0.8),
        both_near_lower=(tsla_pos < 0.2 and spy_pos < 0.2),
        events=events,
        tsla_window_scores=window_scores,
    )


def extract_per_window_features(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    window: int,
    timestamp: pd.Timestamp,
    lookforward_bars: int = 200
) -> PerWindowFeatures:
    """Extract window-dependent features for a specific window."""
    # Extract TSLA features at all timeframes using this window
    tsla_features = {}
    tsla_channels_dict = {}

    # First pass: detect channels at all TFs
    for tf in TIMEFRAMES:
        df_tf = resample_ohlc(tsla_df, tf) if tf != '5min' else tsla_df
        try:
            tsla_channels_dict[tf] = detect_channel(df_tf, window=window)
        except (ValueError, IndexError):
            pass

    # Second pass: extract features with longer TF context
    for tf in TIMEFRAMES:
        df_tf = resample_ohlc(tsla_df, tf) if tf != '5min' else tsla_df
        try:
            longer_tfs = get_longer_timeframes(tf)
            longer_channels = {ltf: tsla_channels_dict.get(ltf) for ltf in longer_tfs}

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
        df_tf = resample_ohlc(spy_df, tf) if tf != '5min' else spy_df
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

#### Phase 4: Update Dataset to Convert to Model Input

```python
class ChannelDataset(Dataset):
    """
    PyTorch Dataset with v11.0.0 multi-window support.
    """

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]

        # v11.0.0: Create features tensor from all windows
        if sample.per_window_features is not None:
            features_tensors = self._get_multi_window_features(sample)
        else:
            # v10.0.0 backward compatibility: single window only
            features_tensors = self._get_single_window_features(sample)

        # Labels remain the same
        labels_dict = self._extract_labels(sample)

        return features_tensors, labels_dict

    def _get_multi_window_features(self, sample: ChannelSample) -> torch.Tensor:
        """
        Convert multi-window features to model input tensor.

        Output shape: [num_windows, num_features_per_window]
        where num_features_per_window = 616 (per-TF) + 145 (shared) = 761

        Each window gets: per_window_features[window] + shared_features
        """
        window_tensors = []

        for window in STANDARD_WINDOWS:
            if window in sample.per_window_features:
                # Combine per-window + shared features
                full_features = FullFeatures.from_split_features(
                    sample.per_window_features[window],
                    sample.shared_features
                )
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
        Backward compatibility: convert v10 single-window features.

        Output shape: [1, num_features] for compatibility with v11 model input
        """
        features_dict = features_to_tensor_dict(sample.features)
        features_array = concatenate_features_dict(features_dict)

        # Wrap in [1, ...] to match multi-window shape
        features_tensor = torch.tensor(features_array, dtype=torch.float32).unsqueeze(0)

        return features_tensor


def concatenate_features_dict(features_dict: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Concatenate feature dict into single array using canonical ordering.

    Args:
        features_dict: Output from features_to_tensor_dict()

    Returns:
        np.ndarray of shape (TOTAL_FEATURES,) = (761,)
    """
    arrays = []
    for key in FEATURE_ORDER:
        if key not in features_dict:
            raise ValueError(f"Missing feature key: {key}")
        arrays.append(features_dict[key])

    return np.concatenate(arrays)
```

## Migration Strategy

### v10.0.0 → v11.0.0 Cache Migration

```python
def migrate_cache_v10_to_v11(
    v10_cache_path: Path,
    v11_cache_path: Path,
    progress: bool = True
) -> None:
    """
    Migrate v10 cache to v11 format.

    v10 caches only have features for the best window.
    v11 migration adds empty per_window_features and constructs shared_features.

    NOTE: Migrated caches will NOT have per-window features - they only provide
    backward compatibility. For full v11 benefits, rebuild cache from scratch.
    """
    # Load v10 cache
    with open(v10_cache_path, 'rb') as f:
        v10_samples = pickle.load(f)

    v11_samples = []

    iterator = tqdm(v10_samples) if progress else v10_samples
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
            features=sample.features,  # Keep for backward compat
            labels=sample.labels,
            channels=sample.channels,
            best_window=sample.best_window,
            labels_per_window=sample.labels_per_window,
            per_window_features=per_window_features,  # NEW
            shared_features=shared_features,  # NEW
        )

        v11_samples.append(v11_sample)

    # Save v11 cache
    with open(v11_cache_path, 'wb') as f:
        pickle.dump(v11_samples, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata
    metadata = {
        'cache_version': 'v11.0.0',
        'num_samples': len(v11_samples),
        'migrated_from': 'v10.0.0',
        'full_multi_window': False,  # Migrated caches don't have all windows
        'created_at': datetime.now().isoformat(),
    }

    meta_path = v11_cache_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
```

### Cache Version Detection

```python
CACHE_VERSION = "v11.0.0"
COMPATIBLE_CACHE_VERSIONS = ["v10.0.0", "v9.0.0", "v8.0.0"]

def load_cache_with_migration(
    cache_path: Path,
    auto_migrate: bool = False
) -> List[ChannelSample]:
    """
    Load cache with automatic migration support.

    Args:
        cache_path: Path to cache file
        auto_migrate: If True, automatically migrate older caches to v11

    Returns:
        List of ChannelSample objects (v11 format)
    """
    metadata = get_cache_metadata(cache_path)

    if metadata is None:
        raise ValueError(f"No metadata found for cache: {cache_path}")

    cached_version = metadata.get('cache_version', 'unknown')

    # v11 cache - load directly
    if cached_version == 'v11.0.0':
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    # v10 cache - migrate if requested
    if cached_version == 'v10.0.0':
        if auto_migrate:
            print(f"Auto-migrating v10.0.0 cache to v11.0.0...")
            v11_path = cache_path.with_name(cache_path.stem + '_v11.pkl')
            migrate_cache_v10_to_v11(cache_path, v11_path)
            with open(v11_path, 'rb') as f:
                return pickle.load(f)
        else:
            print("WARNING: Loading v10.0.0 cache. Per-window features not available.")
            print("         Set auto_migrate=True to upgrade, or rebuild cache.")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

    # Older versions
    raise ValueError(
        f"Cache version {cached_version} not supported. "
        f"Please rebuild cache with v11.0.0."
    )
```

## Storage Requirements

### Per Sample
- **v10.0.0:** 761 features (single window)
- **v11.0.0:** ~5,073 features (8 windows × 616 per-TF + 145 shared)
- **Increase:** ~6.7x feature count

### Full Dataset (100k samples)
- **v10.0.0:** 761 × 100k × 4 bytes = ~305 MB
- **v11.0.0:** 5,073 × 100k × 4 bytes = ~2.0 GB
- **Additional:** ~1.7 GB per 100k samples

With typical dataset sizes (50k-200k samples), v11 caches will be 2-8 GB.

## Implementation Checklist

### Phase 1: Data Structures ✓ (Design Complete)
- [ ] Define `PerWindowFeatures` dataclass
- [ ] Define `SharedFeatures` dataclass
- [ ] Update `FullFeatures` with `from_split_features()` classmethod
- [ ] Update `ChannelSample` with new fields
- [ ] Add `get_features_for_window()` helper method

### Phase 2: Feature Extraction
- [ ] Implement `extract_shared_features()`
- [ ] Implement `extract_per_window_features()`
- [ ] Implement `extract_full_features_multi_window()`
- [ ] Update parallel scanner to call new extraction functions
- [ ] Add caching for shared feature extraction

### Phase 3: Dataset Loading
- [ ] Update `ChannelDataset.__getitem__()` for multi-window features
- [ ] Implement `_get_multi_window_features()`
- [ ] Implement `concatenate_features_dict()`
- [ ] Update batch collation functions
- [ ] Add unit tests for tensor shapes

### Phase 4: Cache Management
- [ ] Update `CACHE_VERSION` to "v11.0.0"
- [ ] Implement `migrate_cache_v10_to_v11()`
- [ ] Update `load_cache_with_migration()`
- [ ] Update cache validation functions
- [ ] Add metadata fields for multi-window info

### Phase 5: Testing
- [ ] Unit tests for split feature extraction
- [ ] Unit tests for multi-window tensor construction
- [ ] Integration test: scan + save + load v11 cache
- [ ] Backward compatibility test: load v10 cache in v11 code
- [ ] Migration test: v10 → v11 conversion
- [ ] Performance benchmark: v10 vs v11 loading speed

### Phase 6: Documentation
- [ ] Update feature extraction documentation
- [ ] Update dataset documentation
- [ ] Add migration guide for users
- [ ] Update cache format specification
- [ ] Add examples of multi-window feature access

## Trade-offs Summary

### Pros
1. **Complete Context:** Model sees features from all windows simultaneously
2. **Better Learning:** Can learn optimal window selection from data
3. **Flexibility:** Easy to add new per-window or shared features
4. **Backward Compatible:** v10 code continues to work with v11 caches
5. **Clean Architecture:** Clear separation of window-dependent vs shared features

### Cons
1. **Storage:** ~6.7x increase in cache size (2 GB per 100k samples)
2. **Memory:** Higher memory usage during training (~6.7x per batch)
3. **Extraction Time:** Must extract features at all 8 windows (slower scanning)
4. **Complexity:** More complex data structures and conversion logic

### Mitigation Strategies
1. **Storage:** Use compression (pickle HIGHEST_PROTOCOL, or HDF5/zarr)
2. **Memory:** Use smaller batch sizes or window subsampling during training
3. **Extraction:** Parallelize per-window extraction, cache shared features
4. **Complexity:** Provide helper functions and clear documentation

## Conclusion

The recommended design (Option 1 Optimized) provides the best balance of flexibility, clarity, and efficiency. It enables the model to learn window selection while maintaining backward compatibility and clean code architecture.

The ~6.7x storage increase is acceptable given:
1. Modern storage is cheap (~$0.02/GB for HDD)
2. Enables significant model improvement potential
3. Can be compressed further if needed
4. Allows incremental adoption (can keep using best_window for simple models)

**Next Steps:**
1. Review and approve this design
2. Implement Phase 1 (data structures)
3. Test with small dataset
4. Roll out to full pipeline
