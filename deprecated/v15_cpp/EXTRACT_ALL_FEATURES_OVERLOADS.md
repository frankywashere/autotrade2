# extract_all_features() Overloads: Visual Explanation

## Why Do We Have 3 Versions?

The three overloads serve different use cases in the scanning pipeline, balancing:
- **Performance** (memory efficiency, zero-copy operations)
- **Compatibility** (supporting different data types)
- **Feature completeness** (with or without channel history)

---

## The 3 Overloads

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ OVERLOAD #1: Vector-based with SlimMaps (LEGACY COMPATIBILITY)             │
├─────────────────────────────────────────────────────────────────────────────┤
│ extract_all_features(                                                       │
│     const std::vector<OHLCV>& tsla_5min,     // ◄─ Owns data              │
│     const std::vector<OHLCV>& spy_5min,                                    │
│     const std::vector<OHLCV>& vix_5min,                                    │
│     int64_t timestamp,                                                      │
│     const SlimLabeledChannelMap& tsla_slim_map,                            │
│     const SlimLabeledChannelMap& spy_slim_map,                             │
│     int source_bar_count = -1,                                             │
│     bool include_bar_metadata = true                                       │
│ )                                                                           │
│                                                                             │
│ HISTORY: Empty (all 670 channel history features = 0.0)                    │
│ USAGE:   Pass 3 scanner (uses vector slices)                               │
│ OUTPUT:  14,170 features (14,840 - 670 history features)                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ OVERLOAD #2: DataView with SlimMaps (ZERO-COPY OPTIMIZATION)               │
├─────────────────────────────────────────────────────────────────────────────┤
│ extract_all_features(                                                       │
│     const DataView& tsla_view,               // ◄─ Zero-copy view          │
│     const DataView& spy_view,                                              │
│     const DataView& vix_view,                                              │
│     int64_t timestamp,                                                      │
│     const SlimLabeledChannelMap& tsla_slim_map,                            │
│     const SlimLabeledChannelMap& spy_slim_map,                             │
│     int source_bar_count = -1,                                             │
│     bool include_bar_metadata = true                                       │
│ )                                                                           │
│                                                                             │
│ HISTORY: Empty (delegates to Overload #3 with empty history)               │
│ USAGE:   DEPRECATED - was for intermediate optimization                    │
│ OUTPUT:  14,170 features (14,840 - 670 history features)                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ OVERLOAD #3: DataView + Channel History (FULL FEATURE SET)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ extract_all_features(                                                       │
│     const DataView& tsla_view,               // ◄─ Zero-copy view          │
│     const DataView& spy_view,                                              │
│     const DataView& vix_view,                                              │
│     int64_t timestamp,                                                      │
│     const SlimLabeledChannelMap& tsla_slim_map,                            │
│     const SlimLabeledChannelMap& spy_slim_map,                             │
│     const unordered_map<string, vector<ChannelHistoryEntry>>& tsla_history,│
│     const unordered_map<string, vector<ChannelHistoryEntry>>& spy_history, │
│     int source_bar_count = -1,                                             │
│     bool include_bar_metadata = true                                       │
│ )                                                                           │
│                                                                             │
│ HISTORY: Real (670 channel history features with actual data!)             │
│ USAGE:   Pass 1/2 scanner (main production path)                           │
│ OUTPUT:  14,840 features (COMPLETE FEATURE SET)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Delegation Flow

```
                    ┌──────────────┐
                    │ OVERLOAD #2  │
                    │ (DataView)   │
                    │              │
                    │ Creates      │
                    │ empty        │
                    │ history maps │
                    └──────┬───────┘
                           │
                           │ Delegates to
                           ▼
    ┌──────────────────────────────────────┐
    │        OVERLOAD #3 (FULL)            │
    │     (DataView + History)             │
    │                                      │
    │  • Converts DataView → vectors      │
    │  • Resamples to all TFs             │
    │  • Extracts all features            │
    │  • Uses real or empty history       │
    │  • Returns complete feature map     │
    └──────────────────────────────────────┘
                           ▲
                           │
                           │ Used directly by
                           │
    ┌──────────────────────┴───────┐
    │     OVERLOAD #1              │
    │     (Vector-based)           │
    │                              │
    │  • Used by Pass 3 scanner   │
    │  • No history available     │
    └──────────────────────────────┘
```

---

## Caller Analysis

### Pass 1/2 Scanner (scanner.cpp:1569)
```cpp
auto tf_features = FeatureExtractor::extract_all_features(
    tsla_view,                          // DataView (zero-copy)
    spy_view,
    vix_view,
    sample_timestamp,
    tsla_slim_map,
    spy_slim_map,
    work_item.tsla_history_by_tf,      // ◄─ REAL HISTORY!
    work_item.spy_history_by_tf,       // ◄─ REAL HISTORY!
    static_cast<int>(tsla_view.size()),
    true
);
```
**Uses: Overload #3 (FULL)**
- Zero-copy DataView for memory efficiency
- Pre-computed channel history from Pass 1/2
- **Gets all 14,840 features including 670 history features**

### Pass 3 Scanner (scanner_pass3.cpp:104)
```cpp
auto tf_features = FeatureExtractor::extract_all_features(
    tsla_slice,                         // std::vector<OHLCV>
    spy_slice,
    vix_slice,
    sample_timestamp,
    tsla_slim_map,
    spy_slim_map,
    idx_5min,                          // source_bar_count
    true
);
```
**Uses: Overload #1 (Vector-based)**
- Vector slices (already copied data)
- No history available in Pass 3
- **Gets 14,170 features (history features = 0.0)**

---

## History Feature Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        PASS 1/2 SCANNER                                  │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Channel Detection                                                   │
│     ┌────────────────────────────────────────┐                         │
│     │ For each Timeframe (10 TFs):          │                         │
│     │   • Detect channels (8 windows)       │                         │
│     │   • Label channels (bull/bear/break)  │                         │
│     │   • Store in SlimLabeledChannelMap    │                         │
│     └────────────────────────────────────────┘                         │
│                      │                                                  │
│                      ▼                                                  │
│  2. Channel History Extraction                                          │
│     ┌────────────────────────────────────────┐                         │
│     │ For each TF:                           │                         │
│     │   • Get last 5 channels                │                         │
│     │   • Extract ChannelHistoryEntry        │                         │
│     │   • Store in map<tf_str, vector<...>>  │                         │
│     └────────────────────────────────────────┘                         │
│                      │                                                  │
│                      ▼                                                  │
│  3. Feature Extraction (Overload #3)                                    │
│     ┌────────────────────────────────────────┐                         │
│     │ For each TF (lines 2039-2052):        │                         │
│     │   • Look up tsla_history[tf_str]      │                         │
│     │   • Look up spy_history[tf_str]       │                         │
│     │   • Call extract_channel_history()    │                         │
│     │   • Get 67 real history features ✓    │                         │
│     └────────────────────────────────────────┘                         │
│                                                                          │
│  OUTPUT: 14,840 features (with 670 history features)                    │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                        PASS 3 SCANNER                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. Uses Pre-computed SlimLabeledChannelMaps                            │
│     (from Pass 1/2, no new channel detection)                           │
│                                                                          │
│  2. NO Channel History Available                                        │
│     (Pass 3 works on sliced data, can't build history)                  │
│                                                                          │
│  3. Feature Extraction (Overload #1)                                    │
│     ┌────────────────────────────────────────┐                         │
│     │ • NO history maps provided             │                         │
│     │ • extract_channel_history() gets       │                         │
│     │   empty vectors                        │                         │
│     │ • All 67 history features = 0.0 ✗      │                         │
│     └────────────────────────────────────────┘                         │
│                                                                          │
│  OUTPUT: 14,170 features (670 history features missing)                 │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Performance Characteristics

```
╔════════════════════════════════════════════════════════════════════════╗
║                      PERFORMANCE COMPARISON                            ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Overload #1 (Vector-based)                                           ║
║  ────────────────────────────                                         ║
║  Memory:     ████████████ (Copies all data)                           ║
║  Speed:      ████████████ (Fast, no delegation)                       ║
║  Features:   ████████░░░░ (14,170 / 14,840)                           ║
║  Use Case:   Pass 3 (already has vector slices)                       ║
║                                                                        ║
║  Overload #2 (DataView, no history) - DEPRECATED                      ║
║  ──────────────────────────────────────────                           ║
║  Memory:     ████░░░░░░░░ (Zero-copy view → vectors)                  ║
║  Speed:      ████████░░░░ (Delegation overhead)                       ║
║  Features:   ████████░░░░ (14,170 / 14,840)                           ║
║  Use Case:   Intermediate optimization (not used anymore)             ║
║                                                                        ║
║  Overload #3 (DataView + History) - PRODUCTION                        ║
║  ──────────────────────────────────────                               ║
║  Memory:     ████░░░░░░░░ (Zero-copy view → vectors)                  ║
║  Speed:      ████████████ (Direct implementation)                     ║
║  Features:   ████████████ (14,840 / 14,840) ✓                         ║
║  Use Case:   Pass 1/2 (main production path)                          ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
```

---

## Feature Breakdown by Overload

```
┌────────────────────────────────────────────────────────────────────┐
│                  FEATURE COUNT BREAKDOWN                           │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Common Features (All 3 Overloads):                               │
│  ─────────────────────────────────────                            │
│    • TSLA price features:      580  (58 × 10 TFs)                 │
│    • Technical indicators:     590  (59 × 10 TFs)                 │
│    • SPY features:           1,170  (117 × 10 TFs)                │
│    • VIX features:             250  (25 × 10 TFs)                 │
│    • Cross-asset features:     590  (59 × 10 TFs)                 │
│    • Channel features:       9,280  (116 × 8 windows × 10 TFs)    │
│    • Window scores:            500  (50 × 10 TFs)                 │
│    • Event features:            30  (TF-independent)              │
│    • Bar metadata:              30  (3 × 10 TFs)                  │
│                              ──────                                │
│                              14,020 features                       │
│                                                                    │
│  + Channel Correlation:        150  (15 × 10 TFs)                 │
│                              ──────                                │
│  SUBTOTAL:                   14,170 features ◄─ Overload #1 & #2  │
│                                                                    │
│  + Channel History (Overload #3 only):                            │
│    • TSLA history:             335  (40 × 10 TFs - 65 aggregate)  │
│    • SPY history:              335  (40 × 10 TFs - 65 aggregate)  │
│                              ──────                                │
│  TOTAL:                      14,840 features ◄─ Overload #3 ONLY  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Overload #3 (Full Implementation)
Lines 1811-2090 in feature_extractor.cpp

```cpp
// Core extraction loop (runs for each TF)
for (int tf_idx = 0; tf_idx < NUM_TIMEFRAMES; ++tf_idx) {
    // ... extract price, technical, SPY, VIX, cross-asset, channels ...

    // 9. Extract channel history features (67) - WITH REAL DATA!
    std::vector<ChannelHistoryEntry> tsla_history;
    std::vector<ChannelHistoryEntry> spy_history;

    auto tsla_hist_it = tsla_history_by_tf.find(tf_str);  // ◄─ Lookup
    if (tsla_hist_it != tsla_history_by_tf.end()) {
        tsla_history = tsla_hist_it->second;  // ◄─ Found real data!
    }

    auto spy_hist_it = spy_history_by_tf.find(tf_str);
    if (spy_hist_it != spy_history_by_tf.end()) {
        spy_history = spy_hist_it->second;
    }

    // Extract 67 history features from real or empty vectors
    auto history_features = extract_channel_history_features(
        tsla_history,  // ◄─ Real data in Pass 1/2, empty in Pass 3
        spy_history
    );

    for (const auto& [name, value] : history_features) {
        local_features[tf_prefix + name] = value;  // ◄─ 67 features per TF
    }
}
```

### Overload #2 (Delegation)
Lines 1779-1805 in feature_extractor.cpp

```cpp
std::unordered_map<std::string, double> extract_all_features(
    const DataView& tsla_view,
    const DataView& spy_view,
    const DataView& vix_view,
    int64_t timestamp,
    const SlimLabeledChannelMap& tsla_slim_map,
    const SlimLabeledChannelMap& spy_slim_map,
    int source_bar_count,
    bool include_bar_metadata
) {
    // Create empty history maps
    std::unordered_map<std::string, std::vector<ChannelHistoryEntry>>
        empty_tsla_history;  // ◄─ Empty!
    std::unordered_map<std::string, std::vector<ChannelHistoryEntry>>
        empty_spy_history;   // ◄─ Empty!

    // Delegate to Overload #3 with empty history
    return extract_all_features(
        tsla_view, spy_view, vix_view,
        timestamp,
        tsla_slim_map, spy_slim_map,
        empty_tsla_history,  // ◄─ All history features = 0.0
        empty_spy_history,
        source_bar_count,
        include_bar_metadata
    );
}
```

---

## Summary

### Why 3 Overloads?

1. **Overload #1 (Vector-based)**
   - **Why:** Pass 3 scanner already has vector slices (not views)
   - **Trade-off:** Owns data (memory) but no delegation overhead
   - **History:** None available in Pass 3 context

2. **Overload #2 (DataView, no history)**
   - **Why:** Intended for intermediate optimization
   - **Status:** DEPRECATED - should be removed
   - **History:** Empty (delegates to #3 with empty maps)

3. **Overload #3 (DataView + History)**
   - **Why:** Main production path for Pass 1/2
   - **Trade-off:** Zero-copy views, complete feature set
   - **History:** Real data from channel detection

### Key Insight
The **only difference** between all three overloads is:
- Input data type (vector vs DataView)
- Presence of channel history (real vs empty)

All three eventually execute the same core feature extraction logic in Overload #3, but with different history data availability.

---

## Recommendations

1. **Keep Overload #1**: Required for Pass 3 (vector-based workflow)
2. **Keep Overload #3**: Main production path (DataView + history)
3. **Remove Overload #2**: Redundant delegation layer, no longer needed

After cleanup:
```
┌─────────────────────────────────────────────────────────────────┐
│  Simplified API (2 overloads)                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Vector-based (Pass 3):                                      │
│     extract_all_features(vector, vector, vector, ...)          │
│                                                                 │
│  2. DataView + History (Pass 1/2 - MAIN PATH):                 │
│     extract_all_features(DataView, DataView, DataView, ...,    │
│                          history_maps)                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
