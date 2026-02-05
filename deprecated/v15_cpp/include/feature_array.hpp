#pragma once

#include <vector>
#include <array>
#include <string>
#include <string_view>
#include <unordered_map>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace v15 {

// =============================================================================
// FEATURE STORAGE CONSTANTS
// =============================================================================

// Total number of features extracted per sample
// Updated to 14840 after adding price-agnostic normalized features
constexpr size_t TOTAL_FEATURE_COUNT = 14840;

// Pre-calculated bucket count for unordered_map (prime number > 14840 / 0.7 load factor)
constexpr size_t FEATURE_MAP_BUCKET_COUNT = 21211;

// =============================================================================
// OPTIMIZED FEATURE MAP
// =============================================================================

/**
 * Create a pre-reserved unordered_map for feature storage.
 * This avoids rehashing during the 14,840 insertions.
 *
 * Usage:
 *   auto features = create_feature_map();
 *   features["feature_name"] = value;  // No rehashing overhead
 */
inline std::unordered_map<std::string, double> create_feature_map() {
    std::unordered_map<std::string, double> map;
    map.reserve(TOTAL_FEATURE_COUNT);
    return map;
}

/**
 * Create a pre-reserved unordered_map with a specific capacity.
 * Useful for intermediate feature extraction functions.
 */
inline std::unordered_map<std::string, double> create_feature_map(size_t capacity) {
    std::unordered_map<std::string, double> map;
    map.reserve(capacity);
    return map;
}

// =============================================================================
// FEATURE ARRAY - INDEX-BASED STORAGE (FAST PATH)
// =============================================================================

/**
 * FeatureArray - High-performance feature storage using indices
 *
 * This class provides O(1) access to features using integer indices instead
 * of string hashing. Features are stored in a contiguous vector for cache
 * efficiency.
 *
 * Design:
 * - During extraction: Use indices for O(1) writes
 * - For serialization: Convert to map once at the end
 *
 * Memory: ~114KB per instance (14190 * 8 bytes)
 */
class FeatureArray {
public:
    // Default constructor - pre-allocates storage
    FeatureArray() : values_(TOTAL_FEATURE_COUNT, std::nan("")) {}

    // Constructor with custom size
    explicit FeatureArray(size_t count) : values_(count, std::nan("")) {}

    // ==========================================================================
    // Fast O(1) access by index
    // ==========================================================================

    // Set a feature value by index (no bounds checking for speed)
    void set(size_t index, double value) noexcept {
        values_[index] = value;
    }

    // Get a feature value by index (no bounds checking for speed)
    double get(size_t index) const noexcept {
        return values_[index];
    }

    // Operator[] for array-like access
    double& operator[](size_t index) noexcept {
        return values_[index];
    }

    const double& operator[](size_t index) const noexcept {
        return values_[index];
    }

    // Safe access with bounds checking
    void set_safe(size_t index, double value) {
        if (index >= values_.size()) {
            throw std::out_of_range("Feature index out of range");
        }
        values_[index] = value;
    }

    double get_safe(size_t index) const {
        if (index >= values_.size()) {
            throw std::out_of_range("Feature index out of range");
        }
        return values_[index];
    }

    // ==========================================================================
    // Bulk operations
    // ==========================================================================

    // Set a range of features starting at offset
    void set_range(size_t offset, const double* data, size_t count) noexcept {
        std::copy_n(data, count, values_.begin() + offset);
    }

    // Set a range from vector
    void set_range(size_t offset, const std::vector<double>& data) noexcept {
        std::copy(data.begin(), data.end(), values_.begin() + offset);
    }

    // Clear all values (set to NaN)
    void clear() noexcept {
        std::fill(values_.begin(), values_.end(), std::nan(""));
    }

    // Reset all values to a specific value
    void reset(double value = 0.0) noexcept {
        std::fill(values_.begin(), values_.end(), value);
    }

    // ==========================================================================
    // Conversion and serialization
    // ==========================================================================

    /**
     * Convert to unordered_map using provided feature names.
     * This is the slow path - only call once when serialization is needed.
     *
     * @param names Vector of feature names (must match values_ size)
     * @return Map of feature name -> value
     */
    std::unordered_map<std::string, double> to_map(
        const std::vector<std::string>& names
    ) const {
        auto map = create_feature_map();
        size_t count = std::min(names.size(), values_.size());
        for (size_t i = 0; i < count; ++i) {
            if (!std::isnan(values_[i])) {
                map[names[i]] = values_[i];
            }
        }
        return map;
    }

    /**
     * Convert to unordered_map, including NaN values as 0.0
     */
    std::unordered_map<std::string, double> to_map_with_defaults(
        const std::vector<std::string>& names,
        double default_value = 0.0
    ) const {
        auto map = create_feature_map();
        size_t count = std::min(names.size(), values_.size());
        for (size_t i = 0; i < count; ++i) {
            map[names[i]] = std::isnan(values_[i]) ? default_value : values_[i];
        }
        return map;
    }

    // ==========================================================================
    // Utility
    // ==========================================================================

    size_t size() const noexcept { return values_.size(); }

    // Get count of non-NaN values
    size_t count_valid() const noexcept {
        size_t count = 0;
        for (double v : values_) {
            if (!std::isnan(v)) ++count;
        }
        return count;
    }

    // Sanitize: replace NaN/Inf with default value
    void sanitize(double default_value = 0.0) noexcept {
        for (double& v : values_) {
            if (!std::isfinite(v)) {
                v = default_value;
            }
        }
    }

    // Direct access to underlying storage (for advanced use)
    const std::vector<double>& data() const noexcept { return values_; }
    std::vector<double>& data() noexcept { return values_; }

    double* raw_data() noexcept { return values_.data(); }
    const double* raw_data() const noexcept { return values_.data(); }

private:
    std::vector<double> values_;
};

// =============================================================================
// FEATURE INDEX REGISTRY
// =============================================================================

/**
 * FeatureIndexRegistry - Maps feature names to indices at compile/init time
 *
 * Usage:
 *   static FeatureIndexRegistry registry;
 *   registry.register_feature("5min_close", 0);
 *   ...
 *   size_t idx = registry.get_index("5min_close");
 *   features.set(idx, value);
 */
class FeatureIndexRegistry {
public:
    FeatureIndexRegistry() {
        name_to_index_.reserve(TOTAL_FEATURE_COUNT);
        index_to_name_.reserve(TOTAL_FEATURE_COUNT);
    }

    // Register a feature name and return its index
    size_t register_feature(const std::string& name) {
        auto it = name_to_index_.find(name);
        if (it != name_to_index_.end()) {
            return it->second;
        }
        size_t index = index_to_name_.size();
        name_to_index_[name] = index;
        index_to_name_.push_back(name);
        return index;
    }

    // Register a feature with a specific index
    void register_feature(const std::string& name, size_t index) {
        name_to_index_[name] = index;
        if (index >= index_to_name_.size()) {
            index_to_name_.resize(index + 1);
        }
        index_to_name_[index] = name;
    }

    // Get index for a feature name (returns SIZE_MAX if not found)
    size_t get_index(const std::string& name) const {
        auto it = name_to_index_.find(name);
        return (it != name_to_index_.end()) ? it->second : SIZE_MAX;
    }

    // Get name for an index
    const std::string& get_name(size_t index) const {
        static const std::string empty;
        return (index < index_to_name_.size()) ? index_to_name_[index] : empty;
    }

    // Check if a feature is registered
    bool has_feature(const std::string& name) const {
        return name_to_index_.find(name) != name_to_index_.end();
    }

    // Get all feature names in index order
    const std::vector<std::string>& get_all_names() const {
        return index_to_name_;
    }

    size_t size() const { return index_to_name_.size(); }

    // Build index from existing feature names vector
    void build_from_names(const std::vector<std::string>& names) {
        name_to_index_.clear();
        name_to_index_.reserve(names.size());
        index_to_name_ = names;
        for (size_t i = 0; i < names.size(); ++i) {
            name_to_index_[names[i]] = i;
        }
    }

private:
    std::unordered_map<std::string, size_t> name_to_index_;
    std::vector<std::string> index_to_name_;
};

// =============================================================================
// FEATURE CATEGORY OFFSETS
// =============================================================================

/**
 * Feature category structure for organizing feature indices.
 * These offsets are computed based on the feature extraction order.
 */
namespace FeatureOffsets {
    // Per-timeframe feature counts (updated for price-agnostic normalized features)
    constexpr size_t TSLA_PRICE_COUNT = 65;      // Was 58, +7 normalized
    constexpr size_t TECHNICAL_COUNT = 59;
    constexpr size_t SPY_COUNT = 121;            // Was 117, +4 normalized
    constexpr size_t VIX_COUNT = 31;             // Was 25, +6 normalized
    constexpr size_t CROSS_ASSET_COUNT = 59;
    constexpr size_t CHANNEL_PER_WINDOW_COUNT = 122;  // Was 116, +6 normalized (61 TSLA + 61 SPY)
    constexpr size_t WINDOW_SCORE_COUNT = 50;
    constexpr size_t CHANNEL_HISTORY_COUNT = 67;
    constexpr size_t CHANNEL_CORRELATION_COUNT = 50;

    // Number of windows and timeframes
    constexpr size_t NUM_WINDOWS = 8;
    constexpr size_t NUM_TFS = 10;

    // Per-timeframe total (excluding channel features per window)
    constexpr size_t PER_TF_BASE = TSLA_PRICE_COUNT + TECHNICAL_COUNT + SPY_COUNT +
                                    VIX_COUNT + CROSS_ASSET_COUNT + WINDOW_SCORE_COUNT +
                                    CHANNEL_HISTORY_COUNT + CHANNEL_CORRELATION_COUNT;
    // = 65 + 59 + 121 + 31 + 59 + 50 + 67 + 50 = 502

    // Channel features per TF = 122 * 8 = 976
    constexpr size_t CHANNEL_PER_TF = CHANNEL_PER_WINDOW_COUNT * NUM_WINDOWS;

    // Total per TF = 502 + 976 = 1478
    constexpr size_t PER_TF_TOTAL = PER_TF_BASE + CHANNEL_PER_TF;

    // TF-independent features
    constexpr size_t EVENT_COUNT = 30;
    constexpr size_t BAR_METADATA_COUNT = 30;  // 3 per TF

    // Calculated total: 1478 * 10 + 30 + 30 = 14,840
    constexpr size_t CALCULATED_TOTAL = PER_TF_TOTAL * NUM_TFS + EVENT_COUNT + BAR_METADATA_COUNT;

    // Verify at compile time
    static_assert(CALCULATED_TOTAL == TOTAL_FEATURE_COUNT,
                  "Feature count mismatch - update offsets");

    /**
     * Get the starting index for a timeframe's features
     */
    constexpr size_t tf_offset(size_t tf_idx) {
        return tf_idx * PER_TF_TOTAL;
    }

    /**
     * Get the starting index for TSLA price features within a TF block
     */
    constexpr size_t tsla_price_offset(size_t tf_idx) {
        return tf_offset(tf_idx);
    }

    /**
     * Get the starting index for technical indicator features within a TF block
     */
    constexpr size_t technical_offset(size_t tf_idx) {
        return tf_offset(tf_idx) + TSLA_PRICE_COUNT;
    }

    /**
     * Get the starting index for SPY features within a TF block
     */
    constexpr size_t spy_offset(size_t tf_idx) {
        return tf_offset(tf_idx) + TSLA_PRICE_COUNT + TECHNICAL_COUNT;
    }

    /**
     * Get the starting index for VIX features within a TF block
     */
    constexpr size_t vix_offset(size_t tf_idx) {
        return tf_offset(tf_idx) + TSLA_PRICE_COUNT + TECHNICAL_COUNT + SPY_COUNT;
    }

    /**
     * Get the starting index for cross-asset features within a TF block
     */
    constexpr size_t cross_asset_offset(size_t tf_idx) {
        return tf_offset(tf_idx) + TSLA_PRICE_COUNT + TECHNICAL_COUNT + SPY_COUNT + VIX_COUNT;
    }

    /**
     * Get the starting index for channel features for a specific window within a TF block
     */
    constexpr size_t channel_window_offset(size_t tf_idx, size_t window_idx) {
        return tf_offset(tf_idx) + TSLA_PRICE_COUNT + TECHNICAL_COUNT + SPY_COUNT +
               VIX_COUNT + CROSS_ASSET_COUNT + (window_idx * CHANNEL_PER_WINDOW_COUNT);
    }

    /**
     * Get the starting index for channel correlation features within a TF block
     */
    constexpr size_t channel_correlation_offset(size_t tf_idx) {
        return tf_offset(tf_idx) + TSLA_PRICE_COUNT + TECHNICAL_COUNT + SPY_COUNT +
               VIX_COUNT + CROSS_ASSET_COUNT + CHANNEL_PER_TF;
    }

    /**
     * Get the starting index for window score features within a TF block
     */
    constexpr size_t window_score_offset(size_t tf_idx) {
        return channel_correlation_offset(tf_idx) + CHANNEL_CORRELATION_COUNT;
    }

    /**
     * Get the starting index for channel history features within a TF block
     */
    constexpr size_t channel_history_offset(size_t tf_idx) {
        return window_score_offset(tf_idx) + WINDOW_SCORE_COUNT;
    }

    /**
     * Get the starting index for event features (TF-independent)
     */
    constexpr size_t event_offset() {
        return NUM_TFS * PER_TF_TOTAL;
    }

    /**
     * Get the starting index for bar metadata features (TF-independent)
     */
    constexpr size_t bar_metadata_offset() {
        return event_offset() + EVENT_COUNT;
    }
}

// =============================================================================
// HELPER FUNCTIONS FOR OPTIMIZED INSERTION
// =============================================================================

/**
 * Merge features from source map into destination map.
 * The destination should be pre-reserved.
 */
inline void merge_features(
    std::unordered_map<std::string, double>& dest,
    const std::unordered_map<std::string, double>& src
) {
    for (const auto& [name, value] : src) {
        dest[name] = value;
    }
}

/**
 * Merge features with a prefix into destination map.
 * More efficient than creating a new prefixed map.
 */
inline void merge_features_prefixed(
    std::unordered_map<std::string, double>& dest,
    const std::unordered_map<std::string, double>& src,
    const std::string& prefix
) {
    for (const auto& [name, value] : src) {
        dest[prefix + name] = value;
    }
}

/**
 * Pre-reserve capacity for an existing map.
 * Call this before bulk insertions if the map wasn't created with create_feature_map().
 */
inline void reserve_features(std::unordered_map<std::string, double>& map, size_t capacity) {
    map.reserve(capacity);
}

} // namespace v15
