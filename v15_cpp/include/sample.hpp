#pragma once

#include "types.hpp"
#include "labels.hpp"
#include <unordered_map>
#include <string>
#include <cstdint>

namespace v15 {

// =============================================================================
// CHANNEL SAMPLE STRUCTURE
// =============================================================================

/**
 * A complete sample for V15 channel prediction.
 *
 * ARCHITECTURE: Samples are created ONLY at channel end positions.
 * Therefore: sample_position = channel_end_idx = the point of prediction.
 *
 * Contains:
 *   - timestamp: Channel end timestamp (= sample timestamp = prediction point)
 *   - channel_end_idx: Index in 5min data where channel ends (= sample position)
 *   - tf_features: Dict of all features (flat, TF-prefixed)
 *   - labels_per_window: Labels for each window/asset/TF combination
 *   - bar_metadata: Partial bar completion info per TF
 *   - best_window: Optimal window size
 *
 * Memory layout:
 *   - Hot data first (timestamp, index, best_window)
 *   - Feature map for fast lookup
 *   - Nested label maps for window-specific data
 *   - Metadata last (cold data)
 */
struct ChannelSample {
    // =========================================================================
    // CORE SAMPLE DATA (hot - accessed most frequently)
    // =========================================================================
    int64_t timestamp;          // Unix timestamp in milliseconds (channel end time)
    int channel_end_idx;        // Index in 5min data array where channel ends
    int best_window;            // Optimal window size (e.g., 50)

    // =========================================================================
    // FEATURES (TF-prefixed flat feature map)
    // =========================================================================
    // Key format: "{timeframe}_{feature_name}"
    // Example: "1h_rsi", "daily_macd", "5min_volume_ratio"
    std::unordered_map<std::string, double> tf_features;

    // =========================================================================
    // LABELS PER WINDOW
    // =========================================================================
    // Outer key: window size (e.g., 50)
    // Inner key: timeframe string (e.g., "1h")
    // Value: Labels for that window/timeframe combination
    std::unordered_map<int, std::unordered_map<std::string, ChannelLabels>> labels_per_window;

    // =========================================================================
    // BAR METADATA
    // =========================================================================
    // Key: timeframe string (e.g., "1h")
    // Value: metadata map with keys like "partial_bar_pct", "bars_since_session_open"
    std::unordered_map<std::string, std::unordered_map<std::string, double>> bar_metadata;

    // =========================================================================
    // CONSTRUCTORS
    // =========================================================================

    ChannelSample()
        : timestamp(0)
        , channel_end_idx(0)
        , best_window(50)
    {}

    ChannelSample(int64_t ts, int end_idx, int best_win = 50)
        : timestamp(ts)
        , channel_end_idx(end_idx)
        , best_window(best_win)
    {}

    // =========================================================================
    // UTILITY METHODS
    // =========================================================================

    // Get feature value (returns 0.0 if not found)
    inline double get_feature(const std::string& key, double default_val = 0.0) const {
        auto it = tf_features.find(key);
        return (it != tf_features.end()) ? it->second : default_val;
    }

    // Set feature value
    inline void set_feature(const std::string& key, double value) {
        tf_features[key] = value;
    }

    // Check if feature exists
    inline bool has_feature(const std::string& key) const {
        return tf_features.find(key) != tf_features.end();
    }

    // Get labels for specific window and timeframe
    inline const ChannelLabels* get_labels(int window, const std::string& tf) const {
        auto win_it = labels_per_window.find(window);
        if (win_it == labels_per_window.end()) {
            return nullptr;
        }
        auto tf_it = win_it->second.find(tf);
        if (tf_it == win_it->second.end()) {
            return nullptr;
        }
        return &(tf_it->second);
    }

    // Set labels for specific window and timeframe
    inline void set_labels(int window, const std::string& tf, const ChannelLabels& labels) {
        labels_per_window[window][tf] = labels;
    }

    // Get labels for best window and timeframe
    inline const ChannelLabels* get_best_labels(const std::string& tf) const {
        return get_labels(best_window, tf);
    }

    // Get bar metadata value
    inline double get_bar_metadata(const std::string& tf, const std::string& key,
                                   double default_val = 0.0) const {
        auto tf_it = bar_metadata.find(tf);
        if (tf_it == bar_metadata.end()) {
            return default_val;
        }
        auto key_it = tf_it->second.find(key);
        if (key_it == tf_it->second.end()) {
            return default_val;
        }
        return key_it->second;
    }

    // Set bar metadata value
    inline void set_bar_metadata(const std::string& tf, const std::string& key, double value) {
        bar_metadata[tf][key] = value;
    }

    // Get total feature count
    inline size_t feature_count() const {
        return tf_features.size();
    }

    // Get total label count (all windows, all timeframes)
    inline size_t label_count() const {
        size_t count = 0;
        for (const auto& win_pair : labels_per_window) {
            count += win_pair.second.size();
        }
        return count;
    }

    // Get available windows
    inline std::vector<int> get_windows() const {
        std::vector<int> windows;
        windows.reserve(labels_per_window.size());
        for (const auto& pair : labels_per_window) {
            windows.push_back(pair.first);
        }
        return windows;
    }

    // Get available timeframes for a window
    inline std::vector<std::string> get_timeframes(int window) const {
        std::vector<std::string> timeframes;
        auto it = labels_per_window.find(window);
        if (it != labels_per_window.end()) {
            timeframes.reserve(it->second.size());
            for (const auto& pair : it->second) {
                timeframes.push_back(pair.first);
            }
        }
        return timeframes;
    }

    // Check if sample is valid
    inline bool is_valid() const {
        return timestamp > 0 &&
               channel_end_idx >= 0 &&
               best_window > 0 &&
               !tf_features.empty() &&
               !labels_per_window.empty();
    }
};

} // namespace v15
