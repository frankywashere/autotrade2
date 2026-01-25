#pragma once

#include "types.hpp"
#include <cstdint>
#include <vector>

namespace v15 {

// Forward declarations for ChannelDetector types
enum class TouchType {
    LOWER = 0,
    UPPER = 1
};

struct Touch {
    int bar_index;
    TouchType touch_type;
    double price;  // HIGH for upper, LOW for lower

    Touch(int idx, TouchType type, double p)
        : bar_index(idx), touch_type(type), price(p) {}
};

// =============================================================================
// CHANNEL STRUCTURE
// =============================================================================

/**
 * Represents a detected price channel with regression parameters.
 *
 * A channel is defined by linear regression on price data within a window,
 * with upper and lower bounds defined by standard deviation bands.
 *
 * This struct combines fields needed by both scanner (start_idx, end_idx, timestamps)
 * and channel_detector (window, bounces, quality metrics).
 */
struct Channel {
    // Validity flag (for ChannelDetector compatibility)
    bool valid;

    // Core indices (most frequently accessed)
    int start_idx;          // Start index in 5min data array
    int end_idx;            // End index in 5min data array
    int window_size;        // Window size used for detection (duplicate with window)
    int window;             // Window size (for ChannelDetector compatibility)

    // Regression parameters (hot data, kept together for cache locality)
    double slope;           // Slope of regression line
    double intercept;       // Y-intercept of regression line
    double r_squared;       // R-squared goodness of fit [0.0, 1.0]
    double std_dev;         // Standard deviation of residuals (channel width)

    // Channel characteristics
    int bounce_count;       // Number of alternating bound touches
    int complete_cycles;    // Number of complete bounce cycles (for ChannelDetector compatibility)
    ChannelDirection direction;  // BEAR(0), SIDEWAYS(1), BULL(2)

    // Metrics (from ChannelDetector)
    double width_pct;           // Channel width as percentage
    int alternations;           // Number of alternating touches
    double alternation_ratio;   // Ratio of alternations to total touches
    int upper_touches;          // Count of upper boundary touches
    int lower_touches;          // Count of lower boundary touches
    double quality_score;       // Overall channel quality score

    // Channel bounds (arrays of length window, from ChannelDetector)
    std::vector<double> upper_line;
    std::vector<double> lower_line;
    std::vector<double> center_line;

    // Bounce detection (from ChannelDetector)
    std::vector<Touch> touches;

    // OHLC data (optional, for position calculations, from ChannelDetector)
    std::vector<double> close;
    std::vector<double> high;
    std::vector<double> low;

    // Metadata
    Timeframe timeframe;    // Which timeframe this channel belongs to

    // Timestamp (in milliseconds since epoch for efficient storage)
    int64_t start_timestamp_ms;
    int64_t end_timestamp_ms;

    // Default constructor
    Channel()
        : valid(false)
        , start_idx(0)
        , end_idx(0)
        , window_size(0)
        , window(0)
        , slope(0.0)
        , intercept(0.0)
        , r_squared(0.0)
        , std_dev(0.0)
        , bounce_count(0)
        , complete_cycles(0)
        , direction(ChannelDirection::UNKNOWN)
        , width_pct(0.0)
        , alternations(0)
        , alternation_ratio(0.0)
        , upper_touches(0)
        , lower_touches(0)
        , quality_score(0.0)
        , timeframe(Timeframe::INVALID)
        , start_timestamp_ms(0)
        , end_timestamp_ms(0)
    {}

    // Full constructor
    Channel(int start, int end, int window, double slp, double intcpt,
            double r2, double std, int bounces, ChannelDirection dir,
            Timeframe tf, int64_t start_ts, int64_t end_ts)
        : start_idx(start)
        , end_idx(end)
        , window_size(window)
        , slope(slp)
        , intercept(intcpt)
        , r_squared(r2)
        , std_dev(std)
        , bounce_count(bounces)
        , direction(dir)
        , timeframe(tf)
        , start_timestamp_ms(start_ts)
        , end_timestamp_ms(end_ts)
    {}

    // Utility methods

    // Get duration in bars
    inline int duration() const {
        return end_idx - start_idx + 1;
    }

    // Get predicted value at index
    inline double predict(int idx) const {
        return slope * idx + intercept;
    }

    // Get upper bound at index
    inline double upper_bound(int idx) const {
        return predict(idx) + std_dev;
    }

    // Get lower bound at index
    inline double lower_bound(int idx) const {
        return predict(idx) - std_dev;
    }

    // Check if price is within channel bounds
    inline bool contains(int idx, double price) const {
        return price >= lower_bound(idx) && price <= upper_bound(idx);
    }

    // Check if price breaks upper bound
    inline bool breaks_upper(int idx, double price) const {
        return price > upper_bound(idx);
    }

    // Check if price breaks lower bound
    inline bool breaks_lower(int idx, double price) const {
        return price < lower_bound(idx);
    }

    // Calculate distance from center line (in std devs)
    inline double distance_from_center(int idx, double price) const {
        double predicted = predict(idx);
        return (std_dev > 0.0) ? (price - predicted) / std_dev : 0.0;
    }

    // Check if channel is valid
    inline bool is_valid() const {
        return end_idx > start_idx &&
               window_size > 0 &&
               std_dev > 0.0 &&
               r_squared >= 0.0 && r_squared <= 1.0 &&
               direction != ChannelDirection::UNKNOWN &&
               timeframe != Timeframe::INVALID;
    }

    // Get position in channel (0=lower, 0.5=center, 1=upper)
    double position_at(int bar_index = -1) const;

    // Get slope as percentage per bar
    double slope_pct() const;
};

} // namespace v15
