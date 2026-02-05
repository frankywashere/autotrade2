#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <ctime>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace v15 {

// =============================================================================
// BASIC TYPES AND CONSTANTS
// =============================================================================

// OHLCV bar structure
struct OHLCV {
    std::time_t timestamp;  // Unix timestamp
    double open;
    double high;
    double low;
    double close;
    double volume;

    OHLCV() : timestamp(0), open(0.0), high(0.0), low(0.0), close(0.0), volume(0.0) {}

    OHLCV(std::time_t ts, double o, double h, double l, double c, double v)
        : timestamp(ts), open(o), high(h), low(l), close(c), volume(v) {}
};

// Timeframe enumeration (10 timeframes total)
enum class Timeframe : int {
    MIN_5 = 0,
    MIN_15 = 1,
    MIN_30 = 2,
    HOUR_1 = 3,
    HOUR_2 = 4,
    HOUR_3 = 5,
    HOUR_4 = 6,
    DAILY = 7,
    WEEKLY = 8,
    MONTHLY = 9,
    INVALID = -1
};

// Channel direction encoding
enum class ChannelDirection : int {
    BEAR = 0,
    SIDEWAYS = 1,
    BULL = 2,
    UNKNOWN = -1
};

// Break direction encoding
enum class BreakDirection : int {
    DOWN = 0,
    UP = 1,
    NONE = -1
};

// RSI divergence types
enum class RSIDivergence : int {
    BEARISH = -1,
    NONE = 0,
    BULLISH = 1
};

// RSI trend types
enum class RSITrend : int {
    FALLING = -1,
    FLAT = 0,
    RISING = 1
};

// Constants
constexpr int NUM_TIMEFRAMES = 10;
constexpr int NUM_STANDARD_WINDOWS = 8;

// Standard window sizes
constexpr std::array<int, NUM_STANDARD_WINDOWS> STANDARD_WINDOWS = {
    10, 20, 30, 40, 50, 60, 70, 80
};

// Timeframe string names
constexpr std::array<const char*, NUM_TIMEFRAMES> TIMEFRAME_NAMES = {
    "5min", "15min", "30min", "1h", "2h", "3h", "4h",
    "daily", "weekly", "monthly"
};

// Bars per timeframe (how many 5min bars per timeframe bar)
constexpr std::array<int, NUM_TIMEFRAMES> BARS_PER_TF = {
    1,      // 5min
    3,      // 15min
    6,      // 30min
    12,     // 1h
    24,     // 2h
    36,     // 3h
    48,     // 4h
    78,     // daily (6.5 hours * 12)
    390,    // weekly (5 days * 78)
    1638    // monthly (~21 trading days * 78)
};

// Maximum bars to scan forward for break detection per timeframe
constexpr std::array<int, NUM_TIMEFRAMES> TF_MAX_SCAN = {
    500,    // 5min
    400,    // 15min
    350,    // 30min
    300,    // 1h
    250,    // 2h
    200,    // 3h
    150,    // 4h
    100,    // daily
    52,     // weekly
    24      // monthly
};

// Per-TF forward requirements for label scanning (in 5-min bars)
// Based on TF_MAX_SCAN * BARS_PER_TF
constexpr std::array<int, NUM_TIMEFRAMES> TF_FORWARD_5MIN = {
    600,        // 5min: 500 + buffer
    1400,       // 15min: 400 * 3 + buffer
    2400,       // 30min: 350 * 6 + buffer
    4000,       // 1h: 300 * 12 + buffer
    6500,       // 2h: 250 * 24 + buffer
    8000,       // 3h: 200 * 36 + buffer
    8000,       // 4h: 150 * 48 + buffer
    8500,       // daily: 100 * 78 + buffer
    21000,      // weekly: 52 * 390 + buffer
    40000       // monthly: 24 * 1638 + buffer
};

// Scanner forward requirement (based on weekly TF for practical limit)
constexpr int SCANNER_FORWARD_5MIN = 21000;

// =============================================================================
// DATA VIEW - ZERO-COPY ACCESS TO OHLCV SLICES
// =============================================================================

/**
 * DataView - Lightweight non-owning view into OHLCV data
 *
 * Provides zero-copy access to a slice of OHLCV data.
 * Similar to std::span (C++20) but compatible with C++17.
 *
 * IMPORTANT: The underlying data must outlive the DataView.
 */
struct DataView {
    const OHLCV* data_;
    size_t size_;

    // Default constructor - empty view
    DataView() noexcept : data_(nullptr), size_(0) {}

    // Construct from pointer and size
    DataView(const OHLCV* data, size_t size) noexcept
        : data_(data), size_(size) {}

    // Construct from iterators (vector begin + offset, size)
    DataView(const std::vector<OHLCV>& vec, size_t offset, size_t count) noexcept
        : data_(vec.data() + offset), size_(count) {}

    // Construct from full vector
    explicit DataView(const std::vector<OHLCV>& vec) noexcept
        : data_(vec.data()), size_(vec.size()) {}

    // Accessors
    const OHLCV* data() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    // Element access
    const OHLCV& operator[](size_t idx) const { return data_[idx]; }
    const OHLCV& at(size_t idx) const {
        if (idx >= size_) throw std::out_of_range("DataView index out of range");
        return data_[idx];
    }

    const OHLCV& front() const { return data_[0]; }
    const OHLCV& back() const { return data_[size_ - 1]; }

    // Iterators (for range-based for loops)
    const OHLCV* begin() const noexcept { return data_; }
    const OHLCV* end() const noexcept { return data_ + size_; }
    const OHLCV* cbegin() const noexcept { return data_; }
    const OHLCV* cend() const noexcept { return data_ + size_; }

    // Create a subview
    DataView subview(size_t offset, size_t count = static_cast<size_t>(-1)) const noexcept {
        if (offset >= size_) return DataView();
        size_t actual_count = std::min(count, size_ - offset);
        return DataView(data_ + offset, actual_count);
    }

    // Conversion to vector (when copy is actually needed, e.g., for resampling)
    std::vector<OHLCV> to_vector() const {
        return std::vector<OHLCV>(data_, data_ + size_);
    }
};

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Convert timeframe enum to string
inline const char* timeframe_to_string(Timeframe tf) {
    if (static_cast<int>(tf) >= 0 && static_cast<int>(tf) < NUM_TIMEFRAMES) {
        return TIMEFRAME_NAMES[static_cast<int>(tf)];
    }
    return "invalid";
}

// Convert string to timeframe enum
inline Timeframe string_to_timeframe(const std::string& tf_str) {
    for (int i = 0; i < NUM_TIMEFRAMES; ++i) {
        if (tf_str == TIMEFRAME_NAMES[i]) {
            return static_cast<Timeframe>(i);
        }
    }
    return Timeframe::INVALID;
}

// Get bars per timeframe
inline int get_bars_per_tf(Timeframe tf) {
    if (static_cast<int>(tf) >= 0 && static_cast<int>(tf) < NUM_TIMEFRAMES) {
        return BARS_PER_TF[static_cast<int>(tf)];
    }
    return 1;
}

// Get max scan bars for timeframe
inline int get_max_scan(Timeframe tf) {
    if (static_cast<int>(tf) >= 0 && static_cast<int>(tf) < NUM_TIMEFRAMES) {
        return TF_MAX_SCAN[static_cast<int>(tf)];
    }
    return 100;
}

// Check if window size is standard
inline bool is_standard_window(int window) {
    for (int w : STANDARD_WINDOWS) {
        if (w == window) return true;
    }
    return false;
}

} // namespace v15
