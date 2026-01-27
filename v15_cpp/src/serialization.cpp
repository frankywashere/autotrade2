/**
 * V15 Channel Sample Serialization
 *
 * High-performance binary serialization for ChannelSample data.
 * Compatible with Python numpy and struct module for cross-language loading.
 */

#include "serialization.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace v15 {

// =============================================================================
// HELPER FUNCTIONS - WRITE PRIMITIVES
// =============================================================================

static void write_uint8(std::ofstream& ofs, uint8_t value) {
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_uint16(std::ofstream& ofs, uint16_t value) {
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_uint32(std::ofstream& ofs, uint32_t value) {
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_uint64(std::ofstream& ofs, uint64_t value) {
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_int32(std::ofstream& ofs, int32_t value) {
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_int64(std::ofstream& ofs, int64_t value) {
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_double(std::ofstream& ofs, double value) {
    ofs.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

static void write_bool(std::ofstream& ofs, bool value) {
    uint8_t byte = value ? 1 : 0;
    write_uint8(ofs, byte);
}

static void write_string(std::ofstream& ofs, const std::string& str) {
    if (str.size() > 65535) {
        throw std::runtime_error("String too long for serialization (max 65535 bytes)");
    }
    uint16_t length = static_cast<uint16_t>(str.size());
    write_uint16(ofs, length);
    ofs.write(str.data(), length);
}

// Write a vector of integers
static void write_int_vector(std::ofstream& ofs, const std::vector<int>& vec) {
    uint32_t count = static_cast<uint32_t>(vec.size());
    write_uint32(ofs, count);
    for (int value : vec) {
        write_int32(ofs, value);
    }
}

// Write a vector of doubles
static void write_double_vector(std::ofstream& ofs, const std::vector<double>& vec) {
    uint32_t count = static_cast<uint32_t>(vec.size());
    write_uint32(ofs, count);
    for (double value : vec) {
        write_double(ofs, value);
    }
}

// Write a vector of bools
static void write_bool_vector(std::ofstream& ofs, const std::vector<bool>& vec) {
    uint32_t count = static_cast<uint32_t>(vec.size());
    write_uint32(ofs, count);
    for (bool value : vec) {
        write_bool(ofs, value);
    }
}

// =============================================================================
// HELPER FUNCTIONS - READ PRIMITIVES
// =============================================================================

static uint8_t read_uint8(std::ifstream& ifs) {
    uint8_t value;
    ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!ifs) throw std::runtime_error("Failed to read uint8");
    return value;
}

static uint16_t read_uint16(std::ifstream& ifs) {
    uint16_t value;
    ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!ifs) throw std::runtime_error("Failed to read uint16");
    return value;
}

static uint32_t read_uint32(std::ifstream& ifs) {
    uint32_t value;
    ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!ifs) throw std::runtime_error("Failed to read uint32");
    return value;
}

static uint64_t read_uint64(std::ifstream& ifs) {
    uint64_t value;
    ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!ifs) throw std::runtime_error("Failed to read uint64");
    return value;
}

static int32_t read_int32(std::ifstream& ifs) {
    int32_t value;
    ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!ifs) throw std::runtime_error("Failed to read int32");
    return value;
}

static int64_t read_int64(std::ifstream& ifs) {
    int64_t value;
    ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!ifs) throw std::runtime_error("Failed to read int64");
    return value;
}

static double read_double(std::ifstream& ifs) {
    double value;
    ifs.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!ifs) throw std::runtime_error("Failed to read double");
    return value;
}

static bool read_bool(std::ifstream& ifs) {
    return read_uint8(ifs) != 0;
}

static std::string read_string(std::ifstream& ifs) {
    uint16_t length = read_uint16(ifs);
    std::string str(length, '\0');
    ifs.read(&str[0], length);
    if (!ifs) throw std::runtime_error("Failed to read string");
    return str;
}

static std::vector<int> read_int_vector(std::ifstream& ifs) {
    uint32_t count = read_uint32(ifs);
    std::vector<int> vec;
    vec.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        vec.push_back(read_int32(ifs));
    }
    return vec;
}

static std::vector<double> read_double_vector(std::ifstream& ifs) {
    uint32_t count = read_uint32(ifs);
    std::vector<double> vec;
    vec.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        vec.push_back(read_double(ifs));
    }
    return vec;
}

static std::vector<bool> read_bool_vector(std::ifstream& ifs) {
    uint32_t count = read_uint32(ifs);
    std::vector<bool> vec;
    vec.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        vec.push_back(read_bool(ifs));
    }
    return vec;
}

// =============================================================================
// CHANNEL LABELS SERIALIZATION
// =============================================================================

static void serialize_channel_labels(std::ofstream& ofs, const ChannelLabels& labels) {
    // Core prediction targets
    write_int32(ofs, labels.duration_bars);
    write_int32(ofs, labels.next_channel_direction);
    write_bool(ofs, labels.permanent_break);
    write_int32(ofs, static_cast<int32_t>(labels.timeframe));

    // TSLA break scan features
    write_int32(ofs, labels.break_direction);
    write_double(ofs, labels.break_magnitude);
    write_int32(ofs, labels.bars_to_first_break);
    write_bool(ofs, labels.returned_to_channel);
    write_int32(ofs, labels.bounces_after_return);
    write_int32(ofs, labels.round_trip_bounces);
    write_bool(ofs, labels.channel_continued);
    write_int32(ofs, labels.permanent_break_direction);
    write_double(ofs, labels.permanent_break_magnitude);
    write_int32(ofs, labels.bars_to_permanent_break);
    write_int32(ofs, labels.duration_to_permanent);
    write_double(ofs, labels.avg_bars_outside);
    write_int32(ofs, labels.total_bars_outside);
    write_double(ofs, labels.durability_score);
    write_bool(ofs, labels.first_break_returned);
    write_double(ofs, labels.exit_return_rate);
    write_int32(ofs, labels.exits_returned_count);
    write_int32(ofs, labels.exits_stayed_out_count);
    write_bool(ofs, labels.scan_timed_out);
    write_int32(ofs, labels.bars_verified_permanent);

    // TSLA exit events (vectors)
    write_int_vector(ofs, labels.exit_bars);
    write_double_vector(ofs, labels.exit_magnitudes);
    write_int_vector(ofs, labels.exit_durations);
    write_int_vector(ofs, labels.exit_types);
    write_bool_vector(ofs, labels.exit_returned);

    // SPY break scan features
    write_int32(ofs, labels.spy_break_direction);
    write_double(ofs, labels.spy_break_magnitude);
    write_int32(ofs, labels.spy_bars_to_first_break);
    write_bool(ofs, labels.spy_returned_to_channel);
    write_int32(ofs, labels.spy_bounces_after_return);
    write_int32(ofs, labels.spy_round_trip_bounces);
    write_bool(ofs, labels.spy_channel_continued);
    write_int32(ofs, labels.spy_permanent_break_direction);
    write_double(ofs, labels.spy_permanent_break_magnitude);
    write_int32(ofs, labels.spy_bars_to_permanent_break);
    write_int32(ofs, labels.spy_duration_to_permanent);
    write_double(ofs, labels.spy_avg_bars_outside);
    write_int32(ofs, labels.spy_total_bars_outside);
    write_double(ofs, labels.spy_durability_score);
    write_bool(ofs, labels.spy_first_break_returned);
    write_double(ofs, labels.spy_exit_return_rate);
    write_int32(ofs, labels.spy_exits_returned_count);
    write_int32(ofs, labels.spy_exits_stayed_out_count);
    write_bool(ofs, labels.spy_scan_timed_out);
    write_int32(ofs, labels.spy_bars_verified_permanent);

    // SPY exit events (vectors)
    write_int_vector(ofs, labels.spy_exit_bars);
    write_double_vector(ofs, labels.spy_exit_magnitudes);
    write_int_vector(ofs, labels.spy_exit_durations);
    write_int_vector(ofs, labels.spy_exit_types);
    write_bool_vector(ofs, labels.spy_exit_returned);

    // Source channel parameters
    write_double(ofs, labels.source_channel_slope);
    write_double(ofs, labels.source_channel_intercept);
    write_double(ofs, labels.source_channel_std_dev);
    write_double(ofs, labels.source_channel_r_squared);
    write_int32(ofs, labels.source_channel_direction);
    write_int32(ofs, labels.source_channel_bounce_count);
    write_int64(ofs, labels.source_channel_start_ts);
    write_int64(ofs, labels.source_channel_end_ts);
    write_double(ofs, labels.spy_source_channel_slope);
    write_double(ofs, labels.spy_source_channel_intercept);
    write_double(ofs, labels.spy_source_channel_std_dev);
    write_double(ofs, labels.spy_source_channel_r_squared);
    write_int32(ofs, labels.spy_source_channel_direction);
    write_int32(ofs, labels.spy_source_channel_bounce_count);
    write_int64(ofs, labels.spy_source_channel_start_ts);
    write_int64(ofs, labels.spy_source_channel_end_ts);

    // Next channel labels
    write_int32(ofs, labels.best_next_channel_direction);
    write_int32(ofs, labels.best_next_channel_bars_away);
    write_int32(ofs, labels.best_next_channel_duration);
    write_double(ofs, labels.best_next_channel_r_squared);
    write_int32(ofs, labels.best_next_channel_bounce_count);
    write_int32(ofs, labels.shortest_next_channel_direction);
    write_int32(ofs, labels.shortest_next_channel_bars_away);
    write_int32(ofs, labels.shortest_next_channel_duration);
    write_int32(ofs, labels.small_channels_before_best);
    write_int32(ofs, labels.spy_best_next_channel_direction);
    write_int32(ofs, labels.spy_best_next_channel_bars_away);
    write_int32(ofs, labels.spy_best_next_channel_duration);
    write_double(ofs, labels.spy_best_next_channel_r_squared);
    write_int32(ofs, labels.spy_best_next_channel_bounce_count);
    write_int32(ofs, labels.spy_shortest_next_channel_direction);
    write_int32(ofs, labels.spy_shortest_next_channel_bars_away);
    write_int32(ofs, labels.spy_shortest_next_channel_duration);
    write_int32(ofs, labels.spy_small_channels_before_best);

    // RSI labels
    write_double(ofs, labels.rsi_at_first_break);
    write_double(ofs, labels.rsi_at_permanent_break);
    write_double(ofs, labels.rsi_at_channel_end);
    write_bool(ofs, labels.rsi_overbought_at_break);
    write_bool(ofs, labels.rsi_oversold_at_break);
    write_int32(ofs, labels.rsi_divergence_at_break);
    write_int32(ofs, labels.rsi_trend_in_channel);
    write_double(ofs, labels.rsi_range_in_channel);
    write_double(ofs, labels.spy_rsi_at_first_break);
    write_double(ofs, labels.spy_rsi_at_permanent_break);
    write_double(ofs, labels.spy_rsi_at_channel_end);
    write_bool(ofs, labels.spy_rsi_overbought_at_break);
    write_bool(ofs, labels.spy_rsi_oversold_at_break);
    write_int32(ofs, labels.spy_rsi_divergence_at_break);
    write_int32(ofs, labels.spy_rsi_trend_in_channel);
    write_double(ofs, labels.spy_rsi_range_in_channel);

    // Validity flags
    write_bool(ofs, labels.duration_valid);
    write_bool(ofs, labels.direction_valid);
    write_bool(ofs, labels.next_channel_valid);
    write_bool(ofs, labels.break_scan_valid);
}

static ChannelLabels deserialize_channel_labels(std::ifstream& ifs) {
    ChannelLabels labels;

    // Core prediction targets
    labels.duration_bars = read_int32(ifs);
    labels.next_channel_direction = read_int32(ifs);
    labels.permanent_break = read_bool(ifs);
    labels.timeframe = static_cast<Timeframe>(read_int32(ifs));

    // TSLA break scan features
    labels.break_direction = read_int32(ifs);
    labels.break_magnitude = read_double(ifs);
    labels.bars_to_first_break = read_int32(ifs);
    labels.returned_to_channel = read_bool(ifs);
    labels.bounces_after_return = read_int32(ifs);
    labels.round_trip_bounces = read_int32(ifs);
    labels.channel_continued = read_bool(ifs);
    labels.permanent_break_direction = read_int32(ifs);
    labels.permanent_break_magnitude = read_double(ifs);
    labels.bars_to_permanent_break = read_int32(ifs);
    labels.duration_to_permanent = read_int32(ifs);
    labels.avg_bars_outside = read_double(ifs);
    labels.total_bars_outside = read_int32(ifs);
    labels.durability_score = read_double(ifs);
    labels.first_break_returned = read_bool(ifs);
    labels.exit_return_rate = read_double(ifs);
    labels.exits_returned_count = read_int32(ifs);
    labels.exits_stayed_out_count = read_int32(ifs);
    labels.scan_timed_out = read_bool(ifs);
    labels.bars_verified_permanent = read_int32(ifs);

    // TSLA exit events
    labels.exit_bars = read_int_vector(ifs);
    labels.exit_magnitudes = read_double_vector(ifs);
    labels.exit_durations = read_int_vector(ifs);
    labels.exit_types = read_int_vector(ifs);
    labels.exit_returned = read_bool_vector(ifs);

    // SPY break scan features
    labels.spy_break_direction = read_int32(ifs);
    labels.spy_break_magnitude = read_double(ifs);
    labels.spy_bars_to_first_break = read_int32(ifs);
    labels.spy_returned_to_channel = read_bool(ifs);
    labels.spy_bounces_after_return = read_int32(ifs);
    labels.spy_round_trip_bounces = read_int32(ifs);
    labels.spy_channel_continued = read_bool(ifs);
    labels.spy_permanent_break_direction = read_int32(ifs);
    labels.spy_permanent_break_magnitude = read_double(ifs);
    labels.spy_bars_to_permanent_break = read_int32(ifs);
    labels.spy_duration_to_permanent = read_int32(ifs);
    labels.spy_avg_bars_outside = read_double(ifs);
    labels.spy_total_bars_outside = read_int32(ifs);
    labels.spy_durability_score = read_double(ifs);
    labels.spy_first_break_returned = read_bool(ifs);
    labels.spy_exit_return_rate = read_double(ifs);
    labels.spy_exits_returned_count = read_int32(ifs);
    labels.spy_exits_stayed_out_count = read_int32(ifs);
    labels.spy_scan_timed_out = read_bool(ifs);
    labels.spy_bars_verified_permanent = read_int32(ifs);

    // SPY exit events
    labels.spy_exit_bars = read_int_vector(ifs);
    labels.spy_exit_magnitudes = read_double_vector(ifs);
    labels.spy_exit_durations = read_int_vector(ifs);
    labels.spy_exit_types = read_int_vector(ifs);
    labels.spy_exit_returned = read_bool_vector(ifs);

    // Source channel parameters
    labels.source_channel_slope = read_double(ifs);
    labels.source_channel_intercept = read_double(ifs);
    labels.source_channel_std_dev = read_double(ifs);
    labels.source_channel_r_squared = read_double(ifs);
    labels.source_channel_direction = read_int32(ifs);
    labels.source_channel_bounce_count = read_int32(ifs);
    labels.source_channel_start_ts = read_int64(ifs);
    labels.source_channel_end_ts = read_int64(ifs);
    labels.spy_source_channel_slope = read_double(ifs);
    labels.spy_source_channel_intercept = read_double(ifs);
    labels.spy_source_channel_std_dev = read_double(ifs);
    labels.spy_source_channel_r_squared = read_double(ifs);
    labels.spy_source_channel_direction = read_int32(ifs);
    labels.spy_source_channel_bounce_count = read_int32(ifs);
    labels.spy_source_channel_start_ts = read_int64(ifs);
    labels.spy_source_channel_end_ts = read_int64(ifs);

    // Next channel labels
    labels.best_next_channel_direction = read_int32(ifs);
    labels.best_next_channel_bars_away = read_int32(ifs);
    labels.best_next_channel_duration = read_int32(ifs);
    labels.best_next_channel_r_squared = read_double(ifs);
    labels.best_next_channel_bounce_count = read_int32(ifs);
    labels.shortest_next_channel_direction = read_int32(ifs);
    labels.shortest_next_channel_bars_away = read_int32(ifs);
    labels.shortest_next_channel_duration = read_int32(ifs);
    labels.small_channels_before_best = read_int32(ifs);
    labels.spy_best_next_channel_direction = read_int32(ifs);
    labels.spy_best_next_channel_bars_away = read_int32(ifs);
    labels.spy_best_next_channel_duration = read_int32(ifs);
    labels.spy_best_next_channel_r_squared = read_double(ifs);
    labels.spy_best_next_channel_bounce_count = read_int32(ifs);
    labels.spy_shortest_next_channel_direction = read_int32(ifs);
    labels.spy_shortest_next_channel_bars_away = read_int32(ifs);
    labels.spy_shortest_next_channel_duration = read_int32(ifs);
    labels.spy_small_channels_before_best = read_int32(ifs);

    // RSI labels
    labels.rsi_at_first_break = read_double(ifs);
    labels.rsi_at_permanent_break = read_double(ifs);
    labels.rsi_at_channel_end = read_double(ifs);
    labels.rsi_overbought_at_break = read_bool(ifs);
    labels.rsi_oversold_at_break = read_bool(ifs);
    labels.rsi_divergence_at_break = read_int32(ifs);
    labels.rsi_trend_in_channel = read_int32(ifs);
    labels.rsi_range_in_channel = read_double(ifs);
    labels.spy_rsi_at_first_break = read_double(ifs);
    labels.spy_rsi_at_permanent_break = read_double(ifs);
    labels.spy_rsi_at_channel_end = read_double(ifs);
    labels.spy_rsi_overbought_at_break = read_bool(ifs);
    labels.spy_rsi_oversold_at_break = read_bool(ifs);
    labels.spy_rsi_divergence_at_break = read_int32(ifs);
    labels.spy_rsi_trend_in_channel = read_int32(ifs);
    labels.spy_rsi_range_in_channel = read_double(ifs);

    // Validity flags
    labels.duration_valid = read_bool(ifs);
    labels.direction_valid = read_bool(ifs);
    labels.next_channel_valid = read_bool(ifs);
    labels.break_scan_valid = read_bool(ifs);

    return labels;
}

// =============================================================================
// FEATURE NAME TABLE SERIALIZATION (v3)
// =============================================================================

static void write_feature_name_table(std::ofstream& ofs, const FeatureNameTable& table) {
    write_uint32(ofs, static_cast<uint32_t>(table.size()));
    for (const auto& name : table.names()) {
        write_string(ofs, name);
    }
}

static FeatureNameTable read_feature_name_table(std::ifstream& ifs) {
    FeatureNameTable table;
    uint32_t count = read_uint32(ifs);
    for (uint32_t i = 0; i < count; ++i) {
        std::string name = read_string(ifs);
        table.add_name(name);
    }
    return table;
}

// =============================================================================
// SAMPLE SERIALIZATION
// =============================================================================

// v2 format: string keys for features
static void serialize_sample_v2(std::ofstream& ofs, const ChannelSample& sample) {
    // Core sample data
    write_int64(ofs, sample.timestamp);
    write_int32(ofs, sample.channel_end_idx);
    write_int32(ofs, sample.best_window);

    // Features (tf_features map) - string keys
    write_uint32(ofs, static_cast<uint32_t>(sample.tf_features.size()));
    for (const auto& pair : sample.tf_features) {
        write_string(ofs, pair.first);
        write_double(ofs, pair.second);
    }

    // Labels per window
    write_uint32(ofs, static_cast<uint32_t>(sample.labels_per_window.size()));
    for (const auto& window_pair : sample.labels_per_window) {
        write_int32(ofs, window_pair.first);  // window size
        write_uint32(ofs, static_cast<uint32_t>(window_pair.second.size()));  // tf count
        for (const auto& tf_pair : window_pair.second) {
            write_string(ofs, tf_pair.first);  // timeframe string
            serialize_channel_labels(ofs, tf_pair.second);
        }
    }

    // Bar metadata
    write_uint32(ofs, static_cast<uint32_t>(sample.bar_metadata.size()));
    for (const auto& tf_pair : sample.bar_metadata) {
        write_string(ofs, tf_pair.first);  // timeframe string
        write_uint32(ofs, static_cast<uint32_t>(tf_pair.second.size()));  // metadata count
        for (const auto& meta_pair : tf_pair.second) {
            write_string(ofs, meta_pair.first);  // metadata key
            write_double(ofs, meta_pair.second);  // metadata value
        }
    }
}

// v3 format: index-based features using feature name table
static void serialize_sample_v3(std::ofstream& ofs, const ChannelSample& sample,
                                 const FeatureNameTable& table) {
    // Core sample data
    write_int64(ofs, sample.timestamp);
    write_int32(ofs, sample.channel_end_idx);
    write_int32(ofs, sample.best_window);

    // Features (tf_features map) - index-based
    write_uint32(ofs, static_cast<uint32_t>(sample.tf_features.size()));
    for (const auto& pair : sample.tf_features) {
        uint16_t index = table.get_index(pair.first);
        write_uint16(ofs, index);
        write_double(ofs, pair.second);
    }

    // Labels per window (unchanged from v2)
    write_uint32(ofs, static_cast<uint32_t>(sample.labels_per_window.size()));
    for (const auto& window_pair : sample.labels_per_window) {
        write_int32(ofs, window_pair.first);  // window size
        write_uint32(ofs, static_cast<uint32_t>(window_pair.second.size()));  // tf count
        for (const auto& tf_pair : window_pair.second) {
            write_string(ofs, tf_pair.first);  // timeframe string
            serialize_channel_labels(ofs, tf_pair.second);
        }
    }

    // Bar metadata (unchanged from v2)
    write_uint32(ofs, static_cast<uint32_t>(sample.bar_metadata.size()));
    for (const auto& tf_pair : sample.bar_metadata) {
        write_string(ofs, tf_pair.first);  // timeframe string
        write_uint32(ofs, static_cast<uint32_t>(tf_pair.second.size()));  // metadata count
        for (const auto& meta_pair : tf_pair.second) {
            write_string(ofs, meta_pair.first);  // metadata key
            write_double(ofs, meta_pair.second);  // metadata value
        }
    }
}

// Legacy alias for backward compatibility
static void serialize_sample(std::ofstream& ofs, const ChannelSample& sample) {
    serialize_sample_v2(ofs, sample);
}

// v2 format: string keys for features
static ChannelSample deserialize_sample_v2(std::ifstream& ifs) {
    ChannelSample sample;

    // Core sample data
    sample.timestamp = read_int64(ifs);
    sample.channel_end_idx = read_int32(ifs);
    sample.best_window = read_int32(ifs);

    // Features - string keys
    uint32_t feature_count = read_uint32(ifs);
    sample.tf_features.reserve(feature_count);
    for (uint32_t i = 0; i < feature_count; ++i) {
        std::string key = read_string(ifs);
        double value = read_double(ifs);
        sample.tf_features[key] = value;
    }

    // Labels per window
    uint32_t window_count = read_uint32(ifs);
    for (uint32_t i = 0; i < window_count; ++i) {
        int32_t window_size = read_int32(ifs);
        uint32_t tf_count = read_uint32(ifs);
        for (uint32_t j = 0; j < tf_count; ++j) {
            std::string tf_key = read_string(ifs);
            ChannelLabels labels = deserialize_channel_labels(ifs);
            sample.labels_per_window[window_size][tf_key] = labels;
        }
    }

    // Bar metadata
    uint32_t metadata_tf_count = read_uint32(ifs);
    for (uint32_t i = 0; i < metadata_tf_count; ++i) {
        std::string tf_key = read_string(ifs);
        uint32_t meta_count = read_uint32(ifs);
        for (uint32_t j = 0; j < meta_count; ++j) {
            std::string meta_key = read_string(ifs);
            double meta_value = read_double(ifs);
            sample.bar_metadata[tf_key][meta_key] = meta_value;
        }
    }

    return sample;
}

// v3 format: index-based features using feature name table
static ChannelSample deserialize_sample_v3(std::ifstream& ifs, const FeatureNameTable& table) {
    ChannelSample sample;

    // Core sample data
    sample.timestamp = read_int64(ifs);
    sample.channel_end_idx = read_int32(ifs);
    sample.best_window = read_int32(ifs);

    // Features - index-based
    uint32_t feature_count = read_uint32(ifs);
    sample.tf_features.reserve(feature_count);
    for (uint32_t i = 0; i < feature_count; ++i) {
        uint16_t index = read_uint16(ifs);
        double value = read_double(ifs);
        const std::string& key = table.get_name(index);
        sample.tf_features[key] = value;
    }

    // Labels per window (unchanged from v2)
    uint32_t window_count = read_uint32(ifs);
    for (uint32_t i = 0; i < window_count; ++i) {
        int32_t window_size = read_int32(ifs);
        uint32_t tf_count = read_uint32(ifs);
        for (uint32_t j = 0; j < tf_count; ++j) {
            std::string tf_key = read_string(ifs);
            ChannelLabels labels = deserialize_channel_labels(ifs);
            sample.labels_per_window[window_size][tf_key] = labels;
        }
    }

    // Bar metadata (unchanged from v2)
    uint32_t metadata_tf_count = read_uint32(ifs);
    for (uint32_t i = 0; i < metadata_tf_count; ++i) {
        std::string tf_key = read_string(ifs);
        uint32_t meta_count = read_uint32(ifs);
        for (uint32_t j = 0; j < meta_count; ++j) {
            std::string meta_key = read_string(ifs);
            double meta_value = read_double(ifs);
            sample.bar_metadata[tf_key][meta_key] = meta_value;
        }
    }

    return sample;
}

// Legacy alias for backward compatibility
static ChannelSample deserialize_sample(std::ifstream& ifs) {
    return deserialize_sample_v2(ifs);
}

// =============================================================================
// PUBLIC API IMPLEMENTATION
// =============================================================================

void save_samples(const std::vector<ChannelSample>& samples, const std::string& filename) {
    if (samples.empty()) {
        throw std::runtime_error("Cannot save empty sample vector");
    }

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    try {
        // Write header
        ofs.write(reinterpret_cast<const char*>(MAGIC_BYTES), sizeof(MAGIC_BYTES));
        write_uint32(ofs, FORMAT_VERSION);  // v3
        write_uint64(ofs, static_cast<uint64_t>(samples.size()));

        // Calculate average feature count for metadata
        uint32_t avg_features = 0;
        if (!samples.empty()) {
            size_t total = 0;
            for (const auto& sample : samples) {
                total += sample.feature_count();
            }
            avg_features = static_cast<uint32_t>(total / samples.size());
        }
        write_uint32(ofs, avg_features);

        // Build and write feature name table from first sample (v3)
        FeatureNameTable feature_table;
        feature_table.build_from_sample(samples[0]);
        write_feature_name_table(ofs, feature_table);

        // Write samples using v3 format (index-based features)
        for (const auto& sample : samples) {
            serialize_sample_v3(ofs, sample, feature_table);
        }

        ofs.close();
        if (!ofs) {
            throw std::runtime_error("Error writing to file: " + filename);
        }

    } catch (const std::exception& e) {
        ofs.close();
        throw std::runtime_error(std::string("Serialization error: ") + e.what());
    }
}

std::vector<ChannelSample> load_samples(const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    try {
        // Read and validate header
        uint8_t magic[8];
        ifs.read(reinterpret_cast<char*>(magic), sizeof(magic));
        if (!ifs || std::memcmp(magic, MAGIC_BYTES, sizeof(MAGIC_BYTES)) != 0) {
            throw std::runtime_error("Invalid file format: magic bytes mismatch");
        }

        uint32_t version = read_uint32(ifs);
        if (version != FORMAT_VERSION && version != FORMAT_VERSION_V2) {
            throw std::runtime_error("Unsupported format version: " + std::to_string(version));
        }

        uint64_t num_samples = read_uint64(ifs);
        uint32_t num_features = read_uint32(ifs);  // Not strictly validated, just metadata

        // Read samples based on version
        std::vector<ChannelSample> samples;
        samples.reserve(num_samples);

        if (version == FORMAT_VERSION) {
            // v3: Read feature name table first
            FeatureNameTable feature_table = read_feature_name_table(ifs);

            for (uint64_t i = 0; i < num_samples; ++i) {
                samples.push_back(deserialize_sample_v3(ifs, feature_table));
            }
        } else {
            // v2: Read samples with string keys
            for (uint64_t i = 0; i < num_samples; ++i) {
                samples.push_back(deserialize_sample_v2(ifs));
            }
        }

        ifs.close();
        return samples;

    } catch (const std::exception& e) {
        ifs.close();
        throw std::runtime_error(std::string("Deserialization error: ") + e.what());
    }
}

bool validate_sample_file(const std::string& filename) {
    try {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) return false;

        // Check magic bytes
        uint8_t magic[8];
        ifs.read(reinterpret_cast<char*>(magic), sizeof(magic));
        if (!ifs || std::memcmp(magic, MAGIC_BYTES, sizeof(MAGIC_BYTES)) != 0) {
            return false;
        }

        // Check version (support both v2 and v3)
        uint32_t version = read_uint32(ifs);
        if (version != FORMAT_VERSION && version != FORMAT_VERSION_V2) {
            return false;
        }

        return true;
    } catch (...) {
        return false;
    }
}

bool get_file_metadata(const std::string& filename,
                       uint32_t& version,
                       uint64_t& num_samples,
                       uint32_t& num_features) {
    try {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs) return false;

        // Check magic bytes
        uint8_t magic[8];
        ifs.read(reinterpret_cast<char*>(magic), sizeof(magic));
        if (!ifs || std::memcmp(magic, MAGIC_BYTES, sizeof(MAGIC_BYTES)) != 0) {
            return false;
        }

        // Read metadata
        version = read_uint32(ifs);
        num_samples = read_uint64(ifs);
        num_features = read_uint32(ifs);

        return true;
    } catch (...) {
        return false;
    }
}

// =============================================================================
// STREAMING SAMPLE WRITER IMPLEMENTATION
// =============================================================================

StreamingSampleWriter::StreamingSampleWriter(const std::string& filename, size_t flush_interval)
    : filename_(filename)
    , flush_interval_(flush_interval)
    , is_open_(false)
    , samples_written_(0)
    , total_features_(0)
    , samples_since_flush_(0)
    , count_position_(0)
    , features_position_(0)
    , feature_table_written_(false)
{
}

StreamingSampleWriter::~StreamingSampleWriter() {
    if (is_open_) {
        try {
            close();
        } catch (...) {
            // Ignore errors in destructor
        }
    }
}

StreamingSampleWriter::StreamingSampleWriter(StreamingSampleWriter&& other) noexcept
    : filename_(std::move(other.filename_))
    , flush_interval_(other.flush_interval_)
    , ofs_(std::move(other.ofs_))
    , is_open_(other.is_open_)
    , samples_written_(other.samples_written_)
    , total_features_(other.total_features_)
    , samples_since_flush_(other.samples_since_flush_)
    , count_position_(other.count_position_)
    , features_position_(other.features_position_)
    , feature_table_(std::move(other.feature_table_))
    , feature_table_written_(other.feature_table_written_)
{
    other.is_open_ = false;
    other.samples_written_ = 0;
    other.total_features_ = 0;
    other.feature_table_written_ = false;
}

StreamingSampleWriter& StreamingSampleWriter::operator=(StreamingSampleWriter&& other) noexcept {
    if (this != &other) {
        if (is_open_) {
            try { close(); } catch (...) {}
        }
        filename_ = std::move(other.filename_);
        flush_interval_ = other.flush_interval_;
        ofs_ = std::move(other.ofs_);
        is_open_ = other.is_open_;
        samples_written_ = other.samples_written_;
        total_features_ = other.total_features_;
        samples_since_flush_ = other.samples_since_flush_;
        count_position_ = other.count_position_;
        features_position_ = other.features_position_;
        feature_table_ = std::move(other.feature_table_);
        feature_table_written_ = other.feature_table_written_;

        other.is_open_ = false;
        other.samples_written_ = 0;
        other.total_features_ = 0;
        other.feature_table_written_ = false;
    }
    return *this;
}

void StreamingSampleWriter::open() {
    if (is_open_) {
        throw std::runtime_error("StreamingSampleWriter already open");
    }

    ofs_.open(filename_, std::ios::binary | std::ios::trunc);
    if (!ofs_) {
        throw std::runtime_error("Failed to open file for writing: " + filename_);
    }

    // Write header (v3 format)
    ofs_.write(reinterpret_cast<const char*>(MAGIC_BYTES), sizeof(MAGIC_BYTES));
    write_uint32(ofs_, FORMAT_VERSION);

    // Save position of sample count (we'll update it on close)
    count_position_ = ofs_.tellp();
    write_uint64(ofs_, 0);  // Placeholder for sample count

    // Save position of avg features (we'll update it on close)
    features_position_ = ofs_.tellp();
    write_uint32(ofs_, 0);  // Placeholder for avg features

    // Note: Feature name table will be written after first sample

    if (!ofs_) {
        throw std::runtime_error("Failed to write header to: " + filename_);
    }

    is_open_ = true;
    samples_written_ = 0;
    total_features_ = 0;
    samples_since_flush_ = 0;
    feature_table_.clear();
    feature_table_written_ = false;
}

void StreamingSampleWriter::write_sample_internal(const ChannelSample& sample) {
    // On first sample, build and write the feature name table
    if (!feature_table_written_) {
        feature_table_.build_from_sample(sample);
        write_feature_name_table(ofs_, feature_table_);
        feature_table_written_ = true;
    }

    // Use v3 format: write sample with index-based features
    serialize_sample_v3(ofs_, sample, feature_table_);
}

void StreamingSampleWriter::write(const ChannelSample& sample) {
    if (!is_open_) {
        throw std::runtime_error("StreamingSampleWriter not open");
    }

    write_sample_internal(sample);

    samples_written_++;
    total_features_ += sample.feature_count();
    samples_since_flush_++;

    // Flush periodically
    if (flush_interval_ > 0 && samples_since_flush_ >= flush_interval_) {
        flush();
    }

    if (!ofs_) {
        throw std::runtime_error("Error writing sample to: " + filename_);
    }
}

void StreamingSampleWriter::write_batch(const std::vector<ChannelSample>& samples) {
    if (!is_open_) {
        throw std::runtime_error("StreamingSampleWriter not open");
    }

    for (const auto& sample : samples) {
        write_sample_internal(sample);
        samples_written_++;
        total_features_ += sample.feature_count();
    }

    samples_since_flush_ += samples.size();

    // Flush periodically
    if (flush_interval_ > 0 && samples_since_flush_ >= flush_interval_) {
        flush();
    }

    if (!ofs_) {
        throw std::runtime_error("Error writing batch to: " + filename_);
    }
}

void StreamingSampleWriter::flush() {
    if (is_open_ && ofs_) {
        ofs_.flush();
        samples_since_flush_ = 0;
    }
}

void StreamingSampleWriter::close() {
    if (!is_open_) {
        return;
    }

    // Flush any remaining data
    flush();

    // Seek back to update sample count in header
    ofs_.seekp(count_position_);
    write_uint64(ofs_, samples_written_);

    // Update avg features
    ofs_.seekp(features_position_);
    uint32_t avg_features = samples_written_ > 0
        ? static_cast<uint32_t>(total_features_ / samples_written_)
        : 0;
    write_uint32(ofs_, avg_features);

    // Close file
    ofs_.close();
    is_open_ = false;

    if (!ofs_) {
        throw std::runtime_error("Error finalizing file: " + filename_);
    }
}

} // namespace v15
