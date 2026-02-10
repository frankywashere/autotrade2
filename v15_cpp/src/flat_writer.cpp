/**
 * Flat Format Writer Implementation
 */

#include "flat_writer.hpp"
#include "npy_writer.hpp"
#include "sample.hpp"
#include "feature_array.hpp"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace v15 {

FlatWriter::FlatWriter(const std::string& output_dir, const std::string& target_tf)
    : output_dir_(output_dir)
    , target_tf_(target_tf)
    , num_samples_(0)
    , closed_(false)
    , num_features_(0)
    , feature_names_initialized_(false)
{
    // Create output directory structure
    fs::create_directories(output_dir_);
    fs::create_directories(output_dir_ + "/labels");
}

FlatWriter::~FlatWriter() {
    if (!closed_) {
        try {
            close();
        } catch (...) {}
    }
}

void FlatWriter::initialize_feature_names(const ChannelSample& sample) {
    if (feature_names_initialized_) return;

    // Get sorted feature names from first sample
    for (const auto& [name, value] : sample.tf_features) {
        feature_names_.push_back(name);
    }
    std::sort(feature_names_.begin(), feature_names_.end());

    // Build index map
    for (size_t i = 0; i < feature_names_.size(); ++i) {
        feature_name_to_idx_[feature_names_[i]] = i;
    }

    num_features_ = feature_names_.size();
    feature_names_initialized_ = true;

    std::cout << "  Flat writer initialized: " << num_features_ << " features" << std::endl;
}

void FlatWriter::write(const ChannelSample& sample) {
    if (closed_) {
        throw std::runtime_error("FlatWriter already closed");
    }

    // Initialize feature names from first sample
    if (!feature_names_initialized_) {
        initialize_feature_names(sample);
    }

    // Extract features into flat array
    std::vector<float> sample_features(num_features_, 0.0f);
    for (const auto& [name, value] : sample.tf_features) {
        auto it = feature_name_to_idx_.find(name);
        if (it != feature_name_to_idx_.end()) {
            sample_features[it->second] = static_cast<float>(value);
        }
    }
    features_.insert(features_.end(), sample_features.begin(), sample_features.end());

    // Extract labels
    extract_labels(sample);

    num_samples_++;

    if (num_samples_ % 10000 == 0) {
        std::cout << "  Flat writer: " << num_samples_ << " samples accumulated" << std::endl;
    }
}

void FlatWriter::extract_labels(const ChannelSample& sample) {
    // Find labels for target timeframe and best window
    int window = sample.best_window;

    // Default values
    float duration = 0.0f;
    int64_t direction = 0;
    int64_t new_channel = 1;  // SIDEWAYS
    bool permanent_break = false;
    bool duration_valid = false;
    bool direction_valid = false;

    // TSLA defaults
    float tsla_bars_to_break = 0.0f;
    int64_t tsla_break_dir = 0;
    float tsla_break_mag = 0.0f;
    float tsla_bounces = 0.0f;
    float tsla_dur_to_perm = -1.0f;
    float tsla_avg_outside = 0.0f;
    float tsla_total_outside = 0.0f;
    float tsla_durability = 0.0f;
    float tsla_exit_rate = 0.0f;
    float tsla_exits_returned = 0.0f;
    float tsla_exits_stayed = 0.0f;
    float tsla_bars_verified = 0.0f;
    float tsla_rsi_break = 50.0f;
    float tsla_rsi_perm = 50.0f;
    float tsla_rsi_end = 50.0f;
    int64_t tsla_rsi_ob = 0;
    int64_t tsla_rsi_os = 0;
    int64_t tsla_rsi_div = 0;
    int64_t tsla_rsi_trend = 0;
    float tsla_rsi_range = 0.0f;

    // SPY defaults (same structure)
    float spy_bars_to_break = 0.0f;
    int64_t spy_break_dir = 0;
    float spy_break_mag = 0.0f;
    float spy_bounces = 0.0f;
    float spy_dur_to_perm = -1.0f;
    float spy_avg_outside = 0.0f;
    float spy_total_outside = 0.0f;
    float spy_durability = 0.0f;
    float spy_exit_rate = 0.0f;
    float spy_exits_returned = 0.0f;
    float spy_exits_stayed = 0.0f;
    float spy_bars_verified = 0.0f;
    float spy_rsi_break = 50.0f;
    float spy_rsi_perm = 50.0f;
    float spy_rsi_end = 50.0f;
    int64_t spy_rsi_ob = 0;
    int64_t spy_rsi_os = 0;
    int64_t spy_rsi_div = 0;
    int64_t spy_rsi_trend = 0;
    float spy_rsi_range = 0.0f;

    // Cross-correlation defaults
    int64_t cross_who_first = 0;
    float cross_lag = 0.0f;
    float cross_mag_spread = 0.0f;
    float cross_dur_spread = 0.0f;

    // Per-TF duration and direction (NUM_TFS timeframes)
    std::vector<float> per_tf_dur(FeatureOffsets::NUM_TFS, 0.0f);
    std::vector<uint8_t> per_tf_valid(FeatureOffsets::NUM_TFS, 0);
    std::vector<int64_t> per_tf_dir(FeatureOffsets::NUM_TFS, 0);
    std::vector<uint8_t> per_tf_dir_valid(FeatureOffsets::NUM_TFS, 0);
    std::vector<int64_t> per_tf_nc(FeatureOffsets::NUM_TFS, 1);  // default=sideways
    std::vector<uint8_t> per_tf_nc_valid(FeatureOffsets::NUM_TFS, 0);

    // Try to get labels from the sample
    // Structure: labels_per_window[window][tf] = ChannelLabels (no asset level)
    auto window_it = sample.labels_per_window.find(window);
    if (window_it != sample.labels_per_window.end()) {
        // Look for target TF labels
        auto tf_it = window_it->second.find(target_tf_);
        if (tf_it != window_it->second.end()) {
            const auto& labels = tf_it->second;

            // Core labels
            duration = static_cast<float>(labels.duration_bars);
            direction = labels.break_direction;
            new_channel = labels.next_channel_direction;
            permanent_break = labels.permanent_break;
            duration_valid = labels.duration_valid;
            direction_valid = labels.direction_valid;

            // TSLA break scan labels
            tsla_bars_to_break = static_cast<float>(labels.bars_to_first_break);
            tsla_break_dir = labels.break_direction;
            tsla_break_mag = static_cast<float>(labels.break_magnitude);
            tsla_bounces = static_cast<float>(labels.bounces_after_return);
            tsla_dur_to_perm = labels.duration_to_permanent >= 0 ?
                static_cast<float>(labels.duration_to_permanent) : -1.0f;
            tsla_avg_outside = static_cast<float>(labels.avg_bars_outside);
            tsla_total_outside = static_cast<float>(labels.total_bars_outside);
            tsla_durability = static_cast<float>(labels.durability_score);
            tsla_exit_rate = static_cast<float>(labels.exit_return_rate);
            tsla_exits_returned = static_cast<float>(labels.exits_returned_count);
            tsla_exits_stayed = static_cast<float>(labels.exits_stayed_out_count);
            tsla_bars_verified = static_cast<float>(labels.bars_verified_permanent);

            // TSLA RSI labels
            tsla_rsi_break = static_cast<float>(labels.rsi_at_first_break);
            tsla_rsi_perm = static_cast<float>(labels.rsi_at_permanent_break);
            tsla_rsi_end = static_cast<float>(labels.rsi_at_channel_end);
            tsla_rsi_ob = labels.rsi_overbought_at_break ? 1 : 0;
            tsla_rsi_os = labels.rsi_oversold_at_break ? 1 : 0;
            tsla_rsi_div = labels.rsi_divergence_at_break;
            tsla_rsi_trend = labels.rsi_trend_in_channel;
            tsla_rsi_range = static_cast<float>(labels.rsi_range_in_channel);

            // SPY break scan labels (cross-referenced from SPY channel)
            spy_bars_to_break = static_cast<float>(labels.spy_bars_to_first_break);
            spy_break_dir = labels.spy_break_direction;
            spy_break_mag = static_cast<float>(labels.spy_break_magnitude);
            spy_bounces = static_cast<float>(labels.spy_bounces_after_return);
            spy_dur_to_perm = labels.spy_duration_to_permanent >= 0 ?
                static_cast<float>(labels.spy_duration_to_permanent) : -1.0f;
            spy_avg_outside = static_cast<float>(labels.spy_avg_bars_outside);
            spy_total_outside = static_cast<float>(labels.spy_total_bars_outside);
            spy_durability = static_cast<float>(labels.spy_durability_score);
            spy_exit_rate = static_cast<float>(labels.spy_exit_return_rate);
            spy_exits_returned = static_cast<float>(labels.spy_exits_returned_count);
            spy_exits_stayed = static_cast<float>(labels.spy_exits_stayed_out_count);
            spy_bars_verified = static_cast<float>(labels.spy_bars_verified_permanent);

            // SPY RSI labels
            spy_rsi_break = static_cast<float>(labels.spy_rsi_at_first_break);
            spy_rsi_perm = static_cast<float>(labels.spy_rsi_at_permanent_break);
            spy_rsi_end = static_cast<float>(labels.spy_rsi_at_channel_end);
            spy_rsi_ob = labels.spy_rsi_overbought_at_break ? 1 : 0;
            spy_rsi_os = labels.spy_rsi_oversold_at_break ? 1 : 0;
            spy_rsi_div = labels.spy_rsi_divergence_at_break;
            spy_rsi_trend = labels.spy_rsi_trend_in_channel;
            spy_rsi_range = static_cast<float>(labels.spy_rsi_range_in_channel);

            // Cross-correlation (computed from TSLA/SPY fields)
            int tsla_btb = labels.bars_to_first_break;
            int spy_btb = labels.spy_bars_to_first_break;
            if (tsla_btb < spy_btb) {
                cross_who_first = 1;   // TSLA broke first
            } else if (spy_btb < tsla_btb) {
                cross_who_first = -1;  // SPY broke first
            } else {
                cross_who_first = 0;   // Simultaneous
            }
            cross_lag = static_cast<float>(std::abs(tsla_btb - spy_btb));
            cross_mag_spread = static_cast<float>(labels.break_magnitude - labels.spy_break_magnitude);
            cross_dur_spread = static_cast<float>(labels.durability_score - labels.spy_durability_score);
        }
    }

    // Per-TF duration extraction
    static const std::vector<std::string> TIMEFRAMES = {
        "5min", "15min", "30min", "1h", "2h", "3h", "4h", "daily", "weekly", "monthly"
    };
    // Verify TIMEFRAMES size matches NUM_TFS at runtime (first call only)
    static bool size_checked = false;
    if (!size_checked) {
        if (TIMEFRAMES.size() != FeatureOffsets::NUM_TFS) {
            throw std::runtime_error(
                "TIMEFRAMES size mismatch: expected " +
                std::to_string(FeatureOffsets::NUM_TFS) +
                ", got " + std::to_string(TIMEFRAMES.size())
            );
        }
        size_checked = true;
    }

    if (window_it != sample.labels_per_window.end()) {
        for (size_t i = 0; i < TIMEFRAMES.size(); ++i) {
            auto tf_it = window_it->second.find(TIMEFRAMES[i]);
            if (tf_it != window_it->second.end()) {
                per_tf_dur[i] = static_cast<float>(tf_it->second.duration_bars);
                per_tf_valid[i] = tf_it->second.duration_valid ? 1 : 0;
                per_tf_dir[i] = tf_it->second.break_direction;
                per_tf_dir_valid[i] = tf_it->second.direction_valid ? 1 : 0;
                per_tf_nc[i] = tf_it->second.next_channel_direction;
                per_tf_nc_valid[i] = tf_it->second.next_channel_valid ? 1 : 0;
            }
        }
    }

    // Push all labels
    L_duration.push_back(duration);
    L_direction.push_back(direction);
    L_new_channel.push_back(new_channel);
    L_permanent_break.push_back(permanent_break ? 1 : 0);
    L_valid.push_back((duration_valid || direction_valid) ? 1 : 0);
    L_duration_valid.push_back(duration_valid ? 1 : 0);
    L_direction_valid.push_back(direction_valid ? 1 : 0);

    // TSLA
    L_tsla_bars_to_first_break.push_back(tsla_bars_to_break);
    L_tsla_break_direction.push_back(tsla_break_dir);
    L_tsla_break_magnitude.push_back(tsla_break_mag);
    L_tsla_bounces_after_return.push_back(tsla_bounces);
    L_tsla_duration_to_permanent.push_back(tsla_dur_to_perm);
    L_tsla_avg_bars_outside.push_back(tsla_avg_outside);
    L_tsla_total_bars_outside.push_back(tsla_total_outside);
    L_tsla_durability_score.push_back(tsla_durability);
    L_tsla_exit_return_rate.push_back(tsla_exit_rate);
    L_tsla_exits_returned_count.push_back(tsla_exits_returned);
    L_tsla_exits_stayed_out_count.push_back(tsla_exits_stayed);
    L_tsla_bars_verified_permanent.push_back(tsla_bars_verified);
    L_tsla_rsi_at_first_break.push_back(tsla_rsi_break);
    L_tsla_rsi_at_permanent_break.push_back(tsla_rsi_perm);
    L_tsla_rsi_at_channel_end.push_back(tsla_rsi_end);
    L_tsla_rsi_overbought_at_break.push_back(tsla_rsi_ob);
    L_tsla_rsi_oversold_at_break.push_back(tsla_rsi_os);
    L_tsla_rsi_divergence_at_break.push_back(tsla_rsi_div);
    L_tsla_rsi_trend_in_channel.push_back(tsla_rsi_trend);
    L_tsla_rsi_range_in_channel.push_back(tsla_rsi_range);

    // SPY
    L_spy_bars_to_first_break.push_back(spy_bars_to_break);
    L_spy_break_direction.push_back(spy_break_dir);
    L_spy_break_magnitude.push_back(spy_break_mag);
    L_spy_bounces_after_return.push_back(spy_bounces);
    L_spy_duration_to_permanent.push_back(spy_dur_to_perm);
    L_spy_avg_bars_outside.push_back(spy_avg_outside);
    L_spy_total_bars_outside.push_back(spy_total_outside);
    L_spy_durability_score.push_back(spy_durability);
    L_spy_exit_return_rate.push_back(spy_exit_rate);
    L_spy_exits_returned_count.push_back(spy_exits_returned);
    L_spy_exits_stayed_out_count.push_back(spy_exits_stayed);
    L_spy_bars_verified_permanent.push_back(spy_bars_verified);
    L_spy_rsi_at_first_break.push_back(spy_rsi_break);
    L_spy_rsi_at_permanent_break.push_back(spy_rsi_perm);
    L_spy_rsi_at_channel_end.push_back(spy_rsi_end);
    L_spy_rsi_overbought_at_break.push_back(spy_rsi_ob);
    L_spy_rsi_oversold_at_break.push_back(spy_rsi_os);
    L_spy_rsi_divergence_at_break.push_back(spy_rsi_div);
    L_spy_rsi_trend_in_channel.push_back(spy_rsi_trend);
    L_spy_rsi_range_in_channel.push_back(spy_rsi_range);

    // Cross
    L_cross_who_broke_first.push_back(cross_who_first);
    L_cross_break_lag_bars.push_back(cross_lag);
    L_cross_magnitude_spread.push_back(cross_mag_spread);
    L_cross_durability_spread.push_back(cross_dur_spread);

    // Per-TF
    L_per_tf_duration.insert(L_per_tf_duration.end(), per_tf_dur.begin(), per_tf_dur.end());
    L_per_tf_duration_valid.insert(L_per_tf_duration_valid.end(), per_tf_valid.begin(), per_tf_valid.end());
    L_per_tf_direction.insert(L_per_tf_direction.end(), per_tf_dir.begin(), per_tf_dir.end());
    L_per_tf_direction_valid.insert(L_per_tf_direction_valid.end(), per_tf_dir_valid.begin(), per_tf_dir_valid.end());
    L_per_tf_new_channel.insert(L_per_tf_new_channel.end(), per_tf_nc.begin(), per_tf_nc.end());
    L_per_tf_new_channel_valid.insert(L_per_tf_new_channel_valid.end(), per_tf_nc_valid.begin(), per_tf_nc_valid.end());
}

void FlatWriter::write_features_npy() {
    std::string path = output_dir_ + "/features.npy";
    NpyWriter::write_float32_2d(path, features_, num_samples_, num_features_);
    std::cout << "  Wrote features: [" << num_samples_ << " x " << num_features_ << "] to " << path << std::endl;
}

void FlatWriter::write_labels_npy() {
    std::string labels_dir = output_dir_ + "/labels/";

    // Core labels
    NpyWriter::write_float32_1d(labels_dir + "duration.npy", L_duration);
    NpyWriter::write_int64_1d(labels_dir + "direction.npy", L_direction);
    NpyWriter::write_int64_1d(labels_dir + "new_channel.npy", L_new_channel);
    NpyWriter::write_bool_1d(labels_dir + "permanent_break.npy", L_permanent_break);
    NpyWriter::write_bool_1d(labels_dir + "valid.npy", L_valid);
    NpyWriter::write_bool_1d(labels_dir + "duration_valid.npy", L_duration_valid);
    NpyWriter::write_bool_1d(labels_dir + "direction_valid.npy", L_direction_valid);

    // TSLA labels
    NpyWriter::write_float32_1d(labels_dir + "tsla_bars_to_first_break.npy", L_tsla_bars_to_first_break);
    NpyWriter::write_int64_1d(labels_dir + "tsla_break_direction.npy", L_tsla_break_direction);
    NpyWriter::write_float32_1d(labels_dir + "tsla_break_magnitude.npy", L_tsla_break_magnitude);
    NpyWriter::write_float32_1d(labels_dir + "tsla_bounces_after_return.npy", L_tsla_bounces_after_return);
    NpyWriter::write_float32_1d(labels_dir + "tsla_duration_to_permanent.npy", L_tsla_duration_to_permanent);
    NpyWriter::write_float32_1d(labels_dir + "tsla_avg_bars_outside.npy", L_tsla_avg_bars_outside);
    NpyWriter::write_float32_1d(labels_dir + "tsla_total_bars_outside.npy", L_tsla_total_bars_outside);
    NpyWriter::write_float32_1d(labels_dir + "tsla_durability_score.npy", L_tsla_durability_score);
    NpyWriter::write_float32_1d(labels_dir + "tsla_exit_return_rate.npy", L_tsla_exit_return_rate);
    NpyWriter::write_float32_1d(labels_dir + "tsla_exits_returned_count.npy", L_tsla_exits_returned_count);
    NpyWriter::write_float32_1d(labels_dir + "tsla_exits_stayed_out_count.npy", L_tsla_exits_stayed_out_count);
    NpyWriter::write_float32_1d(labels_dir + "tsla_bars_verified_permanent.npy", L_tsla_bars_verified_permanent);
    NpyWriter::write_float32_1d(labels_dir + "tsla_rsi_at_first_break.npy", L_tsla_rsi_at_first_break);
    NpyWriter::write_float32_1d(labels_dir + "tsla_rsi_at_permanent_break.npy", L_tsla_rsi_at_permanent_break);
    NpyWriter::write_float32_1d(labels_dir + "tsla_rsi_at_channel_end.npy", L_tsla_rsi_at_channel_end);
    NpyWriter::write_int64_1d(labels_dir + "tsla_rsi_overbought_at_break.npy", L_tsla_rsi_overbought_at_break);
    NpyWriter::write_int64_1d(labels_dir + "tsla_rsi_oversold_at_break.npy", L_tsla_rsi_oversold_at_break);
    NpyWriter::write_int64_1d(labels_dir + "tsla_rsi_divergence_at_break.npy", L_tsla_rsi_divergence_at_break);
    NpyWriter::write_int64_1d(labels_dir + "tsla_rsi_trend_in_channel.npy", L_tsla_rsi_trend_in_channel);
    NpyWriter::write_float32_1d(labels_dir + "tsla_rsi_range_in_channel.npy", L_tsla_rsi_range_in_channel);

    // SPY labels
    NpyWriter::write_float32_1d(labels_dir + "spy_bars_to_first_break.npy", L_spy_bars_to_first_break);
    NpyWriter::write_int64_1d(labels_dir + "spy_break_direction.npy", L_spy_break_direction);
    NpyWriter::write_float32_1d(labels_dir + "spy_break_magnitude.npy", L_spy_break_magnitude);
    NpyWriter::write_float32_1d(labels_dir + "spy_bounces_after_return.npy", L_spy_bounces_after_return);
    NpyWriter::write_float32_1d(labels_dir + "spy_duration_to_permanent.npy", L_spy_duration_to_permanent);
    NpyWriter::write_float32_1d(labels_dir + "spy_avg_bars_outside.npy", L_spy_avg_bars_outside);
    NpyWriter::write_float32_1d(labels_dir + "spy_total_bars_outside.npy", L_spy_total_bars_outside);
    NpyWriter::write_float32_1d(labels_dir + "spy_durability_score.npy", L_spy_durability_score);
    NpyWriter::write_float32_1d(labels_dir + "spy_exit_return_rate.npy", L_spy_exit_return_rate);
    NpyWriter::write_float32_1d(labels_dir + "spy_exits_returned_count.npy", L_spy_exits_returned_count);
    NpyWriter::write_float32_1d(labels_dir + "spy_exits_stayed_out_count.npy", L_spy_exits_stayed_out_count);
    NpyWriter::write_float32_1d(labels_dir + "spy_bars_verified_permanent.npy", L_spy_bars_verified_permanent);
    NpyWriter::write_float32_1d(labels_dir + "spy_rsi_at_first_break.npy", L_spy_rsi_at_first_break);
    NpyWriter::write_float32_1d(labels_dir + "spy_rsi_at_permanent_break.npy", L_spy_rsi_at_permanent_break);
    NpyWriter::write_float32_1d(labels_dir + "spy_rsi_at_channel_end.npy", L_spy_rsi_at_channel_end);
    NpyWriter::write_int64_1d(labels_dir + "spy_rsi_overbought_at_break.npy", L_spy_rsi_overbought_at_break);
    NpyWriter::write_int64_1d(labels_dir + "spy_rsi_oversold_at_break.npy", L_spy_rsi_oversold_at_break);
    NpyWriter::write_int64_1d(labels_dir + "spy_rsi_divergence_at_break.npy", L_spy_rsi_divergence_at_break);
    NpyWriter::write_int64_1d(labels_dir + "spy_rsi_trend_in_channel.npy", L_spy_rsi_trend_in_channel);
    NpyWriter::write_float32_1d(labels_dir + "spy_rsi_range_in_channel.npy", L_spy_rsi_range_in_channel);

    // Cross-correlation labels
    NpyWriter::write_int64_1d(labels_dir + "cross_who_broke_first.npy", L_cross_who_broke_first);
    NpyWriter::write_float32_1d(labels_dir + "cross_break_lag_bars.npy", L_cross_break_lag_bars);
    NpyWriter::write_float32_1d(labels_dir + "cross_magnitude_spread.npy", L_cross_magnitude_spread);
    NpyWriter::write_float32_1d(labels_dir + "cross_durability_spread.npy", L_cross_durability_spread);

    // Per-TF duration (2D arrays)
    NpyWriter::write_float32_2d(labels_dir + "per_tf_duration.npy", L_per_tf_duration, num_samples_, FeatureOffsets::NUM_TFS);
    // Per-TF valid needs special handling for bool 2D
    // For now write as 1D and reshape in Python
    NpyWriter::write_bool_1d(labels_dir + "per_tf_duration_valid.npy", L_per_tf_duration_valid);

    // Per-TF direction (2D arrays)
    NpyWriter::write_int64_2d(labels_dir + "per_tf_direction.npy", L_per_tf_direction, num_samples_, FeatureOffsets::NUM_TFS);
    NpyWriter::write_bool_1d(labels_dir + "per_tf_direction_valid.npy", L_per_tf_direction_valid);

    // Per-TF new_channel
    NpyWriter::write_int64_2d(labels_dir + "per_tf_new_channel.npy", L_per_tf_new_channel, num_samples_, FeatureOffsets::NUM_TFS);
    NpyWriter::write_bool_1d(labels_dir + "per_tf_new_channel_valid.npy", L_per_tf_new_channel_valid);

    std::cout << "  Wrote " << 53 << " label arrays to " << labels_dir << std::endl;
}

void FlatWriter::write_metadata() {
    // Write feature_names.json
    std::string fn_path = output_dir_ + "/feature_names.json";
    std::ofstream fn_file(fn_path);
    fn_file << "[\n";
    for (size_t i = 0; i < feature_names_.size(); ++i) {
        fn_file << "  \"" << feature_names_[i] << "\"";
        if (i < feature_names_.size() - 1) fn_file << ",";
        fn_file << "\n";
    }
    fn_file << "]\n";
    fn_file.close();

    // Write meta.json
    std::string meta_path = output_dir_ + "/meta.json";
    std::ofstream meta_file(meta_path);
    meta_file << "{\n";
    meta_file << "  \"num_samples\": " << num_samples_ << ",\n";
    meta_file << "  \"num_features\": " << num_features_ << ",\n";
    meta_file << "  \"target_tf\": \"" << target_tf_ << "\",\n";
    meta_file << "  \"format_version\": 1,\n";
    meta_file << "  \"source\": \"v15_cpp_scanner_flat_output\"\n";
    meta_file << "}\n";
    meta_file.close();

    std::cout << "  Wrote metadata files" << std::endl;
}

void FlatWriter::close() {
    if (closed_) return;

    std::cout << "Finalizing flat output to " << output_dir_ << "..." << std::endl;

    if (num_samples_ == 0) {
        std::cout << "  Warning: No samples to write" << std::endl;
        closed_ = true;
        return;
    }

    write_features_npy();
    write_labels_npy();
    write_metadata();

    std::cout << "Flat output complete: " << num_samples_ << " samples, "
              << num_features_ << " features" << std::endl;

    closed_ = true;
}

} // namespace v15
