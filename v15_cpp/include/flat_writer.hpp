#pragma once

/**
 * Flat Format Writer
 *
 * Writes samples directly to .flat directory format for instant training.
 * Output structure:
 *   output.flat/
 *     features.npy      - [N x F] float32
 *     feature_names.json
 *     meta.json
 *     labels/
 *       duration.npy, direction.npy, etc.
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <cstdint>

namespace v15 {

// Forward declarations
struct ChannelSample;

class FlatWriter {
public:
    /**
     * Create a flat writer.
     * @param output_dir Directory to create (e.g., "data.flat")
     * @param target_tf Target timeframe for labels (e.g., "daily")
     */
    explicit FlatWriter(const std::string& output_dir, const std::string& target_tf = "daily");
    ~FlatWriter();

    /**
     * Add a sample to the writer.
     * Features and labels are accumulated in memory.
     */
    void write(const ChannelSample& sample);

    /**
     * Finalize and write all data to disk.
     * Must be called after all samples are added.
     */
    void close();

    size_t samples_written() const { return num_samples_; }

private:
    std::string output_dir_;
    std::string target_tf_;
    size_t num_samples_;
    bool closed_;

    // Feature data
    std::vector<std::string> feature_names_;
    std::unordered_map<std::string, size_t> feature_name_to_idx_;
    std::vector<float> features_;  // [N * F] flattened
    size_t num_features_;
    bool feature_names_initialized_;

    // Label arrays (accumulated)
    std::vector<float> L_duration;
    std::vector<int64_t> L_direction;
    std::vector<int64_t> L_new_channel;
    std::vector<uint8_t> L_permanent_break;
    std::vector<uint8_t> L_valid;
    std::vector<uint8_t> L_duration_valid;
    std::vector<uint8_t> L_direction_valid;

    // TSLA break scan
    std::vector<float> L_tsla_bars_to_first_break;
    std::vector<int64_t> L_tsla_break_direction;
    std::vector<float> L_tsla_break_magnitude;
    std::vector<float> L_tsla_bounces_after_return;
    std::vector<float> L_tsla_duration_to_permanent;
    std::vector<float> L_tsla_avg_bars_outside;
    std::vector<float> L_tsla_total_bars_outside;
    std::vector<float> L_tsla_durability_score;
    std::vector<float> L_tsla_exit_return_rate;
    std::vector<float> L_tsla_exits_returned_count;
    std::vector<float> L_tsla_exits_stayed_out_count;
    std::vector<float> L_tsla_bars_verified_permanent;

    // TSLA RSI
    std::vector<float> L_tsla_rsi_at_first_break;
    std::vector<float> L_tsla_rsi_at_permanent_break;
    std::vector<float> L_tsla_rsi_at_channel_end;
    std::vector<int64_t> L_tsla_rsi_overbought_at_break;
    std::vector<int64_t> L_tsla_rsi_oversold_at_break;
    std::vector<int64_t> L_tsla_rsi_divergence_at_break;
    std::vector<int64_t> L_tsla_rsi_trend_in_channel;
    std::vector<float> L_tsla_rsi_range_in_channel;

    // SPY break scan
    std::vector<float> L_spy_bars_to_first_break;
    std::vector<int64_t> L_spy_break_direction;
    std::vector<float> L_spy_break_magnitude;
    std::vector<float> L_spy_bounces_after_return;
    std::vector<float> L_spy_duration_to_permanent;
    std::vector<float> L_spy_avg_bars_outside;
    std::vector<float> L_spy_total_bars_outside;
    std::vector<float> L_spy_durability_score;
    std::vector<float> L_spy_exit_return_rate;
    std::vector<float> L_spy_exits_returned_count;
    std::vector<float> L_spy_exits_stayed_out_count;
    std::vector<float> L_spy_bars_verified_permanent;

    // SPY RSI
    std::vector<float> L_spy_rsi_at_first_break;
    std::vector<float> L_spy_rsi_at_permanent_break;
    std::vector<float> L_spy_rsi_at_channel_end;
    std::vector<int64_t> L_spy_rsi_overbought_at_break;
    std::vector<int64_t> L_spy_rsi_oversold_at_break;
    std::vector<int64_t> L_spy_rsi_divergence_at_break;
    std::vector<int64_t> L_spy_rsi_trend_in_channel;
    std::vector<float> L_spy_rsi_range_in_channel;

    // Cross-correlation
    std::vector<int64_t> L_cross_who_broke_first;
    std::vector<float> L_cross_break_lag_bars;
    std::vector<float> L_cross_magnitude_spread;
    std::vector<float> L_cross_durability_spread;

    // Per-TF duration (10 timeframes)
    std::vector<float> L_per_tf_duration;  // [N * 10] flattened
    std::vector<uint8_t> L_per_tf_duration_valid;  // [N * 10] flattened

    // Per-TF direction (10 timeframes)
    std::vector<int64_t> L_per_tf_direction;  // [N * 10] flattened
    std::vector<uint8_t> L_per_tf_direction_valid;  // [N * 10] flattened

    void initialize_feature_names(const ChannelSample& sample);
    void extract_labels(const ChannelSample& sample);
    void write_features_npy();
    void write_labels_npy();
    void write_metadata();
};

} // namespace v15
