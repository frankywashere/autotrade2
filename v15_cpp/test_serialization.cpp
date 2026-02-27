/**
 * Test program for binary serialization
 */

#include "sample.hpp"
#include "serialization.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "Testing Binary Serialization\n";
    std::cout << std::string(60, '=') << "\n\n";

    try {
        // Create test samples
        std::vector<v15::ChannelSample> samples;

        for (int i = 0; i < 5; ++i) {
            v15::ChannelSample sample;
            sample.timestamp = 1640000000000LL + i * 60000;  // 1 min apart
            sample.channel_end_idx = 1000 + i * 10;
            sample.best_window = 50;

            // Add some features
            sample.set_feature("5min_rsi", 50.0 + i);
            sample.set_feature("1h_macd", 0.5 * i);
            sample.set_feature("daily_volume_ratio", 1.2 + 0.1 * i);

            // Add some labels
            v15::ChannelLabels labels;
            labels.duration_bars = 10 + i;
            labels.next_channel_direction = i % 3;
            labels.permanent_break = (i % 2 == 0);
            labels.timeframe = v15::Timeframe::HOUR_1;
            labels.break_direction = i % 2;
            labels.break_magnitude = 2.5 + 0.5 * i;
            labels.bars_to_first_break = 5 + i;
            labels.source_channel_slope = 0.01 * i;
            labels.source_channel_intercept = 100.0 + i;
            labels.source_channel_std_dev = 2.0;
            labels.source_channel_r_squared = 0.85 + 0.01 * i;
            labels.duration_valid = true;
            labels.direction_valid = true;

            sample.set_labels(50, "1h", labels);

            // Add bar metadata
            sample.set_bar_metadata("1h", "partial_bar_pct", 0.75);
            sample.set_bar_metadata("1h", "bars_since_session_open", 10.0);

            samples.push_back(sample);
        }

        std::cout << "Created " << samples.size() << " test samples\n";
        std::cout << "First sample:\n";
        std::cout << "  Timestamp: " << samples[0].timestamp << "\n";
        std::cout << "  Channel end idx: " << samples[0].channel_end_idx << "\n";
        std::cout << "  Features: " << samples[0].feature_count() << "\n";
        std::cout << "  Labels: " << samples[0].label_count() << "\n\n";

        // Save to file
        std::string filename = "test_samples.bin";
        std::cout << "Saving to " << filename << "...\n";
        v15::save_samples(samples, filename);
        std::cout << "  Success!\n\n";

        // Validate file
        std::cout << "Validating file format...\n";
        if (v15::validate_sample_file(filename)) {
            std::cout << "  Valid!\n\n";
        } else {
            std::cout << "  INVALID!\n\n";
            return 1;
        }

        // Get metadata
        uint32_t version;
        uint64_t num_samples;
        uint32_t num_features;
        std::cout << "Reading file metadata...\n";
        if (v15::get_file_metadata(filename, version, num_samples, num_features)) {
            std::cout << "  Version: " << version << "\n";
            std::cout << "  Num samples: " << num_samples << "\n";
            std::cout << "  Avg features: " << num_features << "\n\n";
        }

        // Load samples back
        std::cout << "Loading samples from file...\n";
        std::vector<v15::ChannelSample> loaded = v15::load_samples(filename);
        std::cout << "  Loaded " << loaded.size() << " samples\n\n";

        // Verify data integrity
        std::cout << "Verifying data integrity...\n";
        bool all_match = true;

        if (loaded.size() != samples.size()) {
            std::cout << "  ERROR: Sample count mismatch!\n";
            all_match = false;
        }

        for (size_t i = 0; i < std::min(loaded.size(), samples.size()); ++i) {
            const auto& orig = samples[i];
            const auto& load = loaded[i];

            if (orig.timestamp != load.timestamp) {
                std::cout << "  ERROR: Sample " << i << " timestamp mismatch\n";
                all_match = false;
            }

            if (orig.channel_end_idx != load.channel_end_idx) {
                std::cout << "  ERROR: Sample " << i << " channel_end_idx mismatch\n";
                all_match = false;
            }

            if (orig.best_window != load.best_window) {
                std::cout << "  ERROR: Sample " << i << " best_window mismatch\n";
                all_match = false;
            }

            if (orig.feature_count() != load.feature_count()) {
                std::cout << "  ERROR: Sample " << i << " feature count mismatch\n";
                all_match = false;
            }

            // Check a few specific features
            double orig_rsi = orig.get_feature("5min_rsi");
            double load_rsi = load.get_feature("5min_rsi");
            if (std::abs(orig_rsi - load_rsi) > 1e-6) {
                std::cout << "  ERROR: Sample " << i << " feature '5min_rsi' mismatch\n";
                all_match = false;
            }

            // Check labels
            const auto* orig_labels = orig.get_labels(50, "1h");
            const auto* load_labels = load.get_labels(50, "1h");

            if (orig_labels && load_labels) {
                if (orig_labels->duration_bars != load_labels->duration_bars) {
                    std::cout << "  ERROR: Sample " << i << " duration_bars mismatch\n";
                    all_match = false;
                }
                if (std::abs(orig_labels->source_channel_slope - load_labels->source_channel_slope) > 1e-9) {
                    std::cout << "  ERROR: Sample " << i << " source_channel_slope mismatch\n";
                    all_match = false;
                }
            } else if (orig_labels || load_labels) {
                std::cout << "  ERROR: Sample " << i << " labels presence mismatch\n";
                all_match = false;
            }
        }

        if (all_match) {
            std::cout << "  All data matches perfectly!\n\n";
        } else {
            std::cout << "  Data integrity check FAILED!\n\n";
            return 1;
        }

        // Print detailed comparison for first sample
        std::cout << "Detailed comparison for first sample:\n";
        std::cout << std::string(60, '-') << "\n";
        const auto& first_orig = samples[0];
        const auto& first_load = loaded[0];

        std::cout << "Timestamp:        " << first_orig.timestamp << " -> " << first_load.timestamp << "\n";
        std::cout << "Channel end idx:  " << first_orig.channel_end_idx << " -> " << first_load.channel_end_idx << "\n";
        std::cout << "Best window:      " << first_orig.best_window << " -> " << first_load.best_window << "\n";

        std::cout << "\nFeatures:\n";
        for (const auto& pair : first_orig.tf_features) {
            double loaded_val = first_load.get_feature(pair.first);
            std::cout << "  " << std::left << std::setw(20) << pair.first
                      << ": " << std::fixed << std::setprecision(6)
                      << pair.second << " -> " << loaded_val << "\n";
        }

        std::cout << "\nLabels (window=50, tf=1h):\n";
        const auto* orig_labels = first_orig.get_labels(50, "1h");
        const auto* load_labels = first_load.get_labels(50, "1h");
        if (orig_labels && load_labels) {
            std::cout << "  Duration bars:    " << orig_labels->duration_bars
                      << " -> " << load_labels->duration_bars << "\n";
            std::cout << "  Break direction:  " << orig_labels->break_direction
                      << " -> " << load_labels->break_direction << "\n";
            std::cout << "  Break magnitude:  " << std::fixed << std::setprecision(4)
                      << orig_labels->break_magnitude
                      << " -> " << load_labels->break_magnitude << "\n";
        }

        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "SERIALIZATION TEST PASSED!\n";
        std::cout << std::string(60, '=') << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}
