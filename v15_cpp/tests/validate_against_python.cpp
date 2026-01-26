/**
 * V15 C++ Scanner Validation Against Python Baseline
 *
 * This program:
 * 1. Loads the same input data as Python scanner
 * 2. Runs C++ scanner with matching configuration
 * 3. Saves output to temporary file for comparison
 * 4. Provides detailed validation report
 *
 * Usage:
 *   ./validate_against_python --data-dir data --output cpp_samples.bin --max-samples 100
 */

#include "scanner.hpp"
#include "data_loader.hpp"
#include "sample.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <getopt.h>
#include <chrono>
#include <iomanip>

struct ValidationConfig {
    std::string data_dir = "data";
    std::string output_path = "cpp_samples.bin";
    int step = 10;
    int max_samples = 100;
    int workers = 4;
    int batch_size = 8;
    int warmup_bars = 32760;
    bool verbose = true;
};

void print_usage(const char* program_name) {
    std::cout << "V15 C++ Scanner Validation Against Python\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --data-dir PATH       Data directory (default: data)\n";
    std::cout << "  --output PATH         Output file for C++ samples (default: cpp_samples.bin)\n";
    std::cout << "  --step N              Channel detection step (default: 10)\n";
    std::cout << "  --max-samples N       Maximum samples to generate (default: 100)\n";
    std::cout << "  --workers N           Worker threads (default: 4)\n";
    std::cout << "  --batch-size N        Batch size (default: 8)\n";
    std::cout << "  --warmup-bars N       Warmup bars (default: 32760)\n";
    std::cout << "  --quiet               Disable verbose output\n";
    std::cout << "  --help                Show this help\n";
}

bool save_samples_to_binary(const std::vector<v15::ChannelSample>& samples,
                            const std::string& filepath) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open output file: " << filepath << "\n";
        return false;
    }

    // Write header: magic number + version + sample count
    uint32_t magic = 0x56313543;  // "V15C"
    uint32_t version = 1;
    uint64_t count = samples.size();

    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&count), sizeof(count));

    // Write each sample
    for (const auto& sample : samples) {
        // Core fields
        file.write(reinterpret_cast<const char*>(&sample.timestamp), sizeof(sample.timestamp));
        file.write(reinterpret_cast<const char*>(&sample.channel_end_idx), sizeof(sample.channel_end_idx));
        file.write(reinterpret_cast<const char*>(&sample.best_window), sizeof(sample.best_window));

        // Features
        uint64_t feature_count = sample.tf_features.size();
        file.write(reinterpret_cast<const char*>(&feature_count), sizeof(feature_count));

        for (const auto& [key, value] : sample.tf_features) {
            uint32_t key_len = key.size();
            file.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
            file.write(key.data(), key_len);
            file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }

        // Labels per window (simplified - just count for validation)
        uint64_t label_window_count = sample.labels_per_window.size();
        file.write(reinterpret_cast<const char*>(&label_window_count), sizeof(label_window_count));

        for (const auto& [window, tf_map] : sample.labels_per_window) {
            file.write(reinterpret_cast<const char*>(&window), sizeof(window));
            uint64_t tf_count = tf_map.size();
            file.write(reinterpret_cast<const char*>(&tf_count), sizeof(tf_count));

            for (const auto& [tf, labels] : tf_map) {
                uint32_t tf_len = tf.size();
                file.write(reinterpret_cast<const char*>(&tf_len), sizeof(tf_len));
                file.write(tf.data(), tf_len);

                // Write label fields
                file.write(reinterpret_cast<const char*>(&labels.direction_valid), sizeof(labels.direction_valid));
                file.write(reinterpret_cast<const char*>(&labels.direction), sizeof(labels.direction));
                file.write(reinterpret_cast<const char*>(&labels.first_break_bar), sizeof(labels.first_break_bar));
                file.write(reinterpret_cast<const char*>(&labels.permanent_break_bar), sizeof(labels.permanent_break_bar));
                file.write(reinterpret_cast<const char*>(&labels.break_magnitude), sizeof(labels.break_magnitude));
                // Add more label fields as needed
            }
        }
    }

    file.close();
    return true;
}

void print_sample_summary(const v15::ChannelSample& sample) {
    std::cout << "\n  Sample Details:\n";
    std::cout << "    Timestamp: " << sample.timestamp << "\n";
    std::cout << "    Channel end idx: " << sample.channel_end_idx << "\n";
    std::cout << "    Best window: " << sample.best_window << "\n";
    std::cout << "    Feature count: " << sample.feature_count() << "\n";
    std::cout << "    Label count: " << sample.label_count() << "\n";

    // Print first 10 features
    if (sample.feature_count() > 0) {
        std::cout << "\n    First 10 features:\n";
        int count = 0;
        for (const auto& [key, value] : sample.tf_features) {
            std::cout << "      " << key << ": " << std::fixed << std::setprecision(8)
                     << value << "\n";
            if (++count >= 10) break;
        }
    }
}

void print_validation_report(const std::vector<v15::ChannelSample>& samples,
                            const v15::ScannerStats& stats) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "VALIDATION REPORT - C++ Scanner Output\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << "\nSample Generation:\n";
    std::cout << "  Total samples: " << samples.size() << "\n";
    std::cout << "  Samples skipped: " << stats.samples_skipped << "\n";
    std::cout << "  Errors: " << stats.errors_encountered << "\n";

    if (!samples.empty()) {
        // Feature count validation
        size_t min_features = samples[0].feature_count();
        size_t max_features = samples[0].feature_count();
        size_t total_features = 0;

        for (const auto& s : samples) {
            size_t fc = s.feature_count();
            min_features = std::min(min_features, fc);
            max_features = std::max(max_features, fc);
            total_features += fc;
        }

        double avg_features = static_cast<double>(total_features) / samples.size();

        std::cout << "\nFeature Statistics:\n";
        std::cout << "  Average features per sample: " << std::fixed << std::setprecision(1)
                 << avg_features << "\n";
        std::cout << "  Min features: " << min_features << "\n";
        std::cout << "  Max features: " << max_features << "\n";

        // Expected feature count (14,840 after adding normalized features)
        const size_t EXPECTED_FEATURES = 14840;
        if (avg_features != EXPECTED_FEATURES) {
            std::cout << "  WARNING: Expected " << EXPECTED_FEATURES
                     << " features per sample!\n";
        } else {
            std::cout << "  PASS: Feature count matches expected ("
                     << EXPECTED_FEATURES << ")\n";
        }

        // Label statistics
        size_t total_labels = 0;
        for (const auto& s : samples) {
            total_labels += s.label_count();
        }
        double avg_labels = static_cast<double>(total_labels) / samples.size();

        std::cout << "\nLabel Statistics:\n";
        std::cout << "  Average labels per sample: " << std::fixed << std::setprecision(1)
                 << avg_labels << "\n";
    }

    std::cout << "\nPerformance:\n";
    std::cout << "  Pass 1 time: " << (stats.pass1_duration_ms / 1000.0) << "s\n";
    std::cout << "  Pass 2 time: " << (stats.pass2_duration_ms / 1000.0) << "s\n";
    std::cout << "  Pass 3 time: " << (stats.pass3_duration_ms / 1000.0) << "s\n";
    std::cout << "  Total time: " << (stats.total_duration_ms / 1000.0) << "s\n";

    if (stats.total_duration_ms > 0 && samples.size() > 0) {
        double samples_per_sec = (samples.size() * 1000.0) / stats.total_duration_ms;
        std::cout << "  Throughput: " << std::fixed << std::setprecision(2)
                 << samples_per_sec << " samples/sec\n";
    }

    std::cout << "\nFirst Sample:" << std::endl;
    if (!samples.empty()) {
        print_sample_summary(samples[0]);
    }

    std::cout << "\nLast Sample:" << std::endl;
    if (!samples.empty()) {
        print_sample_summary(samples[samples.size() - 1]);
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
}

int main(int argc, char* argv[]) {
    ValidationConfig config;

    // Parse command-line options
    static struct option long_options[] = {
        {"data-dir",     required_argument, 0, 'd'},
        {"output",       required_argument, 0, 'o'},
        {"step",         required_argument, 0, 's'},
        {"max-samples",  required_argument, 0, 'm'},
        {"workers",      required_argument, 0, 'w'},
        {"batch-size",   required_argument, 0, 'b'},
        {"warmup-bars",  required_argument, 0, 'u'},
        {"quiet",        no_argument,       0, 'q'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:o:s:m:w:b:u:qh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'd': config.data_dir = optarg; break;
            case 'o': config.output_path = optarg; break;
            case 's': config.step = std::atoi(optarg); break;
            case 'm': config.max_samples = std::atoi(optarg); break;
            case 'w': config.workers = std::atoi(optarg); break;
            case 'b': config.batch_size = std::atoi(optarg); break;
            case 'u': config.warmup_bars = std::atoi(optarg); break;
            case 'q': config.verbose = false; break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    try {
        std::cout << std::string(70, '=') << "\n";
        std::cout << "C++ Scanner Validation Against Python Baseline\n";
        std::cout << std::string(70, '=') << "\n\n";

        std::cout << "Configuration:\n";
        std::cout << "  Data directory: " << config.data_dir << "\n";
        std::cout << "  Output file: " << config.output_path << "\n";
        std::cout << "  Step: " << config.step << "\n";
        std::cout << "  Max samples: " << config.max_samples << "\n";
        std::cout << "  Workers: " << config.workers << "\n";
        std::cout << "  Batch size: " << config.batch_size << "\n";
        std::cout << "  Warmup bars: " << config.warmup_bars << "\n\n";

        // Load market data
        std::cout << "Loading market data from " << config.data_dir << "...\n";
        x14::DataLoader loader(config.data_dir, config.verbose);
        x14::MarketData market_data = loader.load();

        std::cout << "Loaded " << market_data.num_bars << " bars\n";

        // Print date range
        char start_str[32], end_str[32];
        std::strftime(start_str, sizeof(start_str), "%Y-%m-%d %H:%M:%S",
                     std::localtime(&market_data.start_time));
        std::strftime(end_str, sizeof(end_str), "%Y-%m-%d %H:%M:%S",
                     std::localtime(&market_data.end_time));
        std::cout << "Date range: " << start_str << " to " << end_str << "\n\n";

        // Configure scanner to match Python
        v15::ScannerConfig scanner_config;
        scanner_config.step = config.step;
        scanner_config.max_samples = config.max_samples;
        scanner_config.workers = config.workers;
        scanner_config.batch_size = config.batch_size;
        scanner_config.warmup_bars = config.warmup_bars;
        scanner_config.min_cycles = 1;
        scanner_config.min_gap_bars = 5;
        scanner_config.labeling_method = "hybrid";
        scanner_config.verbose = config.verbose;
        scanner_config.progress = true;

        // Create scanner and run
        v15::Scanner scanner(scanner_config);

        std::cout << "Running C++ scanner (3-pass architecture)...\n";
        auto scan_start = std::chrono::high_resolution_clock::now();

        std::vector<v15::ChannelSample> samples = scanner.scan(
            market_data.tsla,
            market_data.spy,
            market_data.vix
        );

        auto scan_end = std::chrono::high_resolution_clock::now();
        auto scan_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            scan_end - scan_start
        ).count();

        std::cout << "\nGenerated " << samples.size() << " samples in "
                 << (scan_duration / 1000.0) << "s\n";

        // Save to binary file
        std::cout << "\nSaving samples to " << config.output_path << "...\n";
        if (save_samples_to_binary(samples, config.output_path)) {
            std::cout << "Successfully saved " << samples.size() << " samples\n";
        } else {
            std::cerr << "Failed to save samples!\n";
            return 1;
        }

        // Print validation report
        print_validation_report(samples, scanner.get_stats());

        std::cout << "\nNext Steps:\n";
        std::cout << "  1. Run Python scanner to generate baseline:\n";
        std::cout << "     python v15/scanner.py --step " << config.step
                 << " --max-samples " << config.max_samples
                 << " --output python_samples.pkl\n\n";
        std::cout << "  2. Run comparison validator:\n";
        std::cout << "     python tests/validate_features.py \\\n";
        std::cout << "       --python python_samples.pkl \\\n";
        std::cout << "       --cpp " << config.output_path << " \\\n";
        std::cout << "       --tolerance 1e-10\n\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}
