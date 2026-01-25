/**
 * V15 Channel Scanner - Main Entry Point
 *
 * Command-line interface for the V15 channel scanner.
 * Matches Python scanner.py functionality and output format.
 */

#include "scanner.hpp"
#include "data_loader.hpp"
#include "serialization.hpp"
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <cstdlib>
#include <getopt.h>
#include <chrono>

void print_usage(const char* program_name) {
    std::cout << "V15 Channel Scanner - CHANNEL-END SAMPLING Architecture\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --step N              Step size for channel detection in Pass 1 (default: 10)\n";
    std::cout << "  --max-samples N       Maximum number of samples to generate (default: unlimited)\n";
    std::cout << "  --output PATH         Output file path for samples (default: none)\n";
    std::cout << "  --workers N           Number of worker threads (default: auto-detect)\n";
    std::cout << "  --batch-size N        Channels per batch for parallel processing (default: 8)\n";
    std::cout << "  --no-parallel         Disable parallel processing (use 1 worker)\n";
    std::cout << "  --data-dir PATH       Data directory path (default: data)\n";
    std::cout << "  --min-cycles N        Minimum cycles for valid channel (default: 1)\n";
    std::cout << "  --min-gap-bars N      Minimum gap between channels (default: 5)\n";
    std::cout << "  --warmup-bars N       Minimum 5min bars before first sample (default: 32760)\n";
    std::cout << "  --labeling-method M   Label generation method (default: hybrid)\n";
    std::cout << "  --quiet               Disable verbose output\n";
    std::cout << "  --no-progress         Disable progress bar\n";
    std::cout << "  --streaming           Enable streaming mode (write to disk as generated, default: on)\n";
    std::cout << "  --no-streaming        Disable streaming (accumulate in memory, may cause OOM)\n";
    std::cout << "  --flush-interval N    Samples between disk flushes in streaming mode (default: 1000)\n";
    std::cout << "  --help                Show this help message\n\n";
    std::cout << "Architecture:\n";
    std::cout << "  - Each detected channel produces exactly ONE sample\n";
    std::cout << "  - Sample position = channel end position\n";
    std::cout << "  - --step controls channel detection spacing, not sample spacing\n\n";
    std::cout << "Streaming Mode:\n";
    std::cout << "  - Enabled by default when --output is specified\n";
    std::cout << "  - Writes samples directly to disk to avoid memory exhaustion\n";
    std::cout << "  - Essential for large scans (step=1) that would otherwise OOM\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " --step 1 --output samples.bin --workers 8  # Streaming enabled\n";
    std::cout << "  " << program_name << " --step 10 --max-samples 1000 --no-streaming  # In-memory\n\n";
}

int main(int argc, char* argv[]) {
    // Default configuration
    v15::ScannerConfig config;
    std::string data_dir = "data";
    bool no_parallel = false;

    // Command-line options
    static struct option long_options[] = {
        {"step",            required_argument, 0, 's'},
        {"max-samples",     required_argument, 0, 'm'},
        {"output",          required_argument, 0, 'o'},
        {"workers",         required_argument, 0, 'w'},
        {"batch-size",      required_argument, 0, 'b'},
        {"no-parallel",     no_argument,       0, 'n'},
        {"data-dir",        required_argument, 0, 'd'},
        {"min-cycles",      required_argument, 0, 'c'},
        {"min-gap-bars",    required_argument, 0, 'g'},
        {"warmup-bars",     required_argument, 0, 'u'},
        {"labeling-method", required_argument, 0, 'l'},
        {"quiet",           no_argument,       0, 'q'},
        {"no-progress",     no_argument,       0, 'p'},
        {"streaming",       no_argument,       0, 'S'},
        {"no-streaming",    no_argument,       0, 'N'},
        {"flush-interval",  required_argument, 0, 'f'},
        {"help",            no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "s:m:o:w:b:nd:c:g:u:l:qpSNf:h", long_options, &option_index)) != -1) {
        switch (opt) {
            case 's':
                config.step = std::atoi(optarg);
                break;
            case 'm':
                config.max_samples = std::atoi(optarg);
                break;
            case 'o':
                config.output_path = optarg;
                break;
            case 'w':
                config.workers = std::atoi(optarg);
                break;
            case 'b':
                config.batch_size = std::atoi(optarg);
                break;
            case 'n':
                no_parallel = true;
                break;
            case 'd':
                data_dir = optarg;
                break;
            case 'c':
                config.min_cycles = std::atoi(optarg);
                break;
            case 'g':
                config.min_gap_bars = std::atoi(optarg);
                break;
            case 'u':
                config.warmup_bars = std::atoi(optarg);
                break;
            case 'l':
                config.labeling_method = optarg;
                break;
            case 'q':
                config.verbose = false;
                break;
            case 'p':
                config.progress = false;
                break;
            case 'S':
                config.streaming = true;
                break;
            case 'N':
                config.streaming = false;
                break;
            case 'f':
                config.flush_interval = std::atoi(optarg);
                break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    // Override workers if no-parallel is set
    if (no_parallel) {
        config.workers = 1;
        config.batch_size = 1;
    }

    try {
        // Print configuration
        std::cout << std::string(60, '=') << "\n";
        std::cout << "V15 Channel Scanner - CHANNEL-END SAMPLING Architecture\n";
        std::cout << std::string(60, '=') << "\n\n";
        std::cout << "Architecture: ONE sample per detected channel at channel END\n";
        std::cout << "  - Each channel produces exactly one sample\n";
        std::cout << "  - Sample position = channel end position\n";
        std::cout << "  - --step controls channel detection step, not sample step\n\n";
        // Determine if streaming will be used
        bool will_stream = config.streaming && !config.output_path.empty();

        std::cout << "Configuration:\n";
        std::cout << "  Channel detection step: " << config.step << "\n";
        std::cout << "  Max samples: " << (config.max_samples > 0 ? std::to_string(config.max_samples) : "unlimited") << "\n";
        std::cout << "  Output file: " << (config.output_path.empty() ? "none" : config.output_path) << "\n";
        std::cout << "  Workers: " << (config.workers > 0 ? std::to_string(config.workers) : "auto-detect") << "\n";
        std::cout << "  Batch size: " << config.batch_size << "\n";
        std::cout << "  Min cycles: " << config.min_cycles << "\n";
        std::cout << "  Min gap bars: " << config.min_gap_bars << "\n";
        std::cout << "  Warmup bars: " << config.warmup_bars << "\n";
        std::cout << "  Labeling method: " << config.labeling_method << "\n";
        std::cout << "  Streaming mode: " << (will_stream ? "ENABLED" : "disabled") << "\n";
        if (will_stream) {
            std::cout << "  Flush interval: " << config.flush_interval << " samples\n";
        }
        std::cout << "\n";

        // Load market data
        std::cout << "Loading market data from " << data_dir << "...\n";
        v15::DataLoader loader(data_dir, true);
        v15::MarketData market_data = loader.load();

        std::cout << "Loaded " << market_data.num_bars << " bars\n";

        // Print date range
        char start_str[32], end_str[32];
        std::strftime(start_str, sizeof(start_str), "%Y-%m-%d %H:%M:%S", std::localtime(&market_data.start_time));
        std::strftime(end_str, sizeof(end_str), "%Y-%m-%d %H:%M:%S", std::localtime(&market_data.end_time));
        std::cout << "Date range: " << start_str << " to " << end_str << "\n\n";

        // Create scanner
        v15::Scanner scanner(config);

        // Run scan
        std::cout << "Running 3-PASS scan (channel detection step=" << config.step << ")...\n";
        std::vector<v15::ChannelSample> samples = scanner.scan(
            market_data.tsla,
            market_data.spy,
            market_data.vix
        );

        // Print results
        std::cout << "\nGenerated " << samples.size() << " samples\n";

        if (!samples.empty()) {
            const v15::ChannelSample& first = samples[0];
            std::cout << "\nFirst sample details:\n";
            std::cout << "  Timestamp: " << first.timestamp << "\n";
            std::cout << "  Channel end idx: " << first.channel_end_idx << "\n";
            std::cout << "  Best window: " << first.best_window << "\n";
            std::cout << "  Feature count: " << first.feature_count() << "\n";
            std::cout << "  Label count: " << first.label_count() << "\n";

            // Print sample feature names (first 10)
            if (first.feature_count() > 0) {
                std::cout << "\n  Sample feature names (first 10):\n";
                int count = 0;
                for (const auto& pair : first.tf_features) {
                    std::cout << "    - " << pair.first << ": " << std::fixed << std::setprecision(4)
                              << pair.second << "\n";
                    if (++count >= 10) break;
                }
            }

            const v15::ChannelSample& last = samples[samples.size() - 1];
            std::cout << "\nLast sample timestamp: " << last.timestamp << "\n";
        }

        // Save if output path specified (skip if streaming was used - already saved)
        bool used_streaming = config.streaming && !config.output_path.empty();
        if (!config.output_path.empty() && !used_streaming && !samples.empty()) {
            std::cout << "\nSaving " << samples.size() << " samples to " << config.output_path << "...\n";

            auto save_start = std::chrono::high_resolution_clock::now();

            try {
                v15::save_samples(samples, config.output_path);

                auto save_end = std::chrono::high_resolution_clock::now();
                auto save_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    save_end - save_start).count();

                std::cout << "  Successfully saved in " << (save_duration / 1000.0) << "s\n";
            } catch (const std::exception& e) {
                std::cerr << "\n[ERROR] Failed to save samples: " << e.what() << "\n";
                return 1;
            }
        }

        // Verify output file (whether from streaming or batch save)
        if (!config.output_path.empty()) {
            uint32_t version;
            uint64_t num_samples;
            uint32_t num_features;
            if (v15::get_file_metadata(config.output_path, version, num_samples, num_features)) {
                std::cout << "\nOutput file verification:\n";
                std::cout << "  File: " << config.output_path << "\n";
                std::cout << "  Samples: " << num_samples << "\n";
                std::cout << "  Avg features: " << num_features << "\n";
                std::cout << "  Format version: " << version << "\n";
            }
        }

        // Print final stats
        const v15::ScannerStats& stats = scanner.get_stats();
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "FINAL STATISTICS:\n";
        std::cout << std::string(60, '=') << "\n";
        std::cout << "  Pass 1: " << stats.tsla_channels_detected << " TSLA + "
                  << stats.spy_channels_detected << " SPY channels in "
                  << (stats.pass1_duration_ms / 1000.0) << "s\n";
        std::cout << "  Pass 2: " << stats.tsla_labels_generated << " TSLA + "
                  << stats.spy_labels_generated << " SPY labels in "
                  << (stats.pass2_duration_ms / 1000.0) << "s\n";
        std::cout << "  Pass 3: " << stats.samples_created << " samples in "
                  << (stats.pass3_duration_ms / 1000.0) << "s\n";
        std::cout << "  Total: " << (stats.total_duration_ms / 1000.0) << "s\n";
        std::cout << std::string(60, '=') << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}
