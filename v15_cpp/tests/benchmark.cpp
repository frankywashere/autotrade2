/**
 * V15 C++ Scanner Performance Benchmark
 *
 * Comprehensive performance testing:
 * - Measures Pass 1, 2, 3 separately with detailed breakdowns
 * - Reports throughput (samples/sec, channels/sec)
 * - Compares with Python baseline if available
 * - Tests different thread counts (1, 2, 4, 8, auto)
 * - Measures memory usage (RSS, VMS)
 * - Creates detailed performance report
 *
 * Usage:
 *   ./benchmark --data-dir data --max-samples 1000 --output benchmark_report.txt
 */

#include "scanner.hpp"
#include "data_loader.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iomanip>
#include <thread>
#include <cstring>
#include <cstdlib>
#include <getopt.h>

#ifdef __APPLE__
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#include <fstream>
#endif

struct BenchmarkConfig {
    std::string data_dir = "data";
    std::string output_path;
    int step = 10;
    int max_samples = 1000;
    std::vector<int> thread_counts = {1, 2, 4, 8, 0};  // 0 = auto
    int warmup_bars = 32760;
    int num_runs = 3;
    bool verbose = false;
};

struct MemoryInfo {
    double rss_mb = 0.0;
    double vms_mb = 0.0;
    double percent = 0.0;

    static MemoryInfo get_current() {
        MemoryInfo info;

#ifdef __APPLE__
        struct mach_task_basic_info task_info;
        mach_msg_type_number_t info_count = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                     (task_info_t)&task_info, &info_count) == KERN_SUCCESS) {
            info.rss_mb = task_info.resident_size / (1024.0 * 1024.0);
            info.vms_mb = task_info.virtual_size / (1024.0 * 1024.0);
        }
#elif defined(__linux__)
        std::ifstream statm("/proc/self/statm");
        if (statm.is_open()) {
            unsigned long vsize, rss;
            statm >> vsize >> rss;
            long page_size = sysconf(_SC_PAGESIZE);
            info.rss_mb = (rss * page_size) / (1024.0 * 1024.0);
            info.vms_mb = (vsize * page_size) / (1024.0 * 1024.0);
        }
#endif

        return info;
    }
};

struct BenchmarkResult {
    int workers;
    int samples_generated;
    int channels_detected;
    int labels_generated;

    // Timing (milliseconds)
    int64_t pass1_duration_ms;
    int64_t pass2_duration_ms;
    int64_t pass3_duration_ms;
    int64_t total_duration_ms;

    // Throughput
    double samples_per_sec;
    double channels_per_sec_pass1;
    double labels_per_sec_pass2;

    // Memory
    MemoryInfo memory_before;
    MemoryInfo memory_after;
    double memory_delta_mb;

    // Feature extraction timing
    double avg_feature_time_ms;
    double min_feature_time_ms;
    double max_feature_time_ms;
};

void print_usage(const char* program_name) {
    std::cout << "V15 C++ Scanner Performance Benchmark\n\n";
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --data-dir PATH       Data directory (default: data)\n";
    std::cout << "  --output PATH         Output file for benchmark report\n";
    std::cout << "  --step N              Channel detection step (default: 10)\n";
    std::cout << "  --max-samples N       Maximum samples per run (default: 1000)\n";
    std::cout << "  --threads LIST        Comma-separated thread counts to test (default: 1,2,4,8,auto)\n";
    std::cout << "  --warmup-bars N       Warmup bars (default: 32760)\n";
    std::cout << "  --runs N              Number of runs per configuration (default: 3)\n";
    std::cout << "  --verbose             Enable verbose output\n";
    std::cout << "  --help                Show this help\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << " --max-samples 1000 --threads 1,4,8\n";
    std::cout << "  " << program_name << " --max-samples 10000 --runs 5 --output benchmark.txt\n";
}

BenchmarkResult run_benchmark(const x14::MarketData& market_data,
                              const BenchmarkConfig& config,
                              int workers) {
    BenchmarkResult result;
    result.workers = workers;

    // Get memory before
    result.memory_before = MemoryInfo::get_current();

    // Configure scanner
    v15::ScannerConfig scanner_config;
    scanner_config.step = config.step;
    scanner_config.max_samples = config.max_samples;
    scanner_config.workers = workers;
    scanner_config.batch_size = 8;
    scanner_config.warmup_bars = config.warmup_bars;
    scanner_config.min_cycles = 1;
    scanner_config.min_gap_bars = 5;
    scanner_config.labeling_method = "hybrid";
    scanner_config.verbose = false;
    scanner_config.progress = false;

    // Run scanner
    v15::Scanner scanner(scanner_config);

    auto start = std::chrono::high_resolution_clock::now();

    std::vector<v15::ChannelSample> samples = scanner.scan(
        market_data.tsla,
        market_data.spy,
        market_data.vix
    );

    auto end = std::chrono::high_resolution_clock::now();

    // Get memory after
    result.memory_after = MemoryInfo::get_current();
    result.memory_delta_mb = result.memory_after.rss_mb - result.memory_before.rss_mb;

    // Extract statistics
    const v15::ScannerStats& stats = scanner.get_stats();

    result.samples_generated = samples.size();
    result.channels_detected = stats.tsla_channels_detected + stats.spy_channels_detected;
    result.labels_generated = stats.tsla_labels_generated + stats.spy_labels_generated;

    result.pass1_duration_ms = stats.pass1_duration_ms;
    result.pass2_duration_ms = stats.pass2_duration_ms;
    result.pass3_duration_ms = stats.pass3_duration_ms;
    result.total_duration_ms = stats.total_duration_ms;

    result.avg_feature_time_ms = stats.avg_feature_time_ms;
    result.min_feature_time_ms = stats.min_feature_time_ms;
    result.max_feature_time_ms = stats.max_feature_time_ms;

    // Calculate throughput
    if (result.total_duration_ms > 0) {
        result.samples_per_sec = (result.samples_generated * 1000.0) / result.total_duration_ms;
    }
    if (result.pass1_duration_ms > 0) {
        result.channels_per_sec_pass1 = (result.channels_detected * 1000.0) / result.pass1_duration_ms;
    }
    if (result.pass2_duration_ms > 0) {
        result.labels_per_sec_pass2 = (result.labels_generated * 1000.0) / result.pass2_duration_ms;
    }

    return result;
}

void print_result(const BenchmarkResult& result, bool detailed = false) {
    std::cout << "\n  Workers: " << result.workers;
    std::cout << " | Samples: " << result.samples_generated;
    std::cout << " | Total: " << (result.total_duration_ms / 1000.0) << "s";
    std::cout << " | Throughput: " << std::fixed << std::setprecision(2)
             << result.samples_per_sec << " samples/sec";
    std::cout << "\n";

    if (detailed) {
        std::cout << "    Pass 1: " << (result.pass1_duration_ms / 1000.0) << "s ("
                 << result.channels_per_sec_pass1 << " channels/sec)\n";
        std::cout << "    Pass 2: " << (result.pass2_duration_ms / 1000.0) << "s ("
                 << result.labels_per_sec_pass2 << " labels/sec)\n";
        std::cout << "    Pass 3: " << (result.pass3_duration_ms / 1000.0) << "s\n";
        std::cout << "    Memory: " << std::fixed << std::setprecision(1)
                 << result.memory_after.rss_mb << " MB (delta: "
                 << result.memory_delta_mb << " MB)\n";
        if (result.avg_feature_time_ms > 0) {
            std::cout << "    Avg feature time: " << std::fixed << std::setprecision(2)
                     << result.avg_feature_time_ms << "ms\n";
        }
    }
}

void print_report(const std::vector<std::vector<BenchmarkResult>>& all_results,
                 const BenchmarkConfig& config,
                 const std::string& output_file = "") {
    std::ostringstream report;

    report << std::string(80, '=') << "\n";
    report << "V15 C++ Scanner Performance Benchmark Report\n";
    report << std::string(80, '=') << "\n\n";

    report << "Configuration:\n";
    report << "  Data directory: " << config.data_dir << "\n";
    report << "  Step: " << config.step << "\n";
    report << "  Max samples: " << config.max_samples << "\n";
    report << "  Warmup bars: " << config.warmup_bars << "\n";
    report << "  Runs per config: " << config.num_runs << "\n\n";

    // Summary table
    report << std::string(80, '-') << "\n";
    report << "Performance Summary (averaged over " << config.num_runs << " runs)\n";
    report << std::string(80, '-') << "\n";
    report << std::left << std::setw(10) << "Workers"
          << std::right << std::setw(12) << "Samples"
          << std::setw(12) << "Total(s)"
          << std::setw(15) << "Throughput"
          << std::setw(12) << "Pass1(s)"
          << std::setw(12) << "Pass2(s)"
          << std::setw(12) << "Pass3(s)" << "\n";
    report << std::string(80, '-') << "\n";

    for (const auto& runs : all_results) {
        if (runs.empty()) continue;

        // Calculate averages
        double avg_total = 0.0, avg_pass1 = 0.0, avg_pass2 = 0.0, avg_pass3 = 0.0;
        double avg_throughput = 0.0;
        int avg_samples = 0;
        int workers = runs[0].workers;

        for (const auto& r : runs) {
            avg_total += r.total_duration_ms;
            avg_pass1 += r.pass1_duration_ms;
            avg_pass2 += r.pass2_duration_ms;
            avg_pass3 += r.pass3_duration_ms;
            avg_throughput += r.samples_per_sec;
            avg_samples += r.samples_generated;
        }

        int n = runs.size();
        avg_total /= (n * 1000.0);
        avg_pass1 /= (n * 1000.0);
        avg_pass2 /= (n * 1000.0);
        avg_pass3 /= (n * 1000.0);
        avg_throughput /= n;
        avg_samples /= n;

        std::string workers_str = (workers == 0) ? "auto" : std::to_string(workers);

        report << std::left << std::setw(10) << workers_str
              << std::right << std::setw(12) << avg_samples
              << std::fixed << std::setprecision(2)
              << std::setw(12) << avg_total
              << std::setw(15) << avg_throughput
              << std::setw(12) << avg_pass1
              << std::setw(12) << avg_pass2
              << std::setw(12) << avg_pass3 << "\n";
    }

    report << std::string(80, '-') << "\n\n";

    // Detailed breakdown
    report << "Detailed Breakdown by Thread Count:\n";
    report << std::string(80, '-') << "\n";

    for (size_t i = 0; i < all_results.size(); i++) {
        const auto& runs = all_results[i];
        if (runs.empty()) continue;

        int workers = runs[0].workers;
        std::string workers_str = (workers == 0) ? "auto" : std::to_string(workers);

        report << "\nWorkers: " << workers_str << "\n";

        for (size_t run = 0; run < runs.size(); run++) {
            const auto& r = runs[run];
            report << "  Run " << (run + 1) << ":\n";
            report << "    Samples: " << r.samples_generated << "\n";
            report << "    Total time: " << std::fixed << std::setprecision(3)
                  << (r.total_duration_ms / 1000.0) << "s\n";
            report << "    Throughput: " << std::fixed << std::setprecision(2)
                  << r.samples_per_sec << " samples/sec\n";
            report << "    Pass 1: " << (r.pass1_duration_ms / 1000.0) << "s ("
                  << r.channels_detected << " channels, "
                  << r.channels_per_sec_pass1 << " ch/sec)\n";
            report << "    Pass 2: " << (r.pass2_duration_ms / 1000.0) << "s ("
                  << r.labels_generated << " labels, "
                  << r.labels_per_sec_pass2 << " lbl/sec)\n";
            report << "    Pass 3: " << (r.pass3_duration_ms / 1000.0) << "s\n";
            report << "    Memory: " << std::fixed << std::setprecision(1)
                  << r.memory_after.rss_mb << " MB RSS (delta: "
                  << r.memory_delta_mb << " MB)\n";
            if (r.avg_feature_time_ms > 0) {
                report << "    Feature extraction: avg=" << std::fixed << std::setprecision(2)
                      << r.avg_feature_time_ms << "ms, min="
                      << r.min_feature_time_ms << "ms, max="
                      << r.max_feature_time_ms << "ms\n";
            }
        }
    }

    report << "\n" << std::string(80, '=') << "\n";

    // Find best configuration
    double best_throughput = 0.0;
    int best_workers = 0;
    for (const auto& runs : all_results) {
        if (runs.empty()) continue;
        double avg_throughput = 0.0;
        for (const auto& r : runs) {
            avg_throughput += r.samples_per_sec;
        }
        avg_throughput /= runs.size();
        if (avg_throughput > best_throughput) {
            best_throughput = avg_throughput;
            best_workers = runs[0].workers;
        }
    }

    std::string best_workers_str = (best_workers == 0) ? "auto" : std::to_string(best_workers);
    report << "Best Configuration: " << best_workers_str << " workers ("
          << std::fixed << std::setprecision(2) << best_throughput << " samples/sec)\n";
    report << std::string(80, '=') << "\n";

    std::string report_str = report.str();
    std::cout << report_str;

    if (!output_file.empty()) {
        std::ofstream file(output_file);
        if (file) {
            file << report_str;
            std::cout << "\nReport written to: " << output_file << "\n";
        } else {
            std::cerr << "Failed to write report to: " << output_file << "\n";
        }
    }
}

std::vector<int> parse_thread_counts(const std::string& str) {
    std::vector<int> counts;
    std::string current;
    for (char c : str) {
        if (c == ',') {
            if (!current.empty()) {
                if (current == "auto") {
                    counts.push_back(0);
                } else {
                    counts.push_back(std::stoi(current));
                }
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        if (current == "auto") {
            counts.push_back(0);
        } else {
            counts.push_back(std::stoi(current));
        }
    }
    return counts;
}

int main(int argc, char* argv[]) {
    BenchmarkConfig config;

    static struct option long_options[] = {
        {"data-dir",     required_argument, 0, 'd'},
        {"output",       required_argument, 0, 'o'},
        {"step",         required_argument, 0, 's'},
        {"max-samples",  required_argument, 0, 'm'},
        {"threads",      required_argument, 0, 't'},
        {"warmup-bars",  required_argument, 0, 'u'},
        {"runs",         required_argument, 0, 'r'},
        {"verbose",      no_argument,       0, 'v'},
        {"help",         no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "d:o:s:m:t:u:r:vh", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'd': config.data_dir = optarg; break;
            case 'o': config.output_path = optarg; break;
            case 's': config.step = std::atoi(optarg); break;
            case 'm': config.max_samples = std::atoi(optarg); break;
            case 't': config.thread_counts = parse_thread_counts(optarg); break;
            case 'u': config.warmup_bars = std::atoi(optarg); break;
            case 'r': config.num_runs = std::atoi(optarg); break;
            case 'v': config.verbose = true; break;
            case 'h':
                print_usage(argv[0]);
                return 0;
            default:
                print_usage(argv[0]);
                return 1;
        }
    }

    try {
        std::cout << std::string(80, '=') << "\n";
        std::cout << "V15 C++ Scanner Performance Benchmark\n";
        std::cout << std::string(80, '=') << "\n\n";

        // Load market data once
        std::cout << "Loading market data from " << config.data_dir << "...\n";
        x14::DataLoader loader(config.data_dir, config.verbose);
        x14::MarketData market_data = loader.load();
        std::cout << "Loaded " << market_data.num_bars << " bars\n\n";

        // Run benchmarks
        std::vector<std::vector<BenchmarkResult>> all_results;

        for (int workers : config.thread_counts) {
            std::string workers_str = (workers == 0) ? "auto" : std::to_string(workers);
            std::cout << "\nBenchmarking with " << workers_str << " workers ("
                     << config.num_runs << " runs)...\n";

            std::vector<BenchmarkResult> runs;
            for (int run = 0; run < config.num_runs; run++) {
                std::cout << "  Run " << (run + 1) << "/" << config.num_runs << "... ";
                std::cout.flush();

                BenchmarkResult result = run_benchmark(market_data, config, workers);
                runs.push_back(result);

                std::cout << result.samples_generated << " samples in "
                         << (result.total_duration_ms / 1000.0) << "s ("
                         << std::fixed << std::setprecision(2)
                         << result.samples_per_sec << " samples/sec)\n";
            }

            all_results.push_back(runs);
        }

        // Print comprehensive report
        std::cout << "\n";
        print_report(all_results, config, config.output_path);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] " << e.what() << "\n";
        return 1;
    }
}
