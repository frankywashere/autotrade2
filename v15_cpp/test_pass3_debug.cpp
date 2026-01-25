#include "scanner.hpp"
#include "types.hpp"
#include <iostream>
#include <vector>
#include <cmath>

using namespace v15;

// Generate minimal synthetic data
std::vector<OHLCV> generate_simple_data(int num_bars, double base_price = 250.0) {
    std::vector<OHLCV> data;
    data.reserve(num_bars);

    std::time_t start_time = 1609459200;

    for (int i = 0; i < num_bars; ++i) {
        OHLCV bar;
        bar.timestamp = start_time + i * 300;

        // Simple uptrend with oscillation
        double trend = base_price + 0.05 * i;
        double cycle = std::sin(i * 0.1) * 2.0;
        double price = trend + cycle;

        bar.open = price - 0.5;
        bar.high = price + 1.0;
        bar.low = price - 1.0;
        bar.close = price;
        bar.volume = 100000 + i * 100;

        data.push_back(bar);
    }

    return data;
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "Pass 3 Debug Test - Minimal Dataset\n";
    std::cout << "============================================================\n\n";

    // Generate dataset with enough warmup AND forward scan
    // Need: warmup (32760) + some channels + forward scan (21000)
    int total_bars = 60000;  // Enough room for valid channels with labels
    auto tsla_data = generate_simple_data(total_bars, 250.0);
    auto spy_data = generate_simple_data(total_bars, 400.0);
    auto vix_data = generate_simple_data(total_bars, 20.0);

    std::cout << "Generated " << total_bars << " bars of data\n\n";

    // Configure scanner with VERBOSE enabled
    ScannerConfig config;
    config.step = 20;  // Larger step to get fewer channels
    config.workers = 1;  // Sequential for easier debugging
    config.max_samples = 1000;  // Request 1000 samples
    config.verbose = false;  // Disable verbose for cleaner output
    config.progress = false;
    config.warmup_bars = 32760;
    config.batch_size = 8;

    std::cout << "Scanner configuration:\n";
    std::cout << "  Step: " << config.step << "\n";
    std::cout << "  Max samples: " << config.max_samples << "\n";
    std::cout << "  Warmup bars: " << config.warmup_bars << "\n";
    std::cout << "  Verbose: " << (config.verbose ? "true" : "false") << "\n";
    std::cout << "  Batch size: " << config.batch_size << "\n\n";

    try {
        Scanner scanner(config);
        std::vector<ChannelSample> samples = scanner.scan(tsla_data, spy_data, vix_data);

        std::cout << "\n============================================================\n";
        std::cout << "RESULTS\n";
        std::cout << "============================================================\n";

        auto stats = scanner.get_stats();
        std::cout << "Pass 1 - Channels detected:\n";
        std::cout << "  TSLA: " << stats.tsla_channels_detected << "\n";
        std::cout << "  SPY: " << stats.spy_channels_detected << "\n";

        std::cout << "\nPass 2 - Labels generated:\n";
        std::cout << "  TSLA: " << stats.tsla_labels_generated << " (" << stats.tsla_labels_valid << " valid)\n";
        std::cout << "  SPY: " << stats.spy_labels_generated << " (" << stats.spy_labels_valid << " valid)\n";

        std::cout << "\nPass 3 - Sample generation:\n";
        std::cout << "  Channels processed: " << stats.channels_processed << "\n";
        std::cout << "  Samples created: " << stats.samples_created << "\n";
        std::cout << "  Samples skipped: " << stats.samples_skipped << "\n";
        std::cout << "  ACTUAL samples returned: " << samples.size() << "\n";

        std::cout << "\n";
        if (samples.size() == config.max_samples) {
            std::cout << "✓ SUCCESS: Got exactly " << config.max_samples << " samples as requested!\n";
        } else {
            std::cout << "✗ PROBLEM: Requested " << config.max_samples << " samples but got " << samples.size() << "\n";
            std::cout << "  Gap: " << (config.max_samples - samples.size()) << " samples missing\n";
        }

        if (!samples.empty()) {
            std::cout << "\nFirst sample details:\n";
            const auto& s = samples[0];
            std::cout << "  Timestamp: " << s.timestamp << "\n";
            std::cout << "  Channel end idx: " << s.channel_end_idx << "\n";
            std::cout << "  Best window: " << s.best_window << "\n";
            std::cout << "  Features: " << s.tf_features.size() << "\n";
            std::cout << "  Labels per window: " << s.labels_per_window.size() << "\n";
            std::cout << "  Is valid: " << (s.is_valid() ? "true" : "false") << "\n";
        }

        std::cout << "\n✓ Test completed successfully\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERROR: " << e.what() << "\n";
        return 1;
    }
}
