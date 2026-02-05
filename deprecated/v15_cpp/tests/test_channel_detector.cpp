#include "channel_detector.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace v15;

// Helper to generate synthetic price data with a known channel
void generate_test_data(
    std::vector<double>& high,
    std::vector<double>& low,
    std::vector<double>& close,
    int num_bars,
    double slope = 0.5,
    double intercept = 100.0,
    double noise = 1.0
) {
    high.clear();
    low.clear();
    close.clear();

    for (int i = 0; i < num_bars; ++i) {
        // Base price from linear trend
        double base = slope * i + intercept;

        // Add some oscillation to create bounces
        double phase = (i % 20) / 20.0 * 2.0 * M_PI;
        double oscillation = std::sin(phase) * noise * 2.0;

        double c = base + oscillation;
        double h = c + noise * 0.5;
        double l = c - noise * 0.5;

        close.push_back(c);
        high.push_back(h);
        low.push_back(l);
    }
}

int main() {
    std::cout << "=== Channel Detector Test ===" << std::endl;
    std::cout << std::endl;

    // Test 1: Single window detection
    {
        std::cout << "Test 1: Single window detection (window=50)" << std::endl;

        std::vector<double> high, low, close;
        generate_test_data(high, low, close, 100, 0.5, 100.0, 2.0);

        Channel channel = ChannelDetector::detect_channel(
            high, low, close,
            50,   // window
            2.0,  // std_multiplier
            0.10, // touch_threshold
            1     // min_cycles
        );

        std::cout << "  Valid: " << (channel.valid ? "YES" : "NO") << std::endl;
        std::cout << "  Direction: ";
        switch (channel.direction) {
            case ChannelDirection::BULL: std::cout << "BULL"; break;
            case ChannelDirection::BEAR: std::cout << "BEAR"; break;
            case ChannelDirection::SIDEWAYS: std::cout << "SIDEWAYS"; break;
            case ChannelDirection::UNKNOWN: std::cout << "UNKNOWN"; break;
        }
        std::cout << std::endl;

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Slope: " << channel.slope << std::endl;
        std::cout << "  Intercept: " << channel.intercept << std::endl;
        std::cout << "  R²: " << channel.r_squared << std::endl;
        std::cout << "  Std Dev: " << channel.std_dev << std::endl;
        std::cout << "  Width %: " << channel.width_pct << std::endl;
        std::cout << "  Bounces: " << channel.bounce_count << std::endl;
        std::cout << "  Complete Cycles: " << channel.complete_cycles << std::endl;
        std::cout << "  Upper Touches: " << channel.upper_touches << std::endl;
        std::cout << "  Lower Touches: " << channel.lower_touches << std::endl;
        std::cout << "  Alternation Ratio: " << channel.alternation_ratio << std::endl;
        std::cout << "  Quality Score: " << channel.quality_score << std::endl;

        std::cout << std::endl;
    }

    // Test 2: Multi-window detection
    {
        std::cout << "Test 2: Multi-window detection (8 windows)" << std::endl;

        std::vector<double> high, low, close;
        generate_test_data(high, low, close, 150, 0.3, 100.0, 2.5);

        std::vector<int> windows = {10, 20, 30, 40, 50, 60, 70, 80};
        std::vector<Channel> channels = ChannelDetector::detect_multi_window(
            high, low, close,
            windows,
            2.0,  // std_multiplier
            0.10, // touch_threshold
            1     // min_cycles
        );

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Window | Valid | Bounces | R²    | Quality" << std::endl;
        std::cout << "  -------+-------+---------+-------+--------" << std::endl;

        for (size_t i = 0; i < channels.size(); ++i) {
            const auto& ch = channels[i];
            std::cout << "  " << std::setw(6) << windows[i]
                     << " | " << (ch.valid ? " YES " : " NO  ")
                     << " | " << std::setw(7) << ch.bounce_count
                     << " | " << std::setw(5) << ch.r_squared
                     << " | " << std::setw(6) << ch.quality_score
                     << std::endl;
        }

        std::cout << std::endl;
    }

    // Test 3: Edge cases
    {
        std::cout << "Test 3: Edge cases" << std::endl;

        // Empty data
        {
            std::vector<double> empty;
            Channel ch = ChannelDetector::detect_channel(empty, empty, empty);
            std::cout << "  Empty data - Valid: " << (ch.valid ? "YES" : "NO") << std::endl;
        }

        // Insufficient data
        {
            std::vector<double> small(10, 100.0);
            Channel ch = ChannelDetector::detect_channel(small, small, small, 50);
            std::cout << "  Insufficient data (10 bars, window=50) - Valid: "
                     << (ch.valid ? "YES" : "NO") << std::endl;
        }

        // No bounces (flat channel)
        {
            std::vector<double> flat(100, 100.0);
            Channel ch = ChannelDetector::detect_channel(flat, flat, flat, 50);
            std::cout << "  Flat prices (no bounces) - Valid: "
                     << (ch.valid ? "YES" : "NO")
                     << ", Bounces: " << ch.bounce_count << std::endl;
        }

        std::cout << std::endl;
    }

    // Test 4: Performance test
    {
        std::cout << "Test 4: Performance test" << std::endl;

        std::vector<double> high, low, close;
        generate_test_data(high, low, close, 10000, 0.2, 100.0, 3.0);

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<int> windows = {10, 20, 30, 40, 50, 60, 70, 80};
        std::vector<Channel> channels = ChannelDetector::detect_multi_window(
            high, low, close, windows
        );

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "  Detected 8 windows on 10,000 bars in "
                 << duration.count() << " μs" << std::endl;
        std::cout << "  Average per window: "
                 << duration.count() / 8 << " μs" << std::endl;

        std::cout << std::endl;
    }

    std::cout << "=== All Tests Complete ===" << std::endl;

    return 0;
}
