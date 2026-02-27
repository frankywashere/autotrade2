#include "channel_detector.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>

using namespace v15;

/**
 * Comprehensive edge case testing for channel detection
 * Tests all safety checks and error conditions
 */

void test_empty_arrays() {
    std::cout << "Test: Empty arrays" << std::endl;
    std::vector<double> empty;
    Channel ch = ChannelDetector::detect_channel(empty, empty, empty, 50);
    std::cout << "  Valid: " << (ch.valid ? "FAIL (should be invalid)" : "PASS") << std::endl;
    std::cout << std::endl;
}

void test_insufficient_data() {
    std::cout << "Test: Insufficient data (10 bars, window=50)" << std::endl;
    std::vector<double> data(10, 100.0);
    Channel ch = ChannelDetector::detect_channel(data, data, data, 50);
    std::cout << "  Valid: " << (ch.valid ? "FAIL (should be invalid)" : "PASS") << std::endl;
    std::cout << std::endl;
}

void test_exact_window_size() {
    std::cout << "Test: Exact window size (10 bars, window=10)" << std::endl;
    std::vector<double> high(11, 105.0);
    std::vector<double> low(11, 95.0);
    std::vector<double> close(11, 100.0);

    // Add some variation
    for (size_t i = 0; i < close.size(); ++i) {
        close[i] = 100.0 + i * 0.5;
        high[i] = close[i] + 2.0;
        low[i] = close[i] - 2.0;
    }

    Channel ch = ChannelDetector::detect_channel(high, low, close, 10);
    std::cout << "  Valid: " << (ch.valid ? "YES" : "NO") << std::endl;
    std::cout << "  Bounces: " << ch.bounce_count << std::endl;
    std::cout << "  R²: " << std::fixed << std::setprecision(4) << ch.r_squared << std::endl;
    std::cout << "  Std Dev: " << ch.std_dev << std::endl;
    std::cout << std::endl;
}

void test_flat_prices() {
    std::cout << "Test: Completely flat prices (no variance)" << std::endl;
    std::vector<double> flat(100, 100.0);
    Channel ch = ChannelDetector::detect_channel(flat, flat, flat, 50);
    std::cout << "  Valid: " << (ch.valid ? "FAIL (should be invalid)" : "PASS") << std::endl;
    std::cout << "  Std Dev: " << ch.std_dev << " (should be 0)" << std::endl;
    std::cout << "  R²: " << ch.r_squared << std::endl;
    std::cout << std::endl;
}

void test_very_small_variance() {
    std::cout << "Test: Very small variance (essentially flat)" << std::endl;
    std::vector<double> close(100);
    std::vector<double> high(100);
    std::vector<double> low(100);

    for (size_t i = 0; i < close.size(); ++i) {
        // Tiny variation: 0.0001% of price
        close[i] = 1000.0 + (i % 2) * 0.001;
        high[i] = close[i] + 0.0005;
        low[i] = close[i] - 0.0005;
    }

    Channel ch = ChannelDetector::detect_channel(high, low, close, 50);
    std::cout << "  Valid: " << (ch.valid ? "YES" : "NO") << std::endl;
    std::cout << "  Std Dev: " << std::scientific << ch.std_dev << std::endl;
    std::cout << "  Width %: " << std::fixed << ch.width_pct << std::endl;
    std::cout << std::endl;
}

void test_zero_prices() {
    std::cout << "Test: Zero prices (invalid data)" << std::endl;
    std::vector<double> zeros(100, 0.0);
    Channel ch = ChannelDetector::detect_channel(zeros, zeros, zeros, 50);
    std::cout << "  Valid: " << (ch.valid ? "FAIL (should be invalid)" : "PASS") << std::endl;
    std::cout << std::endl;
}

void test_negative_prices() {
    std::cout << "Test: Negative prices (invalid data)" << std::endl;
    std::vector<double> negative(100, -100.0);
    Channel ch = ChannelDetector::detect_channel(negative, negative, negative, 50);
    std::cout << "  Valid: " << (ch.valid ? "FAIL (should be invalid)" : "PASS") << std::endl;
    std::cout << std::endl;
}

void test_nan_prices() {
    std::cout << "Test: NaN prices (invalid data)" << std::endl;
    std::vector<double> close(100, 100.0);
    std::vector<double> high(100, 105.0);
    std::vector<double> low(100, 95.0);

    // Inject NaN at position 50
    close[50] = std::numeric_limits<double>::quiet_NaN();

    Channel ch = ChannelDetector::detect_channel(high, low, close, 50);
    std::cout << "  Valid: " << (ch.valid ? "FAIL (should be invalid)" : "PASS") << std::endl;
    std::cout << std::endl;
}

void test_infinite_prices() {
    std::cout << "Test: Infinite prices (invalid data)" << std::endl;
    std::vector<double> close(100, 100.0);
    std::vector<double> high(100, 105.0);
    std::vector<double> low(100, 95.0);

    // Inject infinity at position 50
    close[50] = std::numeric_limits<double>::infinity();

    Channel ch = ChannelDetector::detect_channel(high, low, close, 50);
    std::cout << "  Valid: " << (ch.valid ? "FAIL (should be invalid)" : "PASS") << std::endl;
    std::cout << std::endl;
}

void test_very_high_rsquared() {
    std::cout << "Test: Very high R² (perfect linear trend)" << std::endl;
    std::vector<double> close(100);
    std::vector<double> high(100);
    std::vector<double> low(100);

    for (size_t i = 0; i < close.size(); ++i) {
        // Perfect linear trend
        close[i] = 100.0 + i * 0.5;
        // Add some width for bounces
        high[i] = close[i] + 2.0;
        low[i] = close[i] - 2.0;
    }

    Channel ch = ChannelDetector::detect_channel(high, low, close, 50);
    std::cout << "  Valid: " << (ch.valid ? "YES" : "NO") << std::endl;
    std::cout << "  R²: " << std::fixed << std::setprecision(6) << ch.r_squared
              << " (should be ~1.0)" << std::endl;
    std::cout << "  Slope: " << ch.slope << std::endl;
    std::cout << std::endl;
}

void test_very_low_rsquared() {
    std::cout << "Test: Very low R² (random walk)" << std::endl;
    std::vector<double> close(100);
    std::vector<double> high(100);
    std::vector<double> low(100);

    // Create noisy data with no trend
    double price = 100.0;
    for (size_t i = 0; i < close.size(); ++i) {
        price += (i % 3 == 0 ? 1.0 : -1.0) * (i % 5);
        close[i] = price;
        high[i] = price + 2.0;
        low[i] = price - 2.0;
    }

    Channel ch = ChannelDetector::detect_channel(high, low, close, 50);
    std::cout << "  Valid: " << (ch.valid ? "YES" : "NO") << std::endl;
    std::cout << "  R²: " << std::fixed << std::setprecision(6) << ch.r_squared
              << " (should be low)" << std::endl;
    std::cout << std::endl;
}

void test_zero_bounces() {
    std::cout << "Test: Zero bounces (price in middle of channel)" << std::endl;
    std::vector<double> close(100);
    std::vector<double> high(100);
    std::vector<double> low(100);

    for (size_t i = 0; i < close.size(); ++i) {
        close[i] = 100.0 + i * 0.1;
        // Tight high/low that don't touch bounds
        high[i] = close[i] + 0.05;
        low[i] = close[i] - 0.05;
    }

    Channel ch = ChannelDetector::detect_channel(high, low, close, 50, 2.0, 0.10, 1);
    std::cout << "  Valid: " << (ch.valid ? "FAIL (should be invalid)" : "PASS") << std::endl;
    std::cout << "  Bounces: " << ch.bounce_count << " (should be 0)" << std::endl;
    std::cout << std::endl;
}

void test_inconsistent_array_sizes() {
    std::cout << "Test: Inconsistent array sizes" << std::endl;
    std::vector<double> high(100, 105.0);
    std::vector<double> low(50, 95.0);   // Different size!
    std::vector<double> close(100, 100.0);

    Channel ch = ChannelDetector::detect_channel(high, low, close, 50);
    std::cout << "  Valid: " << (ch.valid ? "FAIL (should be invalid)" : "PASS") << std::endl;
    std::cout << std::endl;
}

void test_negative_window() {
    std::cout << "Test: Negative window size" << std::endl;
    std::vector<double> data(100, 100.0);
    Channel ch = ChannelDetector::detect_channel(data, data, data, -10);
    std::cout << "  Valid: " << (ch.valid ? "FAIL (should be invalid)" : "PASS") << std::endl;
    std::cout << std::endl;
}

void test_zero_window() {
    std::cout << "Test: Zero window size" << std::endl;
    std::vector<double> data(100, 100.0);
    Channel ch = ChannelDetector::detect_channel(data, data, data, 0);
    std::cout << "  Valid: " << (ch.valid ? "FAIL (should be invalid)" : "PASS") << std::endl;
    std::cout << std::endl;
}

void test_position_at_edge_cases() {
    std::cout << "Test: position_at() edge cases" << std::endl;

    // Valid channel
    std::vector<double> close(100);
    std::vector<double> high(100);
    std::vector<double> low(100);

    for (size_t i = 0; i < close.size(); ++i) {
        close[i] = 100.0 + i * 0.5;
        high[i] = close[i] + 3.0;
        low[i] = close[i] - 3.0;
    }

    Channel ch = ChannelDetector::detect_channel(high, low, close, 50);

    if (ch.valid) {
        // Test negative indexing
        double pos = ch.position_at(-1);
        std::cout << "  position_at(-1): " << std::fixed << std::setprecision(3) << pos << std::endl;

        // Test out of bounds
        pos = ch.position_at(9999);
        std::cout << "  position_at(9999): " << pos << " (should be 0.5)" << std::endl;

        // Test first element
        pos = ch.position_at(0);
        std::cout << "  position_at(0): " << pos << std::endl;
    } else {
        std::cout << "  SKIP: Channel is invalid" << std::endl;
    }
    std::cout << std::endl;
}

void test_multi_window_with_invalid_data() {
    std::cout << "Test: Multi-window detection with invalid data" << std::endl;
    std::vector<double> data(100, 100.0);  // Flat data

    std::vector<int> windows = {10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<Channel> channels = ChannelDetector::detect_multi_window(
        data, data, data, windows
    );

    int valid_count = 0;
    for (const auto& ch : channels) {
        if (ch.valid) valid_count++;
    }

    std::cout << "  Valid channels: " << valid_count << "/" << channels.size()
              << " (should be 0)" << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "===========================================================" << std::endl;
    std::cout << "    CHANNEL DETECTOR EDGE CASE TESTS" << std::endl;
    std::cout << "===========================================================" << std::endl;
    std::cout << std::endl;

    // Input validation tests
    test_empty_arrays();
    test_insufficient_data();
    test_inconsistent_array_sizes();
    test_negative_window();
    test_zero_window();

    // Data quality tests
    test_flat_prices();
    test_very_small_variance();
    test_zero_prices();
    test_negative_prices();
    test_nan_prices();
    test_infinite_prices();

    // Statistical edge cases
    test_very_high_rsquared();
    test_very_low_rsquared();

    // Boundary conditions
    test_exact_window_size();
    test_zero_bounces();

    // Method edge cases
    test_position_at_edge_cases();

    // Multi-window tests
    test_multi_window_with_invalid_data();

    std::cout << "===========================================================" << std::endl;
    std::cout << "    ALL EDGE CASE TESTS COMPLETE" << std::endl;
    std::cout << "===========================================================" << std::endl;

    return 0;
}
