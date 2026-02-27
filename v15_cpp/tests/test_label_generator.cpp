#include "label_generator.hpp"
#include "channel.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

using namespace v15;

// Test helper to create a simple channel
Channel create_test_channel(int window_size, double slope, double intercept, double std_dev) {
    Channel ch;
    ch.timeframe = Timeframe::MIN_15;
    ch.window_size = window_size;
    ch.slope = slope;
    ch.intercept = intercept;
    ch.std_dev = std_dev;
    ch.r_squared = 0.8;
    ch.direction = ChannelDirection::BULL;
    ch.bounce_count = 3;
    ch.start_idx = 0;
    ch.end_idx = window_size - 1;
    ch.start_timestamp_ms = 0;
    ch.end_timestamp_ms = window_size * 60000;
    return ch;
}

// Test 1: No breaks detected (consolidation)
void test_no_break_detected() {
    std::cout << "Test 1: No breaks detected (consolidation)...\n";

    LabelGenerator::Config config;
    config.min_break_magnitude = 0.5;
    config.return_threshold_bars = 5;
    LabelGenerator gen(config);

    // Create channel: slope=0.1, intercept=100, std_dev=2.0
    Channel channel = create_test_channel(20, 0.1, 100.0, 2.0);

    // Create forward data that stays inside channel bounds
    int n_forward = 50;
    std::vector<double> forward_high(n_forward);
    std::vector<double> forward_low(n_forward);
    std::vector<double> forward_close(n_forward);
    std::vector<double> full_close(100, 100.0);  // Flat prices for RSI

    for (int i = 0; i < n_forward; ++i) {
        // Project center at this bar
        double x = channel.window_size + i;
        double center = channel.slope * x + channel.intercept;

        // Stay well inside bounds (< 0.5 std_dev)
        forward_close[i] = center;
        forward_high[i] = center + 0.3 * channel.std_dev;
        forward_low[i] = center - 0.3 * channel.std_dev;

        if (i + channel.window_size < 100) {
            full_close[i + channel.window_size] = center;
        }
    }

    ChannelLabels labels = gen.generate_labels_forward_scan(
        channel,
        channel.window_size - 1,  // channel_end_idx
        forward_high.data(),
        forward_low.data(),
        forward_close.data(),
        n_forward,
        50,  // max_scan
        -1,  // next_channel_direction
        full_close.data(),
        100
    );

    // Validate no-break scenario
    assert(labels.break_scan_valid == true);
    assert(labels.duration_valid == true);
    assert(labels.direction_valid == false);  // Can't determine direction without break
    assert(labels.break_magnitude == 0.0);
    assert(labels.permanent_break == false);
    assert(labels.permanent_break_direction == -1);

    // RSI should be set to defaults
    assert(labels.rsi_at_first_break == 50.0);
    assert(labels.rsi_at_permanent_break == 50.0);
    assert(labels.rsi_overbought_at_break == false);
    assert(labels.rsi_oversold_at_break == false);

    std::cout << "  ✓ No break scenario handled correctly\n";
}

// Test 2: Immediate break (bar 0)
void test_immediate_break() {
    std::cout << "Test 2: Immediate break at bar 0...\n";

    LabelGenerator::Config config;
    LabelGenerator gen(config);

    Channel channel = create_test_channel(20, 0.1, 100.0, 2.0);

    int n_forward = 50;
    std::vector<double> forward_high(n_forward);
    std::vector<double> forward_low(n_forward);
    std::vector<double> forward_close(n_forward);
    std::vector<double> full_close(100, 100.0);

    // First bar breaks upward strongly
    double x0 = channel.window_size;
    double center0 = channel.slope * x0 + channel.intercept;
    double upper0 = center0 + 2.0 * channel.std_dev;

    forward_close[0] = upper0 + 1.5 * channel.std_dev;  // magnitude = 1.5
    forward_high[0] = forward_close[0] + 1.0;
    forward_low[0] = center0;
    full_close[channel.window_size] = forward_close[0];

    // Rest stay outside
    for (int i = 1; i < n_forward; ++i) {
        forward_close[i] = forward_close[0];
        forward_high[i] = forward_close[i] + 1.0;
        forward_low[i] = forward_close[i] - 0.5;
        if (i + channel.window_size < 100) {
            full_close[i + channel.window_size] = forward_close[i];
        }
    }

    ChannelLabels labels = gen.generate_labels_forward_scan(
        channel,
        channel.window_size - 1,
        forward_high.data(),
        forward_low.data(),
        forward_close.data(),
        n_forward,
        50,
        -1,
        full_close.data(),
        100
    );

    // Validate immediate break
    assert(labels.break_scan_valid == true);
    assert(labels.bars_to_first_break == 0);
    assert(labels.break_direction == 1);  // UP
    assert(labels.break_magnitude >= 1.0);
    // permanent_break requires staying out 5+ bars, which happened here
    assert(labels.permanent_break_direction == 1);  // Permanent break detected

    std::cout << "  ✓ Immediate break at bar 0 handled correctly\n";
    std::cout << "    bars_to_first_break: " << labels.bars_to_first_break << "\n";
    std::cout << "    break_magnitude: " << labels.break_magnitude << "\n";
}

// Test 3: Break at end of scan window
void test_break_at_end() {
    std::cout << "Test 3: Break at end of scan window...\n";

    LabelGenerator::Config config;
    LabelGenerator gen(config);

    Channel channel = create_test_channel(20, 0.1, 100.0, 2.0);

    int n_forward = 50;
    int max_scan = 30;
    std::vector<double> forward_high(n_forward);
    std::vector<double> forward_low(n_forward);
    std::vector<double> forward_close(n_forward);
    std::vector<double> full_close(100, 100.0);

    // Stay inside until bar 29 (last bar of scan)
    for (int i = 0; i < n_forward; ++i) {
        double x = channel.window_size + i;
        double center = channel.slope * x + channel.intercept;

        if (i < 29) {
            // Inside channel
            forward_close[i] = center;
            forward_high[i] = center + 0.3 * channel.std_dev;
            forward_low[i] = center - 0.3 * channel.std_dev;
        } else {
            // Break downward
            double lower = center - 2.0 * channel.std_dev;
            forward_close[i] = lower - 1.0 * channel.std_dev;
            forward_high[i] = center;
            forward_low[i] = forward_close[i] - 0.5;
        }

        if (i + channel.window_size < 100) {
            full_close[i + channel.window_size] = forward_close[i];
        }
    }

    ChannelLabels labels = gen.generate_labels_forward_scan(
        channel,
        channel.window_size - 1,
        forward_high.data(),
        forward_low.data(),
        forward_close.data(),
        n_forward,
        max_scan,
        -1,
        full_close.data(),
        100
    );

    // Validate break at scan boundary
    assert(labels.break_scan_valid == true);
    assert(labels.bars_to_first_break == 29);
    assert(labels.break_direction == 0);  // DOWN

    std::cout << "  ✓ Break at end of scan window handled correctly\n";
    std::cout << "    bars_to_first_break: " << labels.bars_to_first_break << "\n";
}

// Test 4: Invalid RSI inputs
void test_rsi_validation() {
    std::cout << "Test 4: RSI validation...\n";

    LabelGenerator::Config config;
    config.rsi_period = 14;
    LabelGenerator gen(config);

    Channel channel = create_test_channel(20, 0.1, 100.0, 2.0);

    int n_forward = 10;
    std::vector<double> forward_high(n_forward, 102.0);
    std::vector<double> forward_low(n_forward, 98.0);
    std::vector<double> forward_close(n_forward, 100.0);

    // Test with no close prices
    ChannelLabels labels1 = gen.generate_labels_forward_scan(
        channel,
        channel.window_size - 1,
        forward_high.data(),
        forward_low.data(),
        forward_close.data(),
        n_forward,
        10,
        -1,
        nullptr,  // No close prices
        0
    );

    // Should use defaults
    assert(labels1.rsi_at_channel_end == 50.0);
    assert(labels1.rsi_at_first_break == 50.0);

    // Test with insufficient close prices (< RSI period)
    std::vector<double> short_close(10, 100.0);
    ChannelLabels labels2 = gen.generate_labels_forward_scan(
        channel,
        5,
        forward_high.data(),
        forward_low.data(),
        forward_close.data(),
        n_forward,
        10,
        -1,
        short_close.data(),
        10  // Less than RSI period + 1
    );

    // Should use defaults
    assert(labels2.rsi_at_channel_end == 50.0);

    std::cout << "  ✓ RSI validation handled correctly\n";
}

// Test 5: No next channels available
void test_no_next_channels() {
    std::cout << "Test 5: No next channels available...\n";

    LabelGenerator::Config config;
    LabelGenerator gen(config);

    Channel channel = create_test_channel(20, 0.1, 100.0, 2.0);

    ChannelLabels labels;
    labels.timeframe = Timeframe::MIN_15;

    // Call compute_next_channel_labels with null pointer
    gen.compute_next_channel_labels(nullptr, 0, 0, labels);

    // Should set defaults
    assert(labels.best_next_channel_direction == -1);
    assert(labels.best_next_channel_bars_away == -1);
    assert(labels.shortest_next_channel_direction == -1);

    std::cout << "  ✓ No next channels handled correctly\n";
}

// Test 6: Array bounds validation
void test_array_bounds() {
    std::cout << "Test 6: Array bounds validation...\n";

    LabelGenerator::Config config;
    LabelGenerator gen(config);

    Channel channel = create_test_channel(20, 0.1, 100.0, 2.0);

    int n_forward = 10;
    std::vector<double> forward_high(n_forward, 102.0);
    std::vector<double> forward_low(n_forward, 98.0);
    std::vector<double> forward_close(n_forward, 100.0);

    // Test with max_scan > n_forward (should clamp)
    ChannelLabels labels1 = gen.generate_labels_forward_scan(
        channel,
        channel.window_size - 1,
        forward_high.data(),
        forward_low.data(),
        forward_close.data(),
        n_forward,
        1000,  // Way larger than n_forward
        -1
    );

    assert(labels1.break_scan_valid == true);

    // Test with null pointers
    ChannelLabels labels2 = gen.generate_labels_forward_scan(
        channel,
        channel.window_size - 1,
        nullptr,  // null
        nullptr,
        nullptr,
        n_forward,
        10,
        -1
    );

    assert(labels2.break_scan_valid == false);

    // Test with zero n_forward
    ChannelLabels labels3 = gen.generate_labels_forward_scan(
        channel,
        channel.window_size - 1,
        forward_high.data(),
        forward_low.data(),
        forward_close.data(),
        0,  // No forward data
        10,
        -1
    );

    assert(labels3.break_scan_valid == false);

    std::cout << "  ✓ Array bounds validation passed\n";
}

// Test 7: RSI range validation
void test_rsi_range() {
    std::cout << "Test 7: RSI stays in [0, 100] range...\n";

    LabelGenerator::Config config;
    config.rsi_period = 14;
    LabelGenerator gen(config);

    Channel channel = create_test_channel(20, 0.1, 100.0, 2.0);

    // Create extreme price movements
    int n_forward = 30;
    std::vector<double> forward_high(n_forward);
    std::vector<double> forward_low(n_forward);
    std::vector<double> forward_close(n_forward);
    std::vector<double> full_close(100);

    // Start at 100
    for (int i = 0; i < 50; ++i) {
        full_close[i] = 100.0;
    }

    // Extreme upward movement (should push RSI toward 100)
    for (int i = 0; i < n_forward; ++i) {
        double price = 100.0 + i * 10.0;  // Rapid increase
        forward_close[i] = price;
        forward_high[i] = price + 5.0;
        forward_low[i] = price - 2.0;
        if (50 + i < 100) {
            full_close[50 + i] = price;
        }
    }

    // Force a break at bar 5
    double x5 = channel.window_size + 5;
    double center5 = channel.slope * x5 + channel.intercept;
    double upper5 = center5 + 2.0 * channel.std_dev;
    forward_close[5] = upper5 + 2.0 * channel.std_dev;

    ChannelLabels labels = gen.generate_labels_forward_scan(
        channel,
        49,  // channel_end_idx in full_close
        forward_high.data(),
        forward_low.data(),
        forward_close.data(),
        n_forward,
        30,
        -1,
        full_close.data(),
        100
    );

    // RSI should be clamped to [0, 100]
    assert(labels.rsi_at_first_break >= 0.0 && labels.rsi_at_first_break <= 100.0);
    assert(labels.rsi_at_permanent_break >= 0.0 && labels.rsi_at_permanent_break <= 100.0);
    assert(labels.rsi_at_channel_end >= 0.0 && labels.rsi_at_channel_end <= 100.0);

    std::cout << "  ✓ RSI clamped to valid range [0, 100]\n";
    std::cout << "    RSI at first break: " << labels.rsi_at_first_break << "\n";
    std::cout << "    RSI at channel end: " << labels.rsi_at_channel_end << "\n";
}

// Test 8: False break that returns
void test_false_break() {
    std::cout << "Test 8: False break that returns to channel...\n";

    LabelGenerator::Config config;
    config.min_break_magnitude = 0.5;
    config.return_threshold_bars = 5;
    LabelGenerator gen(config);

    Channel channel = create_test_channel(20, 0.1, 100.0, 2.0);

    int n_forward = 50;
    std::vector<double> forward_high(n_forward);
    std::vector<double> forward_low(n_forward);
    std::vector<double> forward_close(n_forward);
    std::vector<double> full_close(100, 100.0);

    for (int i = 0; i < n_forward; ++i) {
        double x = channel.window_size + i;
        double center = channel.slope * x + channel.intercept;
        double upper = center + 2.0 * channel.std_dev;

        if (i >= 10 && i < 14) {
            // Break upward for 4 bars
            forward_close[i] = upper + 0.8 * channel.std_dev;
        } else {
            // Inside channel
            forward_close[i] = center;
        }

        forward_high[i] = forward_close[i] + 1.0;
        forward_low[i] = forward_close[i] - 1.0;

        if (i + channel.window_size < 100) {
            full_close[i + channel.window_size] = forward_close[i];
        }
    }

    ChannelLabels labels = gen.generate_labels_forward_scan(
        channel,
        channel.window_size - 1,
        forward_high.data(),
        forward_low.data(),
        forward_close.data(),
        n_forward,
        50,
        -1,
        full_close.data(),
        100
    );

    // Should detect false break
    assert(labels.break_scan_valid == true);
    assert(labels.bars_to_first_break >= 10);
    assert(labels.returned_to_channel == true);
    assert(labels.permanent_break == false);

    std::cout << "  ✓ False break detected and handled correctly\n";
    std::cout << "    bars_to_first_break: " << labels.bars_to_first_break << "\n";
    std::cout << "    returned_to_channel: " << labels.returned_to_channel << "\n";
}

int main() {
    std::cout << "=== Label Generator Edge Case Tests ===\n\n";

    try {
        test_no_break_detected();
        test_immediate_break();
        test_break_at_end();
        test_rsi_validation();
        test_no_next_channels();
        test_array_bounds();
        test_rsi_range();
        test_false_break();

        std::cout << "\n=== All tests passed! ===\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n!!! Test failed with exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "\n!!! Test failed with unknown exception\n";
        return 1;
    }
}
