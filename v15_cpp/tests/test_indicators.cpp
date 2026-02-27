#include "../include/indicators.hpp"
#include <iostream>
#include <iomanip>
#include <random>

using namespace v15;

// Generate synthetic OHLCV data for testing
void generate_synthetic_data(
    int n,
    std::vector<double>& open,
    std::vector<double>& high,
    std::vector<double>& low,
    std::vector<double>& close,
    std::vector<double>& volume
) {
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<> price_dist(100.0, 2.0);
    std::normal_distribution<> volume_dist(1000000.0, 100000.0);

    open.resize(n);
    high.resize(n);
    low.resize(n);
    close.resize(n);
    volume.resize(n);

    double base_price = 100.0;
    for (int i = 0; i < n; ++i) {
        base_price += price_dist(gen) * 0.01;

        open[i] = base_price;
        close[i] = base_price + price_dist(gen) * 0.1;

        high[i] = std::max(open[i], close[i]) + std::abs(price_dist(gen)) * 0.05;
        low[i] = std::min(open[i], close[i]) - std::abs(price_dist(gen)) * 0.05;

        volume[i] = std::abs(volume_dist(gen));
    }
}

int main() {
    std::cout << "=== Technical Indicators Test ===" << std::endl;
    std::cout << std::endl;

    // Test 1: Feature count
    auto feature_names = TechnicalIndicators::get_feature_names();
    std::cout << "Feature Count: " << feature_names.size() << std::endl;
    std::cout << "Expected: 59" << std::endl;
    std::cout << std::endl;

    // Test 2: Feature names by category
    std::cout << "Feature Names by Category:" << std::endl;
    std::cout << "MACD (5): " << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "  - " << feature_names[i] << std::endl;
    }
    std::cout << "Bollinger Bands (8): " << std::endl;
    for (int i = 5; i < 13; ++i) {
        std::cout << "  - " << feature_names[i] << std::endl;
    }
    std::cout << "Keltner (5): " << std::endl;
    for (int i = 13; i < 18; ++i) {
        std::cout << "  - " << feature_names[i] << std::endl;
    }
    std::cout << "ADX (4): " << std::endl;
    for (int i = 18; i < 22; ++i) {
        std::cout << "  - " << feature_names[i] << std::endl;
    }
    std::cout << "Ichimoku (6): " << std::endl;
    for (int i = 22; i < 28; ++i) {
        std::cout << "  - " << feature_names[i] << std::endl;
    }
    std::cout << "Volume (8): " << std::endl;
    for (int i = 28; i < 36; ++i) {
        std::cout << "  - " << feature_names[i] << std::endl;
    }
    std::cout << "Oscillators (6): " << std::endl;
    for (int i = 36; i < 42; ++i) {
        std::cout << "  - " << feature_names[i] << std::endl;
    }
    std::cout << "Pivots (3): " << std::endl;
    for (int i = 42; i < 45; ++i) {
        std::cout << "  - " << feature_names[i] << std::endl;
    }
    std::cout << "Fibonacci (3): " << std::endl;
    for (int i = 45; i < 48; ++i) {
        std::cout << "  - " << feature_names[i] << std::endl;
    }
    std::cout << "Candlestick (7): " << std::endl;
    for (int i = 48; i < 55; ++i) {
        std::cout << "  - " << feature_names[i] << std::endl;
    }
    std::cout << "Additional (3): " << std::endl;
    for (int i = 55; i < 58; ++i) {
        std::cout << "  - " << feature_names[i] << std::endl;
    }
    std::cout << std::endl;

    // Test 3: Extract features from synthetic data
    std::cout << "=== Feature Extraction Test ===" << std::endl;
    std::vector<double> open, high, low, close, volume;
    generate_synthetic_data(200, open, high, low, close, volume);

    auto features = TechnicalIndicators::extract_features(open, high, low, close, volume);

    std::cout << "Extracted " << features.size() << " features" << std::endl;
    std::cout << std::endl;

    // Display sample features
    std::cout << "Sample Feature Values:" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    std::vector<std::string> sample_features = {
        "macd_line", "macd_signal", "macd_histogram",
        "bb_upper", "bb_middle", "bb_lower",
        "adx", "plus_di", "minus_di",
        "rsi", "obv", "mfi",
        "aroon_up", "aroon_down",
        "pivot", "r1", "s1"
    };

    for (const auto& name : sample_features) {
        auto it = features.find(name);
        if (it != features.end()) {
            std::cout << "  " << std::setw(20) << std::left << name << ": "
                      << std::setw(12) << std::right << it->second << std::endl;
        }
    }
    std::cout << std::endl;

    // Test 4: Edge case - insufficient data
    std::cout << "=== Edge Case Test (Insufficient Data) ===" << std::endl;
    std::vector<double> short_data = {100.0};
    auto edge_features = TechnicalIndicators::extract_features(
        short_data, short_data, short_data, short_data
    );
    std::cout << "Features from 1 data point: " << edge_features.size() << std::endl;
    std::cout << "All values should be 0.0 or default" << std::endl;
    std::cout << std::endl;

    // Test 5: Verify no NaN/inf values
    std::cout << "=== NaN/Inf Validation ===" << std::endl;
    bool has_invalid = false;
    for (const auto& [name, value] : features) {
        if (!std::isfinite(value)) {
            std::cout << "ERROR: Invalid value for " << name << ": " << value << std::endl;
            has_invalid = true;
        }
    }
    if (!has_invalid) {
        std::cout << "✓ All feature values are valid (no NaN/inf)" << std::endl;
    }
    std::cout << std::endl;

    std::cout << "=== Test Complete ===" << std::endl;
    return 0;
}
