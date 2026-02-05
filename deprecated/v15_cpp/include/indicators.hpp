#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace v15 {

/**
 * TechnicalIndicators - High-performance C++ implementation of 59 technical indicators
 *
 * Matches Python implementation exactly for cross-validation.
 * Uses vectorized operations and optional SIMD intrinsics for performance.
 *
 * Categories:
 * - MACD (5 features)
 * - Bollinger Bands (8 features)
 * - Keltner Channels (5 features)
 * - ADX (4 features)
 * - Ichimoku (6 features)
 * - Volume Indicators (8 features)
 * - Oscillators (6 features)
 * - Pivot Points (3 features)
 * - Fibonacci (3 features)
 * - Candlestick Patterns (7 features)
 * - Additional (3 features)
 */
class TechnicalIndicators {
public:
    /**
     * Extract all 59 technical indicator features from OHLCV data
     *
     * @param open Open prices
     * @param high High prices
     * @param low Low prices
     * @param close Close prices
     * @param volume Volume (optional, can be empty for VIX)
     * @return Map of feature names to values
     */
    static std::unordered_map<std::string, double> extract_features(
        const std::vector<double>& open,
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close,
        const std::vector<double>& volume = {}
    );

    /**
     * Get list of all 59 feature names
     */
    static std::vector<std::string> get_feature_names();

    /**
     * Get total feature count (59)
     */
    static int get_feature_count() { return 59; }

    // Helper functions - Moving Averages (public for use by FeatureExtractor)
    static std::vector<double> ema(const std::vector<double>& values, int period);
    static std::vector<double> sma(const std::vector<double>& values, int period);

    // Helper functions - Momentum (public for use by FeatureExtractor)
    static std::vector<double> rsi(const std::vector<double>& values, int period = 14);

    // Helper functions - Volatility (public for use by FeatureExtractor)
    static std::vector<double> true_range(
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close
    );

    static std::vector<double> atr(
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close,
        int period = 14
    );

    // =============================================================================
    // Optimized _last variants - Return only the last value without full array allocation
    // =============================================================================

    /**
     * Compute only the last SMA value (optimized - no full array allocation)
     * @param values Input price series
     * @param period SMA period
     * @return The last SMA value, or 0.0 if insufficient data
     */
    static double sma_last(const std::vector<double>& values, int period);

    /**
     * Compute only the last EMA value (optimized - no full array allocation)
     * @param values Input price series
     * @param period EMA period
     * @return The last EMA value, or 0.0 if insufficient data
     */
    static double ema_last(const std::vector<double>& values, int period);

    /**
     * Compute only the last RSI value (optimized - no full array allocation)
     * @param values Input price series
     * @param period RSI period (default 14)
     * @return The last RSI value (0-100), or 50.0 if insufficient data
     */
    static double rsi_last(const std::vector<double>& values, int period = 14);

    /**
     * Compute only the last ATR value (optimized - no full array allocation)
     * @param high High prices
     * @param low Low prices
     * @param close Close prices
     * @param period ATR period (default 14)
     * @return The last ATR value, or 0.0 if insufficient data
     */
    static double atr_last(
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close,
        int period = 14
    );

    // Helper functions - Utilities (public for use by FeatureExtractor)
    static double safe_divide(double numerator, double denominator, double default_val = 0.0);
    static double safe_float(double value, double default_val = 0.0);
    static double get_last_valid(const std::vector<double>& arr, double default_val = 0.0);
    static double scalar_pct_change(double current, double previous, double default_val = 0.0);

private:

    // Category-specific calculation functions
    static std::unordered_map<std::string, double> calculate_macd(
        const std::vector<double>& close
    );

    static std::unordered_map<std::string, double> calculate_bollinger_bands(
        const std::vector<double>& close,
        double current_close
    );

    static std::unordered_map<std::string, double> calculate_keltner(
        const std::vector<double>& close,
        const std::vector<double>& high,
        const std::vector<double>& low,
        double current_close
    );

    static std::unordered_map<std::string, double> calculate_adx(
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close
    );

    static std::unordered_map<std::string, double> calculate_ichimoku(
        const std::vector<double>& high,
        const std::vector<double>& low,
        double current_close
    );

    static std::unordered_map<std::string, double> calculate_volume_indicators(
        const std::vector<double>& open,
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close,
        const std::vector<double>& volume
    );

    static std::unordered_map<std::string, double> calculate_oscillators(
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close
    );

    static std::unordered_map<std::string, double> calculate_pivot_points(
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close
    );

    static std::unordered_map<std::string, double> calculate_fibonacci(
        const std::vector<double>& high,
        const std::vector<double>& low,
        double current_close
    );

    static std::unordered_map<std::string, double> calculate_candlestick_patterns(
        const std::vector<double>& open,
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close
    );

    static std::unordered_map<std::string, double> calculate_additional(
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close
    );
};

} // namespace v15
