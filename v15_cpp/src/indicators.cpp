#include "indicators.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

namespace v15 {

// =============================================================================
// Helper Functions - Utilities
// =============================================================================

double TechnicalIndicators::safe_divide(double numerator, double denominator, double default_val) {
    if (denominator == 0.0 || !std::isfinite(denominator) || !std::isfinite(numerator)) {
        return default_val;
    }
    double result = numerator / denominator;
    return std::isfinite(result) ? result : default_val;
}

double TechnicalIndicators::safe_float(double value, double default_val) {
    return std::isfinite(value) ? value : default_val;
}

double TechnicalIndicators::get_last_valid(const std::vector<double>& arr, double default_val) {
    for (int i = arr.size() - 1; i >= 0; --i) {
        if (std::isfinite(arr[i])) {
            return arr[i];
        }
    }
    return default_val;
}

double TechnicalIndicators::scalar_pct_change(double current, double previous, double default_val) {
    if (previous == 0.0 || !std::isfinite(previous) || !std::isfinite(current)) {
        return default_val;
    }
    double result = ((current - previous) / previous) * 100.0;
    return std::isfinite(result) ? result : default_val;
}

// =============================================================================
// Helper Functions - Moving Averages
// =============================================================================

std::vector<double> TechnicalIndicators::sma(const std::vector<double>& values, int period) {
    size_t n = values.size();
    std::vector<double> result(n, 0.0);

    if (n == 0 || period <= 0) {
        return result;
    }

    double sum = 0.0;
    int count = 0;

    for (size_t i = 0; i < n; ++i) {
        sum += values[i];
        count++;

        if (count > period) {
            sum -= values[i - period];
            count--;
        }

        result[i] = sum / count;
    }

    return result;
}

std::vector<double> TechnicalIndicators::ema(const std::vector<double>& values, int period) {
    size_t n = values.size();
    std::vector<double> result(n, 0.0);

    if (n == 0 || period <= 0) {
        return result;
    }

    double multiplier = 2.0 / (period + 1.0);
    result[0] = values[0];

    for (size_t i = 1; i < n; ++i) {
        result[i] = (values[i] - result[i-1]) * multiplier + result[i-1];
    }

    return result;
}

// =============================================================================
// Helper Functions - Volatility
// =============================================================================

std::vector<double> TechnicalIndicators::true_range(
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close
) {
    size_t n = std::min({high.size(), low.size(), close.size()});
    std::vector<double> result(n, 0.0);

    if (n == 0) {
        return result;
    }

    result[0] = high[0] - low[0];

    for (size_t i = 1; i < n; ++i) {
        double tr1 = high[i] - low[i];
        double tr2 = std::abs(high[i] - close[i-1]);
        double tr3 = std::abs(low[i] - close[i-1]);
        result[i] = std::max({tr1, tr2, tr3});
    }

    return result;
}

std::vector<double> TechnicalIndicators::atr(
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close,
    int period
) {
    std::vector<double> tr = true_range(high, low, close);
    return ema(tr, period);
}

// =============================================================================
// Optimized _last variants - Return only the last value
// =============================================================================

double TechnicalIndicators::sma_last(const std::vector<double>& values, int period) {
    size_t n = values.size();

    if (n == 0 || period <= 0) {
        return 0.0;
    }

    // For the last SMA value, we only need to sum the last 'period' values
    // If we have fewer values than period, use all available values
    size_t start = (n >= static_cast<size_t>(period)) ? (n - period) : 0;
    size_t count = n - start;

    double sum = 0.0;
    for (size_t i = start; i < n; ++i) {
        sum += values[i];
    }

    return sum / count;
}

double TechnicalIndicators::ema_last(const std::vector<double>& values, int period) {
    size_t n = values.size();

    if (n == 0 || period <= 0) {
        return 0.0;
    }

    double multiplier = 2.0 / (period + 1.0);
    double ema_value = values[0];

    // Iterate through all values to compute EMA, but only keep the last value
    for (size_t i = 1; i < n; ++i) {
        ema_value = (values[i] - ema_value) * multiplier + ema_value;
    }

    return ema_value;
}

double TechnicalIndicators::rsi_last(const std::vector<double>& values, int period) {
    size_t n = values.size();

    if (n <= 1 || period <= 0) {
        return 50.0;
    }

    double multiplier = 2.0 / (period + 1.0);

    // Match original behavior: gains[0] and losses[0] are 0, EMA starts with those
    // EMA formula: result[0] = values[0], then result[i] = (values[i] - result[i-1]) * mult + result[i-1]
    // So avg_gain and avg_loss start at 0.0 (matching gains[0] and losses[0])
    double avg_gain = 0.0;
    double avg_loss = 0.0;

    // Process all values starting from index 1 (where gains/losses are computed)
    for (size_t i = 1; i < n; ++i) {
        double delta = values[i] - values[i-1];
        double gain = (delta > 0) ? delta : 0.0;
        double loss = (delta < 0) ? -delta : 0.0;

        // EMA update: new_ema = (value - old_ema) * multiplier + old_ema
        avg_gain = (gain - avg_gain) * multiplier + avg_gain;
        avg_loss = (loss - avg_loss) * multiplier + avg_loss;
    }

    // Calculate final RSI
    double rsi_value;
    if (avg_loss == 0.0) {
        rsi_value = (avg_gain > 0.0) ? 100.0 : 50.0;
    } else {
        double rs = avg_gain / avg_loss;
        rsi_value = 100.0 - (100.0 / (1.0 + rs));
    }

    return std::clamp(rsi_value, 0.0, 100.0);
}

double TechnicalIndicators::atr_last(
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close,
    int period
) {
    size_t n = std::min({high.size(), low.size(), close.size()});

    if (n == 0 || period <= 0) {
        return 0.0;
    }

    double multiplier = 2.0 / (period + 1.0);

    // First TR value
    double atr_value = high[0] - low[0];

    // Compute true range and EMA iteratively (no array allocation)
    for (size_t i = 1; i < n; ++i) {
        double tr1 = high[i] - low[i];
        double tr2 = std::abs(high[i] - close[i-1]);
        double tr3 = std::abs(low[i] - close[i-1]);
        double tr = std::max({tr1, tr2, tr3});

        // EMA update
        atr_value = (tr - atr_value) * multiplier + atr_value;
    }

    return atr_value;
}

// =============================================================================
// Helper Functions - Momentum
// =============================================================================

std::vector<double> TechnicalIndicators::rsi(const std::vector<double>& values, int period) {
    size_t n = values.size();
    std::vector<double> result(n, 50.0);

    if (n <= 1 || period <= 0) {
        return result;
    }

    // Calculate gains and losses
    std::vector<double> gains(n, 0.0);
    std::vector<double> losses(n, 0.0);

    for (size_t i = 1; i < n; ++i) {
        double delta = values[i] - values[i-1];
        if (delta > 0) {
            gains[i] = delta;
        } else {
            losses[i] = -delta;
        }
    }

    // Calculate EMA of gains and losses
    std::vector<double> avg_gains = ema(gains, period);
    std::vector<double> avg_losses = ema(losses, period);

    // Calculate RSI
    for (size_t i = 0; i < n; ++i) {
        if (avg_losses[i] == 0.0) {
            result[i] = (avg_gains[i] > 0.0) ? 100.0 : 50.0;
        } else {
            double rs = avg_gains[i] / avg_losses[i];
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        }
        result[i] = std::clamp(result[i], 0.0, 100.0);
    }

    return result;
}

// =============================================================================
// MACD Indicators (5 features)
// =============================================================================

std::unordered_map<std::string, double> TechnicalIndicators::calculate_macd(
    const std::vector<double>& close
) {
    std::unordered_map<std::string, double> features;
    features.reserve(5);  // MACD has 5 features

    if (close.size() < 26) {
        features["macd_line"] = 0.0;
        features["macd_signal"] = 0.0;
        features["macd_histogram"] = 0.0;
        features["macd_crossover"] = 0.0;
        features["macd_divergence"] = 0.0;
        return features;
    }

    std::vector<double> ema_12 = ema(close, 12);
    std::vector<double> ema_26 = ema(close, 26);

    // MACD line
    std::vector<double> macd_line(close.size());
    for (size_t i = 0; i < close.size(); ++i) {
        macd_line[i] = ema_12[i] - ema_26[i];
    }

    // Signal line
    std::vector<double> macd_signal = ema(macd_line, 9);

    // Histogram
    std::vector<double> macd_histogram(close.size());
    for (size_t i = 0; i < close.size(); ++i) {
        macd_histogram[i] = macd_line[i] - macd_signal[i];
    }

    // Use [-2] to avoid data leakage
    size_t n = close.size();
    features["macd_line"] = (n > 1) ? get_last_valid(std::vector<double>(macd_line.begin(), macd_line.end() - 1), 0.0) : 0.0;
    features["macd_signal"] = (n > 1) ? get_last_valid(std::vector<double>(macd_signal.begin(), macd_signal.end() - 1), 0.0) : 0.0;
    features["macd_histogram"] = (n > 1) ? get_last_valid(std::vector<double>(macd_histogram.begin(), macd_histogram.end() - 1), 0.0) : 0.0;

    // Crossover detection (use [-2] and [-3])
    double crossover = 0.0;
    if (n >= 3) {
        double prev_macd = std::isfinite(macd_line[n-3]) ? macd_line[n-3] : 0.0;
        double prev_signal = std::isfinite(macd_signal[n-3]) ? macd_signal[n-3] : 0.0;
        double curr_macd = std::isfinite(macd_line[n-2]) ? macd_line[n-2] : 0.0;
        double curr_signal = std::isfinite(macd_signal[n-2]) ? macd_signal[n-2] : 0.0;

        if (prev_macd <= prev_signal && curr_macd > curr_signal) {
            crossover = 1.0;
        } else if (prev_macd >= prev_signal && curr_macd < curr_signal) {
            crossover = -1.0;
        }
    }
    features["macd_crossover"] = crossover;

    // Divergence
    double divergence = 0.0;
    if (n >= 11) {
        double price_change = close[n-2] - close[n-11];
        double macd_change = macd_line[n-2] - macd_line[n-11];

        if (price_change > 0 && macd_change < 0) {
            divergence = -1.0;  // Bearish divergence
        } else if (price_change < 0 && macd_change > 0) {
            divergence = 1.0;  // Bullish divergence
        }
    }
    features["macd_divergence"] = divergence;

    return features;
}

// =============================================================================
// Bollinger Bands (8 features)
// =============================================================================

std::unordered_map<std::string, double> TechnicalIndicators::calculate_bollinger_bands(
    const std::vector<double>& close,
    double current_close
) {
    std::unordered_map<std::string, double> features;
    features.reserve(8);  // Bollinger Bands has 8 features

    if (close.size() < 20) {
        features["bb_upper"] = 0.0;
        features["bb_middle"] = 0.0;
        features["bb_lower"] = 0.0;
        features["bb_width"] = 0.0;
        features["bb_pct_b"] = 0.5;
        features["price_vs_bb_upper"] = 0.0;
        features["price_vs_bb_lower"] = 0.0;
        features["bb_squeeze"] = 0.0;
        return features;
    }

    int period = 20;
    double std_dev_mult = 2.0;
    size_t n = close.size();

    std::vector<double> middle = sma(close, period);

    // Calculate rolling standard deviation
    std::vector<double> rolling_std(n, 0.0);
    for (size_t i = period - 1; i < n; ++i) {
        double sum = 0.0;
        double mean = middle[i];
        for (size_t j = i - period + 1; j <= i; ++j) {
            double diff = close[j] - mean;
            sum += diff * diff;
        }
        rolling_std[i] = std::sqrt(sum / period);
    }

    std::vector<double> upper(n);
    std::vector<double> lower(n);
    for (size_t i = 0; i < n; ++i) {
        upper[i] = middle[i] + std_dev_mult * rolling_std[i];
        lower[i] = middle[i] - std_dev_mult * rolling_std[i];
    }

    // Use [-2] to avoid data leakage
    double bb_upper = (n > 1) ? get_last_valid(std::vector<double>(upper.begin(), upper.end() - 1), 0.0) : 0.0;
    double bb_middle = (n > 1) ? get_last_valid(std::vector<double>(middle.begin(), middle.end() - 1), 0.0) : 0.0;
    double bb_lower = (n > 1) ? get_last_valid(std::vector<double>(lower.begin(), lower.end() - 1), 0.0) : 0.0;

    features["bb_upper"] = bb_upper;
    features["bb_middle"] = bb_middle;
    features["bb_lower"] = bb_lower;
    features["bb_width"] = safe_divide(bb_upper - bb_lower, bb_middle, 0.0);

    // %B
    double band_range = bb_upper - bb_lower;
    double bb_pct_b = safe_divide(current_close - bb_lower, band_range, 0.5);
    features["bb_pct_b"] = std::clamp(bb_pct_b, 0.0, 1.0);

    // Price vs bands
    features["price_vs_bb_upper"] = (bb_upper > 0) ? scalar_pct_change(current_close, bb_upper, 0.0) : 0.0;
    features["price_vs_bb_lower"] = (bb_lower > 0) ? scalar_pct_change(current_close, bb_lower, 0.0) : 0.0;

    // Squeeze detection
    double bb_squeeze = 0.0;
    if (n >= 51) {
        double current_std = std::isfinite(rolling_std[n-2]) ? rolling_std[n-2] : 0.0;
        double sum = 0.0;
        int count = 0;
        for (size_t i = n - 51; i < n - 1; ++i) {
            if (std::isfinite(rolling_std[i])) {
                sum += rolling_std[i];
                count++;
            }
        }
        double avg_std = (count > 0) ? sum / count : 0.0;
        if (avg_std > 0 && current_std < avg_std * 0.5) {
            bb_squeeze = 1.0;
        }
    }
    features["bb_squeeze"] = bb_squeeze;

    return features;
}

// =============================================================================
// Keltner Channels (5 features)
// =============================================================================

std::unordered_map<std::string, double> TechnicalIndicators::calculate_keltner(
    const std::vector<double>& close,
    const std::vector<double>& high,
    const std::vector<double>& low,
    double current_close
) {
    std::unordered_map<std::string, double> features;
    features.reserve(5);  // Keltner Channels has 5 features

    if (close.size() < 20) {
        features["keltner_upper"] = 0.0;
        features["keltner_middle"] = 0.0;
        features["keltner_lower"] = 0.0;
        features["keltner_width"] = 0.0;
        features["keltner_position"] = 0.5;
        return features;
    }

    int period = 20;
    double multiplier = 2.0;
    size_t n = close.size();

    std::vector<double> middle = ema(close, period);
    std::vector<double> atr_values = atr(high, low, close, period);

    std::vector<double> upper(n);
    std::vector<double> lower(n);
    for (size_t i = 0; i < n; ++i) {
        upper[i] = middle[i] + multiplier * atr_values[i];
        lower[i] = middle[i] - multiplier * atr_values[i];
    }

    // Use [-2] to avoid data leakage
    double keltner_upper = (n > 1) ? get_last_valid(std::vector<double>(upper.begin(), upper.end() - 1), 0.0) : 0.0;
    double keltner_middle = (n > 1) ? get_last_valid(std::vector<double>(middle.begin(), middle.end() - 1), 0.0) : 0.0;
    double keltner_lower = (n > 1) ? get_last_valid(std::vector<double>(lower.begin(), lower.end() - 1), 0.0) : 0.0;

    features["keltner_upper"] = keltner_upper;
    features["keltner_middle"] = keltner_middle;
    features["keltner_lower"] = keltner_lower;
    features["keltner_width"] = safe_divide(keltner_upper - keltner_lower, keltner_middle, 0.0);

    // Position within channel
    double channel_range = keltner_upper - keltner_lower;
    double keltner_position = safe_divide(current_close - keltner_lower, channel_range, 0.5);
    features["keltner_position"] = std::clamp(keltner_position, 0.0, 1.0);

    return features;
}

// =============================================================================
// ADX Indicators (4 features)
// =============================================================================

std::unordered_map<std::string, double> TechnicalIndicators::calculate_adx(
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close
) {
    std::unordered_map<std::string, double> features;
    features.reserve(4);  // ADX has 4 features

    if (close.size() < 28) {
        features["adx"] = 0.0;
        features["plus_di"] = 0.0;
        features["minus_di"] = 0.0;
        features["di_crossover"] = 0.0;
        return features;
    }

    int period = 14;
    size_t n = close.size();

    // Calculate +DM and -DM
    std::vector<double> plus_dm(n, 0.0);
    std::vector<double> minus_dm(n, 0.0);

    for (size_t i = 1; i < n; ++i) {
        double up_move = high[i] - high[i-1];
        double down_move = low[i-1] - low[i];

        if (up_move > down_move && up_move > 0) {
            plus_dm[i] = up_move;
        }
        if (down_move > up_move && down_move > 0) {
            minus_dm[i] = down_move;
        }
    }

    // Smoothed values
    std::vector<double> tr_values = true_range(high, low, close);
    std::vector<double> atr_smooth = ema(tr_values, period);
    std::vector<double> plus_dm_smooth = ema(plus_dm, period);
    std::vector<double> minus_dm_smooth = ema(minus_dm, period);

    // Calculate +DI and -DI
    std::vector<double> plus_di(n);
    std::vector<double> minus_di(n);
    for (size_t i = 0; i < n; ++i) {
        plus_di[i] = (atr_smooth[i] > 0) ? 100.0 * plus_dm_smooth[i] / atr_smooth[i] : 0.0;
        minus_di[i] = (atr_smooth[i] > 0) ? 100.0 * minus_dm_smooth[i] / atr_smooth[i] : 0.0;
    }

    // Calculate DX and ADX
    std::vector<double> dx(n);
    for (size_t i = 0; i < n; ++i) {
        double di_sum = plus_di[i] + minus_di[i];
        dx[i] = (di_sum > 0) ? 100.0 * std::abs(plus_di[i] - minus_di[i]) / di_sum : 0.0;
    }

    std::vector<double> adx_values = ema(dx, period);

    // Use [-2] to avoid data leakage
    features["adx"] = (n > 1) ? get_last_valid(std::vector<double>(adx_values.begin(), adx_values.end() - 1), 0.0) : 0.0;
    features["plus_di"] = (n > 1) ? get_last_valid(std::vector<double>(plus_di.begin(), plus_di.end() - 1), 0.0) : 0.0;
    features["minus_di"] = (n > 1) ? get_last_valid(std::vector<double>(minus_di.begin(), minus_di.end() - 1), 0.0) : 0.0;

    // DI crossover
    double di_crossover = 0.0;
    if (n >= 3) {
        double prev_plus = std::isfinite(plus_di[n-3]) ? plus_di[n-3] : 0.0;
        double prev_minus = std::isfinite(minus_di[n-3]) ? minus_di[n-3] : 0.0;
        double curr_plus = std::isfinite(plus_di[n-2]) ? plus_di[n-2] : 0.0;
        double curr_minus = std::isfinite(minus_di[n-2]) ? minus_di[n-2] : 0.0;

        if (prev_plus <= prev_minus && curr_plus > curr_minus) {
            di_crossover = 1.0;
        } else if (prev_plus >= prev_minus && curr_plus < curr_minus) {
            di_crossover = -1.0;
        }
    }
    features["di_crossover"] = di_crossover;

    return features;
}

// =============================================================================
// Ichimoku Indicators (6 features)
// =============================================================================

std::unordered_map<std::string, double> TechnicalIndicators::calculate_ichimoku(
    const std::vector<double>& high,
    const std::vector<double>& low,
    double current_close
) {
    std::unordered_map<std::string, double> features;
    features.reserve(6);  // Ichimoku has 6 features

    if (high.size() < 52) {
        features["tenkan"] = 0.0;
        features["kijun"] = 0.0;
        features["senkou_a"] = 0.0;
        features["senkou_b"] = 0.0;
        features["price_vs_cloud"] = 0.0;
        features["cloud_thickness"] = 0.0;
        return features;
    }

    size_t n = high.size();

    // Tenkan-sen (9-period) - use [-10:-1] to avoid leakage
    double tenkan_high = *std::max_element(high.end() - 10, high.end() - 1);
    double tenkan_low = *std::min_element(low.end() - 10, low.end() - 1);
    double tenkan = (tenkan_high + tenkan_low) / 2.0;

    // Kijun-sen (26-period) - use [-27:-1]
    double kijun_high = *std::max_element(high.end() - 27, high.end() - 1);
    double kijun_low = *std::min_element(low.end() - 27, low.end() - 1);
    double kijun = (kijun_high + kijun_low) / 2.0;

    // Senkou Span A
    double senkou_a = (tenkan + kijun) / 2.0;

    // Senkou Span B (52-period) - use [-53:-1]
    double senkou_b_high = *std::max_element(high.end() - 53, high.end() - 1);
    double senkou_b_low = *std::min_element(low.end() - 53, low.end() - 1);
    double senkou_b = (senkou_b_high + senkou_b_low) / 2.0;

    features["tenkan"] = safe_float(tenkan, 0.0);
    features["kijun"] = safe_float(kijun, 0.0);
    features["senkou_a"] = safe_float(senkou_a, 0.0);
    features["senkou_b"] = safe_float(senkou_b, 0.0);

    // Price vs Cloud
    double cloud_top = std::max(senkou_a, senkou_b);
    double cloud_bottom = std::min(senkou_a, senkou_b);

    double price_vs_cloud;
    if (current_close > cloud_top) {
        price_vs_cloud = 1.0;  // Above cloud
    } else if (current_close < cloud_bottom) {
        price_vs_cloud = -1.0;  // Below cloud
    } else {
        price_vs_cloud = 0.0;  // Inside cloud
    }
    features["price_vs_cloud"] = price_vs_cloud;

    // Cloud thickness
    features["cloud_thickness"] = safe_divide(cloud_top - cloud_bottom, current_close, 0.0);

    return features;
}

// =============================================================================
// Volume Indicators (8 features)
// =============================================================================

std::unordered_map<std::string, double> TechnicalIndicators::calculate_volume_indicators(
    const std::vector<double>& open,
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close,
    const std::vector<double>& volume
) {
    std::unordered_map<std::string, double> features;
    features.reserve(8);  // Volume indicators has 8 features
    size_t n = close.size();

    if (n < 20) {
        features["obv"] = 0.0;
        features["obv_trend"] = 0.0;
        features["obv_divergence"] = 0.0;
        features["mfi"] = 50.0;
        features["mfi_divergence"] = 0.0;
        features["accumulation_dist"] = 0.0;
        features["chaikin_mf"] = 0.0;
        features["volume_oscillator"] = 0.0;
        return features;
    }

    // Use empty volume array if not provided
    std::vector<double> vol = volume.empty() ? std::vector<double>(n, 0.0) : volume;

    // OBV (On Balance Volume)
    std::vector<double> obv(n, 0.0);
    for (size_t i = 1; i < n; ++i) {
        if (close[i] > close[i-1]) {
            obv[i] = obv[i-1] + vol[i];
        } else if (close[i] < close[i-1]) {
            obv[i] = obv[i-1] - vol[i];
        } else {
            obv[i] = obv[i-1];
        }
    }

    features["obv"] = (n >= 2) ? safe_float(obv[n-2], 0.0) : 0.0;

    // OBV Trend
    double obv_trend = 0.0;
    if (n >= 11) {
        double obv_change = obv[n-2] - obv[n-11];
        double sum_vol = 0.0;
        for (size_t i = n - 11; i < n - 1; ++i) {
            sum_vol += vol[i];
        }
        double avg_vol = sum_vol / 10.0;
        obv_trend = safe_divide(obv_change, avg_vol * 10.0, 0.0);
    }
    features["obv_trend"] = obv_trend;

    // OBV Divergence
    double obv_divergence = 0.0;
    if (n >= 11) {
        double price_change = close[n-2] - close[n-11];
        double obv_change = obv[n-2] - obv[n-11];
        if (price_change > 0 && obv_change < 0) {
            obv_divergence = -1.0;
        } else if (price_change < 0 && obv_change > 0) {
            obv_divergence = 1.0;
        }
    }
    features["obv_divergence"] = obv_divergence;

    // MFI (Money Flow Index)
    std::vector<double> typical_price(n);
    for (size_t i = 0; i < n; ++i) {
        typical_price[i] = (high[i] + low[i] + close[i]) / 3.0;
    }

    std::vector<double> positive_flow(n, 0.0);
    std::vector<double> negative_flow(n, 0.0);
    for (size_t i = 1; i < n; ++i) {
        double raw_money_flow = typical_price[i] * vol[i];
        if (typical_price[i] > typical_price[i-1]) {
            positive_flow[i] = raw_money_flow;
        } else if (typical_price[i] < typical_price[i-1]) {
            negative_flow[i] = raw_money_flow;
        }
    }

    int period = 14;
    double pos_sum = 0.0;
    double neg_sum = 0.0;
    for (int i = std::max(0, (int)n - period); i < (int)n; ++i) {
        pos_sum += positive_flow[i];
        neg_sum += negative_flow[i];
    }

    double mfi;
    if (neg_sum == 0.0) {
        mfi = (pos_sum > 0.0) ? 100.0 : 50.0;
    } else {
        double money_ratio = pos_sum / neg_sum;
        mfi = 100.0 - (100.0 / (1.0 + money_ratio));
    }
    features["mfi"] = std::clamp(mfi, 0.0, 100.0);

    // MFI Divergence
    double mfi_divergence = 0.0;
    if (n >= 24) {
        double pos_sum_old = 0.0;
        double neg_sum_old = 0.0;
        for (size_t i = n - 24; i < n - 10; ++i) {
            pos_sum_old += positive_flow[i];
            neg_sum_old += negative_flow[i];
        }

        double mfi_old;
        if (neg_sum_old > 0.0) {
            double mr_old = pos_sum_old / neg_sum_old;
            mfi_old = 100.0 - (100.0 / (1.0 + mr_old));
        } else {
            mfi_old = (pos_sum_old > 0.0) ? 100.0 : 50.0;
        }

        double price_change = close[n-2] - close[n-11];
        double mfi_change = mfi - mfi_old;
        if (price_change > 0 && mfi_change < 0) {
            mfi_divergence = -1.0;
        } else if (price_change < 0 && mfi_change > 0) {
            mfi_divergence = 1.0;
        }
    }
    features["mfi_divergence"] = mfi_divergence;

    // Accumulation/Distribution
    std::vector<double> ad(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        double hl_range = high[i] - low[i];
        double clv = (hl_range > 0) ? ((close[i] - low[i]) - (high[i] - close[i])) / hl_range : 0.0;
        double ad_change = clv * vol[i];
        ad[i] = (i > 0 ? ad[i-1] : 0.0) + ad_change;
    }
    features["accumulation_dist"] = (n >= 2) ? safe_float(ad[n-2], 0.0) : 0.0;

    // Chaikin Money Flow
    int cmf_period = 20;
    double cmf_num = 0.0;
    double cmf_den = 0.0;

    for (size_t i = n - cmf_period - 1; i < n - 1; ++i) {
        double hl_range = high[i] - low[i];
        double clv = (hl_range > 0) ? ((close[i] - low[i]) - (high[i] - close[i])) / hl_range : 0.0;
        cmf_num += clv * vol[i];
        cmf_den += vol[i];
    }

    double chaikin_mf = safe_divide(cmf_num, cmf_den, 0.0);
    features["chaikin_mf"] = std::clamp(chaikin_mf, -1.0, 1.0);

    // Volume Oscillator
    std::vector<double> vol_short = ema(vol, 5);
    std::vector<double> vol_long = ema(vol, 20);
    double vol_short_val = (n > 1) ? get_last_valid(std::vector<double>(vol_short.begin(), vol_short.end() - 1), 1.0) : 1.0;
    double vol_long_val = (n > 1) ? get_last_valid(std::vector<double>(vol_long.begin(), vol_long.end() - 1), 1.0) : 1.0;
    features["volume_oscillator"] = safe_divide(vol_short_val - vol_long_val, vol_long_val, 0.0);

    return features;
}

// =============================================================================
// Oscillators (6 features)
// =============================================================================

std::unordered_map<std::string, double> TechnicalIndicators::calculate_oscillators(
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close
) {
    std::unordered_map<std::string, double> features;
    features.reserve(6);  // Oscillators has 6 features
    size_t n = close.size();

    if (n < 28) {
        features["aroon_up"] = 0.0;
        features["aroon_down"] = 0.0;
        features["aroon_oscillator"] = 0.0;
        features["ultimate_oscillator"] = 50.0;
        features["ppo"] = 0.0;
        features["dpo"] = 0.0;
        return features;
    }

    // Aroon (25 period)
    int period = 25;
    auto high_it = std::max_element(high.end() - period - 1, high.end() - 1);
    auto low_it = std::min_element(low.end() - period - 1, low.end() - 1);

    int days_since_high = (high.end() - 1) - high_it;
    int days_since_low = (low.end() - 1) - low_it;

    double aroon_up = ((period - days_since_high) / (double)period) * 100.0;
    double aroon_down = ((period - days_since_low) / (double)period) * 100.0;

    features["aroon_up"] = safe_float(aroon_up, 0.0);
    features["aroon_down"] = safe_float(aroon_down, 0.0);
    features["aroon_oscillator"] = safe_float(aroon_up - aroon_down, 0.0);

    // Ultimate Oscillator
    std::vector<double> tr_vals = true_range(high, low, close);
    std::vector<double> bp(n, 0.0);

    for (size_t i = 1; i < n; ++i) {
        bp[i] = close[i] - std::min(low[i], close[i-1]);
    }

    double bp7 = 0.0, bp14 = 0.0, bp28 = 0.0;
    double tr7 = 0.0, tr14 = 0.0, tr28 = 0.0;

    for (size_t i = n - 8; i < n - 1; ++i) { bp7 += bp[i]; tr7 += tr_vals[i]; }
    for (size_t i = n - 15; i < n - 1; ++i) { bp14 += bp[i]; tr14 += tr_vals[i]; }
    for (size_t i = n - 29; i < n - 1; ++i) { bp28 += bp[i]; tr28 += tr_vals[i]; }

    double avg7 = safe_divide(bp7, tr7, 0.5);
    double avg14 = safe_divide(bp14, tr14, 0.5);
    double avg28 = safe_divide(bp28, tr28, 0.5);

    double ultimate_oscillator = ((4.0 * avg7) + (2.0 * avg14) + avg28) / 7.0 * 100.0;
    features["ultimate_oscillator"] = std::clamp(ultimate_oscillator, 0.0, 100.0);

    // PPO (Percentage Price Oscillator)
    std::vector<double> ema_12 = ema(close, 12);
    std::vector<double> ema_26 = ema(close, 26);

    std::vector<double> ppo_line(n);
    for (size_t i = 0; i < n; ++i) {
        ppo_line[i] = (ema_26[i] > 0) ? ((ema_12[i] - ema_26[i]) / ema_26[i]) * 100.0 : 0.0;
    }
    features["ppo"] = (n > 1) ? get_last_valid(std::vector<double>(ppo_line.begin(), ppo_line.end() - 1), 0.0) : 0.0;

    // DPO (Detrended Price Oscillator)
    int dpo_period = 20;
    std::vector<double> sma_vals = sma(close, dpo_period);
    int shift = dpo_period / 2 + 1;

    double dpo = 0.0;
    if (n > (size_t)shift + 1 && std::isfinite(sma_vals[n - shift - 1])) {
        dpo = close[n-2] - sma_vals[n - shift - 1];
    }
    features["dpo"] = safe_float(dpo, 0.0);

    return features;
}

// =============================================================================
// Pivot Points (3 features)
// =============================================================================

std::unordered_map<std::string, double> TechnicalIndicators::calculate_pivot_points(
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close
) {
    std::unordered_map<std::string, double> features;
    features.reserve(3);  // Pivot points has 3 features

    if (close.size() < 2) {
        features["pivot"] = 0.0;
        features["r1"] = 0.0;
        features["s1"] = 0.0;
        return features;
    }

    size_t n = close.size();
    double prev_high = safe_float(high[n-2], high[n-1]);
    double prev_low = safe_float(low[n-2], low[n-1]);
    double prev_close = safe_float(close[n-2], close[n-1]);

    double pivot = (prev_high + prev_low + prev_close) / 3.0;
    double r1 = 2.0 * pivot - prev_low;
    double s1 = 2.0 * pivot - prev_high;

    features["pivot"] = safe_float(pivot, 0.0);
    features["r1"] = safe_float(r1, 0.0);
    features["s1"] = safe_float(s1, 0.0);

    return features;
}

// =============================================================================
// Fibonacci Levels (3 features)
// =============================================================================

std::unordered_map<std::string, double> TechnicalIndicators::calculate_fibonacci(
    const std::vector<double>& high,
    const std::vector<double>& low,
    double current_close
) {
    std::unordered_map<std::string, double> features;
    features.reserve(3);  // Fibonacci has 3 features

    if (high.size() < 20) {
        features["fib_382"] = 0.0;
        features["fib_500"] = 0.0;
        features["fib_618"] = 0.0;
        return features;
    }

    size_t n = high.size();
    int lookback = std::min(50, (int)n - 1);

    double swing_high = *std::max_element(high.end() - lookback - 1, high.end() - 1);
    double swing_low = *std::min_element(low.end() - lookback - 1, low.end() - 1);

    double range_size = swing_high - swing_low;

    if (range_size == 0.0) {
        features["fib_382"] = 0.0;
        features["fib_500"] = 0.0;
        features["fib_618"] = 0.0;
        return features;
    }

    features["fib_382"] = safe_float(swing_high - 0.382 * range_size, 0.0);
    features["fib_500"] = safe_float(swing_high - 0.500 * range_size, 0.0);
    features["fib_618"] = safe_float(swing_high - 0.618 * range_size, 0.0);

    return features;
}

// =============================================================================
// Candlestick Patterns (7 features)
// =============================================================================

std::unordered_map<std::string, double> TechnicalIndicators::calculate_candlestick_patterns(
    const std::vector<double>& open,
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close
) {
    std::unordered_map<std::string, double> features;
    features.reserve(7);  // Candlestick patterns has 7 features
    size_t n = close.size();

    features["is_doji"] = 0.0;
    features["is_hammer"] = 0.0;
    features["is_shooting_star"] = 0.0;
    features["is_engulfing_bull"] = 0.0;
    features["is_engulfing_bear"] = 0.0;
    features["is_morning_star"] = 0.0;
    features["is_evening_star"] = 0.0;

    if (n < 4) {
        return features;
    }

    // Use previous candle to avoid data leakage
    double o = open[n-2];
    double h = high[n-2];
    double l = low[n-2];
    double c = close[n-2];

    double body = std::abs(c - o);
    double full_range = h - l;
    double upper_shadow = h - std::max(o, c);
    double lower_shadow = std::min(o, c) - l;

    // Average body size
    double sum_body = 0.0;
    for (int i = -11; i < -1; ++i) {
        sum_body += std::abs(close[n + i] - open[n + i]);
    }
    double avg_body = sum_body / 10.0;
    if (avg_body == 0.0) avg_body = 1.0;

    // Doji
    if (full_range > 0 && body / full_range < 0.1) {
        features["is_doji"] = 1.0;
    }

    // Hammer
    if (full_range > 0 && lower_shadow >= 2.0 * body && upper_shadow < body && body > 0) {
        features["is_hammer"] = 1.0;
    }

    // Shooting Star
    if (full_range > 0 && upper_shadow >= 2.0 * body && lower_shadow < body && body > 0) {
        features["is_shooting_star"] = 1.0;
    }

    // Two-bar patterns
    if (n >= 3) {
        double o1 = open[n-3];
        double c1 = close[n-3];

        // Bullish Engulfing
        if (c1 < o1 && c > o && c > o1 && o < c1) {
            features["is_engulfing_bull"] = 1.0;
        }

        // Bearish Engulfing
        if (c1 > o1 && c < o && c < o1 && o > c1) {
            features["is_engulfing_bear"] = 1.0;
        }
    }

    // Three-bar patterns
    if (n >= 4) {
        double o2 = open[n-4];
        double c2 = close[n-4];
        double o1 = open[n-3];
        double c1 = close[n-3];
        double body2 = std::abs(c2 - o2);
        double body1 = std::abs(c1 - o1);

        // Morning Star
        if (c2 < o2 && body1 < body2 * 0.3 && c > o && c > (o2 + c2) / 2.0) {
            features["is_morning_star"] = 1.0;
        }

        // Evening Star
        if (c2 > o2 && body1 < body2 * 0.3 && c < o && c < (o2 + c2) / 2.0) {
            features["is_evening_star"] = 1.0;
        }
    }

    return features;
}

// =============================================================================
// Additional Indicators (3 features)
// =============================================================================

std::unordered_map<std::string, double> TechnicalIndicators::calculate_additional(
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close
) {
    std::unordered_map<std::string, double> features;
    features.reserve(3);  // Additional has 3 features
    size_t n = close.size();

    if (n < 20) {
        features["cci"] = 0.0;
        features["price_channel_upper"] = 0.0;
        features["price_channel_lower"] = 0.0;
        return features;
    }

    // CCI (Commodity Channel Index)
    int period = 20;
    std::vector<double> typical_price(n);
    for (size_t i = 0; i < n; ++i) {
        typical_price[i] = (high[i] + low[i] + close[i]) / 3.0;
    }

    std::vector<double> tp_sma = sma(typical_price, period);

    // Mean deviation
    std::vector<double> mean_dev(n, 0.0);
    for (size_t i = period - 1; i < n; ++i) {
        double sum = 0.0;
        for (size_t j = i - period + 1; j <= i; ++j) {
            sum += std::abs(typical_price[j] - tp_sma[i]);
        }
        mean_dev[i] = sum / period;
    }

    std::vector<double> cci(n, 0.0);
    for (size_t i = period - 1; i < n; ++i) {
        if (mean_dev[i] > 0) {
            cci[i] = (typical_price[i] - tp_sma[i]) / (0.015 * mean_dev[i]);
        }
    }

    features["cci"] = (n > 1) ? get_last_valid(std::vector<double>(cci.begin(), cci.end() - 1), 0.0) : 0.0;

    // Price Channel (Donchian)
    int channel_period = 20;
    double price_channel_upper = *std::max_element(high.end() - channel_period - 1, high.end() - 1);
    double price_channel_lower = *std::min_element(low.end() - channel_period - 1, low.end() - 1);

    features["price_channel_upper"] = safe_float(price_channel_upper, 0.0);
    features["price_channel_lower"] = safe_float(price_channel_lower, 0.0);

    return features;
}

// =============================================================================
// Main Feature Extraction
// =============================================================================

std::unordered_map<std::string, double> TechnicalIndicators::extract_features(
    const std::vector<double>& open,
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close,
    const std::vector<double>& volume
) {
    // Pre-reserve for 59 technical indicator features
    std::unordered_map<std::string, double> features;
    features.reserve(59);

    size_t n = close.size();
    if (n < 2) {
        // Return all zeros for insufficient data
        for (const auto& name : get_feature_names()) {
            features[name] = 0.0;
        }
        return features;
    }

    // Use previous bar's close to avoid data leakage
    double current_close = safe_float(close[n-2], 0.0);

    // Calculate all indicator categories
    auto macd_features = calculate_macd(close);
    auto bb_features = calculate_bollinger_bands(close, current_close);
    auto keltner_features = calculate_keltner(close, high, low, current_close);
    auto adx_features = calculate_adx(high, low, close);
    auto ichimoku_features = calculate_ichimoku(high, low, current_close);
    auto volume_features = calculate_volume_indicators(open, high, low, close, volume);
    auto oscillator_features = calculate_oscillators(high, low, close);
    auto pivot_features = calculate_pivot_points(high, low, close);
    auto fib_features = calculate_fibonacci(high, low, current_close);
    auto candle_features = calculate_candlestick_patterns(open, high, low, close);
    auto additional_features = calculate_additional(high, low, close);

    // Merge all features
    features.insert(macd_features.begin(), macd_features.end());
    features.insert(bb_features.begin(), bb_features.end());
    features.insert(keltner_features.begin(), keltner_features.end());
    features.insert(adx_features.begin(), adx_features.end());
    features.insert(ichimoku_features.begin(), ichimoku_features.end());
    features.insert(volume_features.begin(), volume_features.end());
    features.insert(oscillator_features.begin(), oscillator_features.end());
    features.insert(pivot_features.begin(), pivot_features.end());
    features.insert(fib_features.begin(), fib_features.end());
    features.insert(candle_features.begin(), candle_features.end());
    features.insert(additional_features.begin(), additional_features.end());

    // Final safety check - ensure all values are finite
    for (auto& [key, value] : features) {
        if (!std::isfinite(value)) {
            features[key] = 0.0;
        }
    }

    return features;
}

// =============================================================================
// Feature Names
// =============================================================================

std::vector<std::string> TechnicalIndicators::get_feature_names() {
    return {
        // MACD (5)
        "macd_line", "macd_signal", "macd_histogram", "macd_crossover", "macd_divergence",
        // Bollinger Bands (8)
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct_b",
        "price_vs_bb_upper", "price_vs_bb_lower", "bb_squeeze",
        // Keltner (5)
        "keltner_upper", "keltner_middle", "keltner_lower", "keltner_width", "keltner_position",
        // ADX (4)
        "adx", "plus_di", "minus_di", "di_crossover",
        // Ichimoku (6)
        "tenkan", "kijun", "senkou_a", "senkou_b", "price_vs_cloud", "cloud_thickness",
        // Volume Indicators (8)
        "obv", "obv_trend", "obv_divergence", "mfi", "mfi_divergence",
        "accumulation_dist", "chaikin_mf", "volume_oscillator",
        // Other Oscillators (6)
        "aroon_up", "aroon_down", "aroon_oscillator",
        "ultimate_oscillator", "ppo", "dpo",
        // Pivot Points (3)
        "pivot", "r1", "s1",
        // Fibonacci (3)
        "fib_382", "fib_500", "fib_618",
        // Candlestick Patterns (7)
        "is_doji", "is_hammer", "is_shooting_star", "is_engulfing_bull", "is_engulfing_bear",
        "is_morning_star", "is_evening_star",
        // Additional (3)
        "cci", "price_channel_upper", "price_channel_lower"
    };
}

} // namespace v15
