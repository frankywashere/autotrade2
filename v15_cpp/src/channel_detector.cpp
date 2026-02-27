#include "channel_detector.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace v15 {

// =============================================================================
// Channel Methods
// =============================================================================

double Channel::position_at(int bar_index) const {
    // Safety: check if data is available
    if (close.empty() || upper_line.empty() || lower_line.empty()) {
        return 0.5;
    }

    // Safety: check array sizes match
    if (close.size() != upper_line.size() || close.size() != lower_line.size()) {
        return 0.5;
    }

    // Handle negative indexing
    if (bar_index < 0) {
        bar_index = static_cast<int>(close.size()) + bar_index;
    }

    // Safety: validate index bounds
    if (bar_index < 0 || bar_index >= static_cast<int>(close.size())) {
        return 0.5;
    }

    double price = close[bar_index];
    double upper = upper_line[bar_index];
    double lower = lower_line[bar_index];

    // Safety: validate all values are finite
    if (!std::isfinite(price) || !std::isfinite(upper) || !std::isfinite(lower)) {
        return 0.5;
    }

    // Safety: check for zero width
    double width = upper - lower;
    if (width <= 0.0) {
        return 0.5;
    }

    double position = (price - lower) / width;

    // Safety: validate result and clamp to [0, 1]
    if (!std::isfinite(position)) {
        return 0.5;
    }

    return std::max(0.0, std::min(1.0, position));
}

double Channel::slope_pct() const {
    if (close.empty()) {
        return 0.0;
    }

    double avg_price = std::accumulate(close.begin(), close.end(), 0.0) / close.size();

    // Safety: check for zero or negative average
    if (avg_price <= 0.0) {
        return 0.0;
    }

    // Safety: check for infinite or NaN slope
    if (!std::isfinite(slope)) {
        return 0.0;
    }

    double result = (slope / avg_price) * 100.0;

    // Safety: validate result
    if (!std::isfinite(result)) {
        return 0.0;
    }

    return result;
}

// =============================================================================
// ChannelDetector Implementation
// =============================================================================

void ChannelDetector::linear_regression(
    const Eigen::VectorXd& y,
    double& slope,
    double& intercept,
    double& r_squared,
    double& std_dev
) {
    // Initialize outputs to safe defaults
    slope = 0.0;
    intercept = 0.0;
    r_squared = 0.0;
    std_dev = 0.0;

    const int n = y.size();

    // ========================================================================
    // SAFETY CHECK: Validate input
    // ========================================================================
    if (n < 2) {
        // Need at least 2 points for regression
        return;
    }

    // Check for NaN or infinite values in input
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(y(i))) {
            // Invalid input data
            return;
        }
    }

    // Create design matrix [1, x] for linear regression
    Eigen::MatrixXd X(n, 2);
    for (int i = 0; i < n; ++i) {
        X(i, 0) = 1.0;      // Intercept column
        X(i, 1) = i;        // X values (0, 1, 2, ...)
    }

    // Solve normal equations: (X^T * X)^-1 * X^T * y
    // Using Eigen's efficient QR decomposition with pivoting
    // This is more numerically stable than normal equations
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(X);

    // Check if matrix is solvable (rank should be 2)
    if (qr.rank() < 2) {
        // Singular matrix - cannot solve
        // This can happen with perfectly flat data
        intercept = y.mean();
        slope = 0.0;
        r_squared = 0.0;
        std_dev = 0.0;
        return;
    }

    Eigen::Vector2d beta = qr.solve(y);

    intercept = beta(0);
    slope = beta(1);

    // ========================================================================
    // SAFETY CHECK: Validate regression coefficients
    // ========================================================================
    if (!std::isfinite(intercept) || !std::isfinite(slope)) {
        // Invalid coefficients - reset to safe values
        intercept = y.mean();
        slope = 0.0;
        r_squared = 0.0;
        std_dev = 0.0;
        return;
    }

    // Calculate fitted values
    Eigen::VectorXd y_fitted = X * beta;

    // Calculate residuals
    Eigen::VectorXd residuals = y - y_fitted;

    // Calculate R²
    double y_mean = y.mean();
    double ss_tot = (y.array() - y_mean).square().sum();
    double ss_res = residuals.squaredNorm();

    // Handle edge case: if ss_tot is zero, all y values are identical
    if (ss_tot <= 0.0 || !std::isfinite(ss_tot)) {
        r_squared = 0.0;
        std_dev = 0.0;
        return;
    }

    r_squared = 1.0 - ss_res / ss_tot;

    // Clamp R² to [0, 1] to handle numerical issues
    r_squared = std::max(0.0, std::min(1.0, r_squared));

    // Calculate standard deviation of residuals
    // Use n for population std (matches numpy default)
    double variance = residuals.squaredNorm() / n;

    // ========================================================================
    // SAFETY CHECK: Validate variance before sqrt
    // ========================================================================
    if (variance < 0.0 || !std::isfinite(variance)) {
        std_dev = 0.0;
        return;
    }

    std_dev = std::sqrt(variance);

    // Final validation
    if (!std::isfinite(std_dev)) {
        std_dev = 0.0;
    }
}

void ChannelDetector::detect_bounces(
    const std::vector<double>& high,
    const std::vector<double>& low,
    const Eigen::VectorXd& upper_line,
    const Eigen::VectorXd& lower_line,
    double threshold,
    std::vector<Touch>& touches,
    int& bounce_count,
    int& complete_cycles,
    int& upper_touches,
    int& lower_touches
) {
    touches.clear();
    bounce_count = 0;
    complete_cycles = 0;
    upper_touches = 0;
    lower_touches = 0;

    // ========================================================================
    // SAFETY CHECK: Validate input arrays
    // ========================================================================
    const int n = high.size();

    if (n != static_cast<int>(low.size()) ||
        n != upper_line.size() ||
        n != lower_line.size()) {
        // Inconsistent array sizes - cannot detect bounces
        return;
    }

    if (n <= 0) {
        // No data
        return;
    }

    if (threshold < 0.0 || !std::isfinite(threshold)) {
        // Invalid threshold
        return;
    }

    // Pre-allocate touches vector to avoid reallocations in hot loop
    // Worst case: every bar could touch both boundaries = 2*n touches
    touches.reserve(static_cast<size_t>(n * 2));

    // Detect touches
    // CRITICAL: This is the hot loop - optimize heavily
    for (int i = 0; i < n; ++i) {
        double width = upper_line(i) - lower_line(i);

        // Safety: skip invalid widths
        if (width <= 0.0 || !std::isfinite(width)) {
            continue;
        }

        // Safety: validate prices are finite
        if (!std::isfinite(high[i]) || !std::isfinite(low[i])) {
            continue;
        }

        // Check upper touch: HIGH near/above upper line
        double upper_dist = (upper_line(i) - high[i]) / width;

        // Check lower touch: LOW near/below lower line
        double lower_dist = (low[i] - lower_line(i)) / width;

        // Touch upper if HIGH is within threshold of upper (or above it)
        if (upper_dist <= threshold) {
            touches.emplace_back(i, TouchType::UPPER, high[i]);
        }

        // Touch lower if LOW is within threshold of lower (or below it)
        if (lower_dist <= threshold) {
            touches.emplace_back(i, TouchType::LOWER, low[i]);
        }
    }

    // Count alternating touches (bounces)
    if (!touches.empty()) {
        TouchType last_type = touches[0].touch_type;

        for (size_t i = 1; i < touches.size(); ++i) {
            if (touches[i].touch_type != last_type) {
                bounce_count++;
                last_type = touches[i].touch_type;
            }
        }
    }

    // Count complete cycles (full round-trips)
    // L→U→L or U→L→U
    for (size_t i = 0; i + 2 < touches.size(); ++i) {
        TouchType t1 = touches[i].touch_type;
        TouchType t2 = touches[i + 1].touch_type;
        TouchType t3 = touches[i + 2].touch_type;

        // Lower → Upper → Lower OR Upper → Lower → Upper
        if ((t1 == TouchType::LOWER && t2 == TouchType::UPPER && t3 == TouchType::LOWER) ||
            (t1 == TouchType::UPPER && t2 == TouchType::LOWER && t3 == TouchType::UPPER)) {
            complete_cycles++;
            i += 2;  // Skip to after this cycle
        }
    }

    // Count upper and lower touches
    for (const auto& touch : touches) {
        if (touch.touch_type == TouchType::UPPER) {
            upper_touches++;
        } else {
            lower_touches++;
        }
    }
}

double ChannelDetector::calculate_quality_score(const Channel& channel) {
    // Score = alternations × (1 + alternation_ratio)
    double quality = channel.alternations * (1.0 + channel.alternation_ratio);
    return quality;
}

Channel ChannelDetector::detect_channel(
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close,
    int window,
    double std_multiplier,
    double touch_threshold,
    int min_cycles
) {
    Channel channel;
    channel.window = window;

    // Debug tracking
    static int reject_count = 0;
    static bool debug_enabled = false;  // Set to true for debugging

    // ========================================================================
    // SAFETY CHECK 1: Validate input arrays
    // ========================================================================
    const size_t data_size = close.size();

    // Check all arrays have same size
    if (high.size() != data_size || low.size() != data_size) {
        // Inconsistent array sizes
        if (reject_count < 3 && debug_enabled) {
            std::cout << "      [REJECT #" << reject_count << "] Inconsistent array sizes\n";
            reject_count++;
        }
        return channel;
    }

    // Check minimum data requirement
    if (data_size < static_cast<size_t>(window)) {
        // Not enough data
        if (reject_count < 3 && debug_enabled) {
            std::cout << "      [REJECT #" << reject_count << "] data_size=" << data_size << " < window=" << window << "\n";
            reject_count++;
        }
        return channel;
    }

    // Check window is positive
    if (window <= 0) {
        if (reject_count < 3 && debug_enabled) {
            std::cout << "      [REJECT #" << reject_count << "] window <= 0\n";
            reject_count++;
        }
        return channel;
    }

    // Use all provided data (scanner already handles data leakage prevention)
    const int n = static_cast<int>(data_size);

    // Use data directly (no need to re-extract)
    const std::vector<double>& close_window = close;
    const std::vector<double>& high_window = high;
    const std::vector<double>& low_window = low;

    // ========================================================================
    // SAFETY CHECK 2: Validate price data quality
    // ========================================================================
    bool has_valid_data = false;
    double min_price = std::numeric_limits<double>::max();
    double max_price = std::numeric_limits<double>::lowest();

    for (int i = 0; i < n; ++i) {
        // Check for NaN or infinite values
        if (!std::isfinite(close_window[i]) ||
            !std::isfinite(high_window[i]) ||
            !std::isfinite(low_window[i])) {
            // Invalid price data
            if (reject_count < 3 && debug_enabled) {
                std::cout << "      [REJECT #" << reject_count << "] Non-finite price at i=" << i << "\n";
                reject_count++;
            }
            return channel;
        }

        // Check for non-positive prices
        if (close_window[i] <= 0.0 || high_window[i] <= 0.0 || low_window[i] <= 0.0) {
            // Invalid price data
            if (reject_count < 3 && debug_enabled) {
                std::cout << "      [REJECT #" << reject_count << "] Non-positive price at i=" << i << "\n";
                reject_count++;
            }
            return channel;
        }

        // Track price range
        min_price = std::min(min_price, low_window[i]);
        max_price = std::max(max_price, high_window[i]);

        // Check if we have any variation
        if (i > 0 && close_window[i] != close_window[0]) {
            has_valid_data = true;
        }
    }

    // Check for completely flat prices (all zeros or all same value)
    // A small epsilon to detect essentially flat data
    const double price_range = max_price - min_price;
    const double avg_price_estimate = (max_price + min_price) / 2.0;

    if (avg_price_estimate <= 0.0) {
        // Invalid average price
        if (reject_count < 3 && debug_enabled) {
            std::cout << "      [REJECT #" << reject_count << "] avg_price_estimate <= 0\n";
            reject_count++;
        }
        return channel;
    }

    // If price range is less than 0.001% of average, treat as flat
    const double range_pct = (price_range / avg_price_estimate) * 100.0;
    if (range_pct < 0.001) {
        // Essentially flat prices - no meaningful channel
        if (reject_count < 3 && debug_enabled) {
            std::cout << "      [REJECT #" << reject_count << "] Flat prices range_pct=" << range_pct << "\n";
            reject_count++;
        }
        return channel;
    }

    // Store OHLC data
    channel.close = close_window;
    channel.high = high_window;
    channel.low = low_window;

    // Convert to Eigen vector for regression
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; ++i) {
        y(i) = close_window[i];
    }

    // ========================================================================
    // SAFETY CHECK 3: Perform linear regression with exception handling
    // ========================================================================
    try {
        linear_regression(y, channel.slope, channel.intercept,
                         channel.r_squared, channel.std_dev);
    } catch (const std::exception& e) {
        // Regression failed - return invalid channel
        return channel;
    }

    // ========================================================================
    // SAFETY CHECK 4: Validate regression outputs
    // ========================================================================
    if (!std::isfinite(channel.slope) ||
        !std::isfinite(channel.intercept) ||
        !std::isfinite(channel.r_squared) ||
        !std::isfinite(channel.std_dev)) {
        // Invalid regression parameters
        if (reject_count < 3 && debug_enabled) {
            std::cout << "      [REJECT #" << reject_count << "] Non-finite regression params\n";
            reject_count++;
        }
        return channel;
    }

    // R² must be in [0, 1]
    if (channel.r_squared < 0.0 || channel.r_squared > 1.0) {
        // Invalid R²
        if (reject_count < 3 && debug_enabled) {
            std::cout << "      [REJECT #" << reject_count << "] Invalid R²=" << channel.r_squared << "\n";
            reject_count++;
        }
        return channel;
    }

    // std_dev must be positive for a valid channel
    if (channel.std_dev <= 0.0) {
        // No variance - flat channel
        if (reject_count < 3 && debug_enabled) {
            std::cout << "      [REJECT #" << reject_count << "] std_dev=" << channel.std_dev << " <= 0\n";
            reject_count++;
        }
        return channel;
    }

    // Calculate center line
    Eigen::VectorXd center_line(n);
    for (int i = 0; i < n; ++i) {
        center_line(i) = channel.slope * i + channel.intercept;
    }

    // Calculate upper and lower bounds
    Eigen::VectorXd upper_line = center_line.array() + std_multiplier * channel.std_dev;
    Eigen::VectorXd lower_line = center_line.array() - std_multiplier * channel.std_dev;

    // ========================================================================
    // SAFETY CHECK 5: Validate channel bounds
    // ========================================================================
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(center_line(i)) ||
            !std::isfinite(upper_line(i)) ||
            !std::isfinite(lower_line(i))) {
            // Invalid channel bounds
            if (reject_count < 3 && debug_enabled) {
                std::cout << "      [REJECT #" << reject_count << "] Non-finite channel bounds at i=" << i << "\n";
                reject_count++;
            }
            return channel;
        }

        // Check that upper > lower (should always be true, but verify)
        if (upper_line(i) <= lower_line(i)) {
            // Invalid bounds
            if (reject_count < 3 && debug_enabled) {
                std::cout << "      [REJECT #" << reject_count << "] upper <= lower at i=" << i << "\n";
                reject_count++;
            }
            return channel;
        }
    }

    // Store bounds
    channel.center_line.resize(n);
    channel.upper_line.resize(n);
    channel.lower_line.resize(n);

    for (int i = 0; i < n; ++i) {
        channel.center_line[i] = center_line(i);
        channel.upper_line[i] = upper_line(i);
        channel.lower_line[i] = lower_line(i);
    }

    // Calculate channel width as percentage
    double avg_price = y.mean();
    if (avg_price > 0.0) {
        double width = upper_line(n - 1) - lower_line(n - 1);
        channel.width_pct = (width / avg_price) * 100.0;

        // ========================================================================
        // SAFETY CHECK 6: Validate width calculation
        // ========================================================================
        if (!std::isfinite(channel.width_pct) || channel.width_pct < 0.0) {
            channel.width_pct = 0.0;
        }
    } else {
        channel.width_pct = 0.0;
    }

    // Debug output removed for cleaner output - uncomment for debugging:
    // static std::atomic<int> detect_call_count{0};
    // if (detect_call_count.fetch_add(1) == 0) { ... }

    // Detect bounces
    detect_bounces(
        high_window, low_window, upper_line, lower_line, touch_threshold,
        channel.touches, channel.bounce_count, channel.complete_cycles,
        channel.upper_touches, channel.lower_touches
    );

    // ========================================================================
    // SAFETY CHECK 7: Validate bounce detection results
    // ========================================================================
    if (channel.bounce_count < 0) {
        channel.bounce_count = 0;
    }

    if (channel.complete_cycles < 0) {
        channel.complete_cycles = 0;
    }

    if (channel.upper_touches < 0) {
        channel.upper_touches = 0;
    }

    if (channel.lower_touches < 0) {
        channel.lower_touches = 0;
    }

    // Calculate alternation metrics
    channel.alternations = channel.bounce_count;

    if (channel.touches.size() > 1) {
        channel.alternation_ratio = static_cast<double>(channel.bounce_count) /
                                   (channel.touches.size() - 1);

        // Clamp to [0, 1] to handle edge cases
        channel.alternation_ratio = std::max(0.0, std::min(1.0, channel.alternation_ratio));
    } else {
        channel.alternation_ratio = 0.0;
    }

    // ========================================================================
    // SAFETY CHECK 8: Validate alternation ratio
    // ========================================================================
    if (!std::isfinite(channel.alternation_ratio)) {
        channel.alternation_ratio = 0.0;
    }

    // Determine direction based on slope
    double slope_pct = channel.slope_pct();

    // ========================================================================
    // SAFETY CHECK 9: Validate slope percentage
    // ========================================================================
    if (!std::isfinite(slope_pct)) {
        slope_pct = 0.0;
    }

    if (slope_pct > 0.05) {
        channel.direction = ChannelDirection::BULL;
    } else if (slope_pct < -0.05) {
        channel.direction = ChannelDirection::BEAR;
    } else {
        channel.direction = ChannelDirection::SIDEWAYS;
    }

    // Valid if enough alternating bounces
    channel.valid = (channel.bounce_count >= min_cycles);

    // Debug: Log valid assignment
    static int valid_debug_count = 0;
    if (valid_debug_count < 5 && debug_enabled) {
        std::cout << "      [VALID_SET #" << valid_debug_count << "] bounce_count=" << channel.bounce_count
                  << " min_cycles=" << min_cycles
                  << " valid=" << channel.valid << "\n";
        valid_debug_count++;
    }

    // Calculate quality score
    channel.quality_score = calculate_quality_score(channel);

    // ========================================================================
    // SAFETY CHECK 10: Validate quality score
    // ========================================================================
    if (!std::isfinite(channel.quality_score) || channel.quality_score < 0.0) {
        channel.quality_score = 0.0;
    }

    // ========================================================================
    // MEMORY OPTIMIZATION: Cache values and strip heavy vectors
    // All vector-dependent calculations are complete at this point
    // ========================================================================

    // Cache first/last values before clearing vectors
    if (!channel.upper_line.empty()) {
        channel.first_upper_val = channel.upper_line.front();
        channel.last_upper_val = channel.upper_line.back();
        channel.tail_count = std::min(5, static_cast<int>(channel.upper_line.size()));
        for (int i = 0; i < channel.tail_count; ++i) {
            int idx = static_cast<int>(channel.upper_line.size()) - channel.tail_count + i;
            channel.upper_line_tail[i] = channel.upper_line[idx];
        }
    }
    if (!channel.lower_line.empty()) {
        channel.first_lower_val = channel.lower_line.front();
        channel.last_lower_val = channel.lower_line.back();
        for (int i = 0; i < channel.tail_count; ++i) {
            int idx = static_cast<int>(channel.lower_line.size()) - channel.tail_count + i;
            channel.lower_line_tail[i] = channel.lower_line[idx];
        }
    }
    if (!channel.center_line.empty()) {
        channel.first_center_val = channel.center_line.front();
        channel.last_center_val = channel.center_line.back();
    }

    // Free heavy vector memory using swap (clear() preserves capacity!)
    std::vector<double>().swap(channel.upper_line);
    std::vector<double>().swap(channel.lower_line);
    std::vector<double>().swap(channel.center_line);
    std::vector<double>().swap(channel.close);
    std::vector<double>().swap(channel.high);
    std::vector<double>().swap(channel.low);
    // Keep touches for now - it's small

    return channel;
}

Channel ChannelDetector::detect_channel(
    const double* high,
    const double* low,
    const double* close,
    size_t data_size,
    int window,
    double std_multiplier,
    double touch_threshold,
    int min_cycles
) {
    // Create vectors from pointers - this is temporary until full refactor
    std::vector<double> high_vec(high, high + data_size);
    std::vector<double> low_vec(low, low + data_size);
    std::vector<double> close_vec(close, close + data_size);

    return detect_channel(high_vec, low_vec, close_vec, window,
                         std_multiplier, touch_threshold, min_cycles);
}

std::vector<Channel> ChannelDetector::detect_multi_window(
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close,
    const std::vector<int>& windows,
    double std_multiplier,
    double touch_threshold,
    int min_cycles
) {
    const int num_windows = windows.size();
    std::vector<Channel> channels(num_windows);

    // Parallelize across windows using OpenMP
    // Each window detection is independent
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int i = 0; i < num_windows; ++i) {
        channels[i] = detect_channel(
            high, low, close,
            windows[i],
            std_multiplier,
            touch_threshold,
            min_cycles
        );
    }

    return channels;
}

} // namespace v15
