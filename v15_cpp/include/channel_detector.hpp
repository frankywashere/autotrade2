#pragma once

#include "channel.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <Eigen/Dense>

namespace v15 {

/**
 * ChannelDetector - High-performance channel detection using Eigen
 *
 * Detects price channels using linear regression with ±2σ bounds.
 * Key insight: Use HIGHS for upper touches, LOWS for lower touches.
 *
 * Optimizations:
 * - Eigen matrices for efficient linear regression
 * - Vectorized operations where possible
 * - OpenMP parallelization for multi-window detection
 * - Cache-friendly memory layout
 *
 * Standard window sizes: [10, 20, 30, 40, 50, 60, 70, 80]
 */
class ChannelDetector {
public:
    /**
     * Detect a price channel in OHLCV data
     *
     * @param high High prices (must have at least window+1 bars)
     * @param low Low prices
     * @param close Close prices
     * @param window Number of bars for regression (default 50)
     * @param std_multiplier Std dev multiplier for bounds (default 2.0 = ±2σ)
     * @param touch_threshold Touch threshold as fraction of channel width (default 0.10 = 10%)
     * @param min_cycles Minimum alternating bounces for valid channel (default 1)
     * @return Channel object with all metrics
     *
     * Note: Uses last 'window' bars, excluding current bar to prevent data leakage
     */
    static Channel detect_channel(
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close,
        int window = 50,
        double std_multiplier = 2.0,
        double touch_threshold = 0.10,
        int min_cycles = 1
    );

    /**
     * Detect channels for multiple windows in parallel
     *
     * @param high High prices
     * @param low Low prices
     * @param close Close prices
     * @param windows Window sizes to detect (default: [10, 20, 30, 40, 50, 60, 70, 80])
     * @param std_multiplier Std dev multiplier for bounds (default 2.0)
     * @param touch_threshold Touch threshold as fraction of channel width (default 0.10)
     * @param min_cycles Minimum alternating bounces for valid channel (default 1)
     * @return Vector of Channel objects, one per window size
     *
     * Uses OpenMP to parallelize across windows
     */
    static std::vector<Channel> detect_multi_window(
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close,
        const std::vector<int>& windows = {10, 20, 30, 40, 50, 60, 70, 80},
        double std_multiplier = 2.0,
        double touch_threshold = 0.10,
        int min_cycles = 1
    );

private:
    /**
     * Detect bounces (boundary touches) in channel
     *
     * @param high High prices
     * @param low Low prices
     * @param upper_line Upper channel bound
     * @param lower_line Lower channel bound
     * @param threshold Touch threshold as fraction of channel width
     * @param[out] touches Touch events detected
     * @param[out] bounce_count Count of alternating touches
     * @param[out] complete_cycles Count of full round-trips (L→U→L or U→L→U)
     * @param[out] upper_touches Count of upper touches
     * @param[out] lower_touches Count of lower touches
     */
    static void detect_bounces(
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
    );

    /**
     * Calculate channel quality score
     *
     * Score = alternations × (1 + alternation_ratio)
     *
     * Rewards:
     * - More alternations (bounces) - primary factor
     * - Cleaner alternation pattern (higher ratio) - secondary factor
     *
     * @param channel Channel to score
     * @return Quality score (0.0+, typically 0-20)
     */
    static double calculate_quality_score(const Channel& channel);

    /**
     * Fast linear regression using Eigen
     *
     * Computes: y = slope * x + intercept
     *
     * @param y Dependent variable (close prices)
     * @param[out] slope Regression slope
     * @param[out] intercept Regression intercept
     * @param[out] r_squared R² value (0-1)
     * @param[out] std_dev Standard deviation of residuals
     */
    static void linear_regression(
        const Eigen::VectorXd& y,
        double& slope,
        double& intercept,
        double& r_squared,
        double& std_dev
    );
};

} // namespace v15
