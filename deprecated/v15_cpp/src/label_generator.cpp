#include "label_generator.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace v15 {

// =============================================================================
// CONSTRUCTOR
// =============================================================================

LabelGenerator::LabelGenerator(const Config& config)
    : config_(config)
{}

// =============================================================================
// BREAK SCANNING
// =============================================================================

void LabelGenerator::project_bounds(
    const Channel& channel,
    int bars_forward,
    double& center_out,
    double& upper_out,
    double& lower_out
) const {
    // x at end of channel = window - 1
    // x for first forward bar = window
    // x projected forward = window + bars_forward
    int projection_x = channel.window_size + bars_forward;

    // Projected center line
    center_out = channel.slope * projection_x + channel.intercept;

    // Use 2 * std_dev for channel width (standard +-2 sigma bounds)
    constexpr double std_multiplier = 2.0;
    upper_out = center_out + std_multiplier * channel.std_dev;
    lower_out = center_out - std_multiplier * channel.std_dev;
}

BreakResult LabelGenerator::scan_for_break(
    const Channel& channel,
    const double* forward_high,
    const double* forward_low,
    const double* forward_close,
    int n_forward,
    int max_scan
) const {
    BreakResult result;

    // Validate inputs
    if (n_forward <= 0 || !forward_high || !forward_low || !forward_close) {
        return result;
    }

    if (channel.std_dev <= 0.0) {
        return result;
    }

    // Validate max_scan is positive
    if (max_scan <= 0) {
        return result;
    }

    // Actual scan length - ensure we don't exceed array bounds
    int actual_scan = std::min(n_forward, max_scan);
    if (actual_scan <= 0) {
        return result;
    }
    result.scan_bars_used = actual_scan;

    // State tracking
    std::vector<ExitEvent> exit_events;
    ExitEvent* current_exit = nullptr;
    bool inside_channel = true;
    bool first_break_found = false;
    bool first_permanent_found = false;

    // Scan each bar
    for (int bar_idx = 0; bar_idx < actual_scan; ++bar_idx) {
        double high = forward_high[bar_idx];
        double low = forward_low[bar_idx];
        double close = forward_close[bar_idx];

        // Project bounds to this bar
        double center, upper, lower;
        project_bounds(channel, bar_idx, center, upper, lower);

        // Track FIRST TOUCH - first bar where CLOSE went outside bounds
        if (result.first_touch_bar < 0) {
            if (close > upper) {
                result.first_touch_bar = bar_idx;
                result.first_touch_direction = 1;  // UP
                result.first_touch_price = close;
            } else if (close < lower) {
                result.first_touch_bar = bar_idx;
                result.first_touch_direction = 0;  // DOWN
                result.first_touch_price = close;
            }
        }

        // Get current exit direction
        int current_exit_direction = -1;
        if (current_exit != nullptr) {
            current_exit_direction = current_exit->exit_type;
        }

        // Check for UPPER breach
        if (close > upper) {
            double magnitude = (channel.std_dev > 0.0)
                ? (close - upper) / channel.std_dev
                : 0.0;

            // Count as break if magnitude exceeds threshold AND either:
            // 1. We're inside the channel (normal break), OR
            // 2. We were outside in OPPOSITE direction (direction reversal)
            bool is_direction_reversal = (current_exit_direction == 0);  // Was lower

            if (magnitude >= config_.min_break_magnitude &&
                (inside_channel || is_direction_reversal)) {

                // If direction reversal, close previous exit first
                if (is_direction_reversal && current_exit != nullptr) {
                    current_exit->returned = true;
                    current_exit->bars_outside = bar_idx - current_exit->bar_index;
                    current_exit->return_bar = bar_idx;
                    exit_events.push_back(*current_exit);
                    delete current_exit;
                    current_exit = nullptr;
                }

                // New exit event
                current_exit = new ExitEvent();
                current_exit->bar_index = bar_idx;
                current_exit->exit_type = 1;  // upper
                current_exit->exit_price = close;
                current_exit->magnitude = magnitude;
                inside_channel = false;

                // Record first break
                if (!first_break_found) {
                    first_break_found = true;
                    result.break_detected = true;
                    result.break_bar = bar_idx;
                    result.break_direction = 1;  // UP
                    result.break_price = close;
                    result.projected_upper = upper;
                    result.projected_lower = lower;
                    result.break_magnitude = magnitude;
                }
            }
        }
        // Check for LOWER breach
        else if (close < lower) {
            double magnitude = (channel.std_dev > 0.0)
                ? (lower - close) / channel.std_dev
                : 0.0;

            bool is_direction_reversal = (current_exit_direction == 1);  // Was upper

            if (magnitude >= config_.min_break_magnitude &&
                (inside_channel || is_direction_reversal)) {

                // If direction reversal, close previous exit first
                if (is_direction_reversal && current_exit != nullptr) {
                    current_exit->returned = true;
                    current_exit->bars_outside = bar_idx - current_exit->bar_index;
                    current_exit->return_bar = bar_idx;
                    exit_events.push_back(*current_exit);
                    delete current_exit;
                    current_exit = nullptr;
                }

                // New exit event
                current_exit = new ExitEvent();
                current_exit->bar_index = bar_idx;
                current_exit->exit_type = 0;  // lower
                current_exit->exit_price = close;
                current_exit->magnitude = magnitude;
                inside_channel = false;

                // Record first break
                if (!first_break_found) {
                    first_break_found = true;
                    result.break_detected = true;
                    result.break_bar = bar_idx;
                    result.break_direction = 0;  // DOWN
                    result.break_price = close;
                    result.projected_upper = upper;
                    result.projected_lower = lower;
                    result.break_magnitude = magnitude;
                }
            }
        }

        // Check for return to channel (close inside bounds)
        if (!inside_channel && lower <= close && close <= upper) {
            inside_channel = true;
            if (current_exit != nullptr) {
                current_exit->returned = true;
                current_exit->bars_outside = bar_idx - current_exit->bar_index;
                current_exit->return_bar = bar_idx;
                exit_events.push_back(*current_exit);
                delete current_exit;
                current_exit = nullptr;
            }
        }

        // Check for PERMANENT break - FIRST exit that stays outside 5+ bars
        if (!inside_channel && current_exit != nullptr && !first_permanent_found) {
            int bars_outside_so_far = bar_idx - current_exit->bar_index;
            if (bars_outside_so_far >= config_.return_threshold_bars) {
                first_permanent_found = true;
                result.permanent_break_bar = current_exit->bar_index;
                result.permanent_break_direction = current_exit->exit_type;

                // Calculate magnitude at permanent break bar
                double perm_center, perm_upper, perm_lower;
                project_bounds(channel, current_exit->bar_index,
                             perm_center, perm_upper, perm_lower);

                if (channel.std_dev > 0.0) {
                    if (current_exit->exit_type == 1) {  // upper
                        result.permanent_break_magnitude =
                            (current_exit->exit_price - perm_upper) / channel.std_dev;
                    } else {  // lower
                        result.permanent_break_magnitude =
                            (perm_lower - current_exit->exit_price) / channel.std_dev;
                    }
                }
            }
        }
    }

    // Handle final exit if still outside at end of scan
    if (!inside_channel && current_exit != nullptr) {
        current_exit->returned = false;
        current_exit->bars_outside = actual_scan - current_exit->bar_index;
        exit_events.push_back(*current_exit);
        delete current_exit;
        current_exit = nullptr;
    }

    // Store exit events
    result.all_exit_events = exit_events;

    // Calculate exit statistics
    if (!exit_events.empty()) {
        result.exits_returned_count = 0;
        for (const auto& evt : exit_events) {
            if (evt.returned) {
                ++result.exits_returned_count;
            }
        }
        result.exits_stayed_out_count = exit_events.size() - result.exits_returned_count;
        result.exit_return_rate = static_cast<double>(result.exits_returned_count) /
                                  exit_events.size();
        result.false_break_count = result.exits_returned_count;
        result.false_break_rate = result.exit_return_rate;
    }

    // Count round-trip bounces
    if (exit_events.size() >= 2) {
        for (size_t i = 1; i < exit_events.size(); ++i) {
            const auto& prev = exit_events[i - 1];
            const auto& curr = exit_events[i];
            if (prev.exit_type != curr.exit_type &&
                prev.returned && curr.returned) {
                ++result.round_trip_bounces;
            }
        }
    }

    // PERMANENT BREAK LOGIC
    // If no exit stayed 5+ bars during scan, find last unreturned exit
    if (!first_permanent_found) {
        const ExitEvent* permanent_exit = nullptr;
        for (auto it = exit_events.rbegin(); it != exit_events.rend(); ++it) {
            if (!it->returned) {
                permanent_exit = &(*it);
                break;
            }
        }

        if (permanent_exit != nullptr) {
            result.permanent_break_bar = permanent_exit->bar_index;
            result.permanent_break_direction = permanent_exit->exit_type;

            double perm_center, perm_upper, perm_lower;
            project_bounds(channel, permanent_exit->bar_index,
                         perm_center, perm_upper, perm_lower);

            if (channel.std_dev > 0.0) {
                if (permanent_exit->exit_type == 1) {  // upper
                    result.permanent_break_magnitude =
                        (permanent_exit->exit_price - perm_upper) / channel.std_dev;
                } else {  // lower
                    result.permanent_break_magnitude =
                        (perm_lower - permanent_exit->exit_price) / channel.std_dev;
                }
            }

            result.bars_verified_permanent = permanent_exit->bars_outside;
        } else {
            // No permanent break found - all returned
            result.permanent_break_direction = -1;
            result.permanent_break_bar = -1;
            result.bars_verified_permanent = 0;
        }
    } else {
        // Permanent break found during scan
        for (const auto& evt : exit_events) {
            if (evt.bar_index == result.permanent_break_bar) {
                result.bars_verified_permanent = evt.bars_outside;
                break;
            }
        }
    }

    // Determine scan timeout
    result.scan_timed_out = false;
    if (result.scan_bars_used >= max_scan) {
        if (result.permanent_break_bar < 0) {
            result.scan_timed_out = true;
        } else if (result.bars_verified_permanent < config_.return_threshold_bars) {
            result.scan_timed_out = true;
        }
    }

    // Determine if first break was permanent or false
    if (result.break_detected && !exit_events.empty()) {
        const auto& first_exit = exit_events[0];
        result.is_false_break = first_exit.returned;
        result.is_permanent = !first_exit.returned;

        if (first_exit.returned) {
            result.bars_until_return = first_exit.bars_outside;
            result.return_bar = first_exit.return_bar;
        } else {
            if (first_exit.bars_outside >= config_.return_threshold_bars) {
                result.is_permanent = true;
            }
        }
    }

    return result;
}

// =============================================================================
// RSI COMPUTATION
// =============================================================================

void LabelGenerator::compute_rsi(
    const double* prices,
    int n_prices,
    int period,
    double* rsi_out
) const {
    if (n_prices < period + 1 || !prices || !rsi_out) {
        return;
    }

    // Initialize RSI to 50 (neutral)
    for (int i = 0; i < n_prices; ++i) {
        rsi_out[i] = 50.0;
    }

    // Calculate price changes
    std::vector<double> gains(n_prices, 0.0);
    std::vector<double> losses(n_prices, 0.0);

    for (int i = 1; i < n_prices; ++i) {
        double change = prices[i] - prices[i - 1];
        if (change > 0) {
            gains[i] = change;
        } else {
            losses[i] = -change;
        }
    }

    // Calculate initial average gain/loss using SMA
    double avg_gain = 0.0;
    double avg_loss = 0.0;
    for (int i = 1; i <= period; ++i) {
        avg_gain += gains[i];
        avg_loss += losses[i];
    }
    avg_gain /= period;
    avg_loss /= period;

    // Calculate RSI for initial point
    if (avg_loss > 0.0) {
        double rs = avg_gain / avg_loss;
        double rsi_val = 100.0 - (100.0 / (1.0 + rs));
        // Clamp to valid range [0, 100]
        rsi_out[period] = std::max(0.0, std::min(100.0, rsi_val));
    } else {
        rsi_out[period] = (avg_gain > 0.0) ? 100.0 : 50.0;
    }

    // Calculate RSI for remaining points using Wilder's smoothing
    for (int i = period + 1; i < n_prices; ++i) {
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period;
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period;

        if (avg_loss > 0.0) {
            double rs = avg_gain / avg_loss;
            double rsi_val = 100.0 - (100.0 / (1.0 + rs));
            // Clamp to valid range [0, 100]
            rsi_out[i] = std::max(0.0, std::min(100.0, rsi_val));
        } else {
            rsi_out[i] = (avg_gain > 0.0) ? 100.0 : 50.0;
        }
    }
}

void LabelGenerator::compute_rsi_labels(
    const double* close_prices,
    int close_array_size,
    int channel_end_idx,
    int first_break_bar,
    int permanent_break_bar,
    int channel_window,
    ChannelLabels& labels_out
) const {
    // Validate inputs
    if (!close_prices || close_array_size < config_.rsi_period + 1 ||
        channel_end_idx < 0 || channel_end_idx >= close_array_size) {
        // Set defaults
        labels_out.rsi_at_first_break = 50.0;
        labels_out.rsi_at_permanent_break = 50.0;
        labels_out.rsi_at_channel_end = 50.0;
        labels_out.rsi_overbought_at_break = false;
        labels_out.rsi_oversold_at_break = false;
        labels_out.rsi_divergence_at_break = 0;
        labels_out.rsi_trend_in_channel = 0;
        labels_out.rsi_range_in_channel = 0.0;
        return;
    }

    // Compute RSI for entire array
    std::vector<double> rsi_values(close_array_size);
    compute_rsi(close_prices, close_array_size, config_.rsi_period, rsi_values.data());

    // 1. RSI at channel end
    labels_out.rsi_at_channel_end = rsi_values[channel_end_idx];

    // 2. RSI at first break
    int first_break_idx = channel_end_idx + 1 + first_break_bar;
    if (first_break_idx >= 0 && first_break_idx < close_array_size) {
        double rsi_at_first = rsi_values[first_break_idx];
        // Validate RSI is in valid range
        rsi_at_first = std::max(0.0, std::min(100.0, rsi_at_first));
        labels_out.rsi_at_first_break = rsi_at_first;
        labels_out.rsi_overbought_at_break = (rsi_at_first > config_.rsi_overbought);
        labels_out.rsi_oversold_at_break = (rsi_at_first < config_.rsi_oversold);
    } else {
        labels_out.rsi_at_first_break = 50.0;
        labels_out.rsi_overbought_at_break = false;
        labels_out.rsi_oversold_at_break = false;
    }

    // 3. RSI at permanent break
    if (permanent_break_bar >= 0) {
        int perm_break_idx = channel_end_idx + 1 + permanent_break_bar;
        if (perm_break_idx >= 0 && perm_break_idx < close_array_size) {
            double rsi_at_perm = rsi_values[perm_break_idx];
            // Validate RSI is in valid range
            labels_out.rsi_at_permanent_break = std::max(0.0, std::min(100.0, rsi_at_perm));
        } else {
            labels_out.rsi_at_permanent_break = 50.0;
        }
    } else {
        labels_out.rsi_at_permanent_break = 50.0;
    }

    // 4. RSI divergence at break
    int channel_start_idx = channel_end_idx - channel_window + 1;
    labels_out.rsi_divergence_at_break = 0;

    if (channel_start_idx >= 0 && first_break_idx < close_array_size &&
        channel_start_idx < close_array_size) {

        double price_start = close_prices[channel_start_idx];
        double price_break = close_prices[std::min(first_break_idx, close_array_size - 1)];
        double rsi_start = rsi_values[channel_start_idx];
        double rsi_break = rsi_values[std::min(first_break_idx, close_array_size - 1)];

        bool price_rising = (price_break > price_start);
        bool price_falling = (price_break < price_start);
        bool rsi_rising = (rsi_break > rsi_start);
        bool rsi_falling = (rsi_break < rsi_start);

        if (price_rising && rsi_falling) {
            labels_out.rsi_divergence_at_break = -1;  // Bearish divergence
        } else if (price_falling && rsi_rising) {
            labels_out.rsi_divergence_at_break = 1;   // Bullish divergence
        }
    }

    // 5. RSI trend in channel
    labels_out.rsi_trend_in_channel = 0;
    labels_out.rsi_range_in_channel = 0.0;

    if (channel_start_idx >= 0 && channel_end_idx < close_array_size) {
        int start_valid = std::max(0, channel_start_idx);
        int end_valid = std::min(channel_end_idx + 1, close_array_size);

        if (end_valid > start_valid) {
            // Trend: compare end to start
            double rsi_start = rsi_values[start_valid];
            double rsi_end = rsi_values[end_valid - 1];
            double rsi_change = rsi_end - rsi_start;

            if (rsi_change > config_.rsi_trend_threshold) {
                labels_out.rsi_trend_in_channel = 1;  // Rising
            } else if (rsi_change < -config_.rsi_trend_threshold) {
                labels_out.rsi_trend_in_channel = -1;  // Falling
            }

            // Range: max - min
            double rsi_min = rsi_values[start_valid];
            double rsi_max = rsi_values[start_valid];
            for (int i = start_valid; i < end_valid; ++i) {
                rsi_min = std::min(rsi_min, rsi_values[i]);
                rsi_max = std::max(rsi_max, rsi_values[i]);
            }
            labels_out.rsi_range_in_channel = rsi_max - rsi_min;
        }
    }
}

// =============================================================================
// LABEL GENERATION
// =============================================================================

double LabelGenerator::calculate_durability_score(
    int false_break_count,
    int total_exits,
    double avg_bars_outside
) const {
    if (total_exits == 0) {
        return 0.0;
    }

    double false_break_rate = static_cast<double>(false_break_count) / total_exits;

    // Quick return factor: e^(-0.1 * avg_bars)
    double quick_return_factor = (false_break_count > 0)
        ? std::exp(-0.1 * avg_bars_outside)
        : 0.0;

    // Volume factor: log scaling
    double volume_factor = std::log1p(false_break_count) / std::log1p(10.0);

    // Composite score
    return false_break_rate * (1.0 + 0.3 * quick_return_factor + 0.2 * volume_factor);
}

ChannelLabels LabelGenerator::generate_labels_forward_scan(
    const Channel& channel,
    int channel_end_idx,
    const double* forward_high,
    const double* forward_low,
    const double* forward_close,
    int n_forward,
    int max_scan,
    int next_channel_direction,
    const double* full_close_prices,
    int full_close_size
) const {
    ChannelLabels labels;
    labels.timeframe = channel.timeframe;

    // Store source channel parameters
    labels.source_channel_slope = channel.slope;
    labels.source_channel_intercept = channel.intercept;
    labels.source_channel_std_dev = channel.std_dev;
    labels.source_channel_r_squared = channel.r_squared;
    labels.source_channel_direction = static_cast<int>(channel.direction);
    labels.source_channel_bounce_count = channel.bounce_count;
    labels.source_channel_start_ts = channel.start_timestamp_ms;
    labels.source_channel_end_ts = channel.end_timestamp_ms;

    // Validate inputs - NULL pointer checks
    if (!forward_high || !forward_low || !forward_close) {
        labels.duration_valid = false;
        labels.direction_valid = false;
        labels.next_channel_valid = false;
        labels.break_scan_valid = false;
        return labels;
    }

    // Validate channel
    if (channel.slope == 0.0 && channel.intercept == 0.0) {
        labels.duration_valid = false;
        labels.direction_valid = false;
        labels.next_channel_valid = false;
        labels.break_scan_valid = false;
        return labels;
    }

    // Check forward data availability
    if (n_forward <= 0) {
        labels.duration_valid = false;
        labels.direction_valid = false;
        labels.next_channel_valid = false;
        labels.break_scan_valid = false;
        return labels;
    }

    // Validate max_scan bounds
    if (max_scan <= 0 || max_scan > n_forward) {
        max_scan = n_forward;
    }

    // Scan for break
    BreakResult result = scan_for_break(
        channel, forward_high, forward_low, forward_close,
        n_forward, max_scan
    );

    // Map BreakResult to ChannelLabels
    if (!result.break_detected) {
        // No break found - consolidation scenario
        labels.duration_bars = result.scan_bars_used;
        labels.break_direction = 1;  // Default UP
        labels.next_channel_direction = 1;  // SIDEWAYS
        labels.permanent_break = false;
        labels.bars_to_first_break = result.scan_bars_used;
        labels.break_magnitude = 0.0;
        labels.returned_to_channel = false;
        labels.bounces_after_return = 0;
        labels.round_trip_bounces = 0;
        labels.channel_continued = true;
        labels.permanent_break_direction = -1;
        labels.permanent_break_magnitude = 0.0;
        labels.bars_to_permanent_break = -1;
        labels.duration_to_permanent = -1;
        labels.avg_bars_outside = 0.0;
        labels.total_bars_outside = 0;
        labels.durability_score = 0.0;

        // Set RSI labels to defaults for no-break scenario
        labels.rsi_at_first_break = 50.0;
        labels.rsi_at_permanent_break = 50.0;
        labels.rsi_at_channel_end = 50.0;
        labels.rsi_overbought_at_break = false;
        labels.rsi_oversold_at_break = false;
        labels.rsi_divergence_at_break = 0;
        labels.rsi_trend_in_channel = 0;
        labels.rsi_range_in_channel = 0.0;

        // Validity: break scan succeeded, but no break found
        // This is VALID data - represents consolidation/no-break scenario
        labels.duration_valid = true;
        labels.direction_valid = true;  // Valid label - channel consolidates without break
        labels.next_channel_valid = false;
        labels.break_scan_valid = true;
        return labels;
    }

    // Break detected - populate labels
    labels.duration_bars = result.break_bar;
    labels.break_direction = result.break_direction;
    labels.permanent_break = result.is_permanent;

    // FIRST break fields
    labels.bars_to_first_break = (result.first_touch_bar >= 0)
        ? result.first_touch_bar
        : result.break_bar;
    labels.break_magnitude = result.break_magnitude;
    labels.returned_to_channel = result.is_false_break;
    labels.channel_continued = result.is_false_break;

    // Bounces after return
    if (result.is_false_break) {
        labels.bounces_after_return = std::max(0, result.false_break_count - 1);
    } else {
        labels.bounces_after_return = result.false_break_count;
    }
    labels.round_trip_bounces = result.round_trip_bounces;

    // PERMANENT break fields
    labels.permanent_break_direction = result.permanent_break_direction;
    labels.permanent_break_magnitude = result.permanent_break_magnitude;
    labels.bars_to_permanent_break = result.permanent_break_bar;
    labels.duration_to_permanent = result.permanent_break_bar;

    // Exit dynamics
    double avg_bars_outside = 0.0;
    int total_bars_outside = 0;
    if (!result.all_exit_events.empty()) {
        for (const auto& evt : result.all_exit_events) {
            if (evt.returned) {
                total_bars_outside += evt.bars_outside;
            }
        }
        if (result.exits_returned_count > 0) {
            avg_bars_outside = static_cast<double>(total_bars_outside) /
                              result.exits_returned_count;
        }
    }
    labels.avg_bars_outside = avg_bars_outside;
    labels.total_bars_outside = total_bars_outside;
    labels.durability_score = calculate_durability_score(
        result.false_break_count,
        static_cast<int>(result.all_exit_events.size()),
        avg_bars_outside
    );

    // Exit verification
    labels.first_break_returned = result.is_false_break;
    labels.exit_return_rate = result.exit_return_rate;
    labels.exits_returned_count = result.exits_returned_count;
    labels.exits_stayed_out_count = result.exits_stayed_out_count;
    labels.scan_timed_out = result.scan_timed_out;
    labels.bars_verified_permanent = result.bars_verified_permanent;

    // Individual exit events
    for (const auto& evt : result.all_exit_events) {
        labels.exit_bars.push_back(evt.bar_index);
        labels.exit_magnitudes.push_back(evt.magnitude);
        labels.exit_durations.push_back(evt.returned ? evt.bars_outside : -1);
        labels.exit_types.push_back(evt.exit_type);
        labels.exit_returned.push_back(evt.returned);
    }

    // Next channel direction
    if (next_channel_direction >= 0) {
        labels.next_channel_direction = next_channel_direction;
        labels.next_channel_valid = true;
    } else {
        labels.next_channel_direction = 1;  // SIDEWAYS (unknown)
        labels.next_channel_valid = false;
    }

    // Compute RSI labels if full close prices provided
    if (full_close_prices != nullptr && full_close_size > 0) {
        compute_rsi_labels(
            full_close_prices,
            full_close_size,
            channel_end_idx,
            labels.bars_to_first_break,
            labels.bars_to_permanent_break,
            channel.window_size,
            labels
        );
    } else {
        // Set default RSI values if no close prices available
        labels.rsi_at_first_break = 50.0;
        labels.rsi_at_permanent_break = 50.0;
        labels.rsi_at_channel_end = 50.0;
        labels.rsi_overbought_at_break = false;
        labels.rsi_oversold_at_break = false;
        labels.rsi_divergence_at_break = 0;
        labels.rsi_trend_in_channel = 0;
        labels.rsi_range_in_channel = 0.0;
    }

    // Validity flags
    labels.duration_valid = true;
    labels.direction_valid = true;
    labels.break_scan_valid = true;

    return labels;
}

// =============================================================================
// NEXT CHANNEL DETECTION
// =============================================================================

double LabelGenerator::channel_sort_key(const Channel& channel) const {
    // Sort by (bounce_count, r_squared) for ranking
    // Scale r_squared to [0,1] range and add to bounce_count
    return channel.bounce_count + channel.r_squared;
}

void LabelGenerator::compute_next_channel_labels(
    const Channel* channels,
    int n_channels,
    int current_idx,
    ChannelLabels& labels_out
) const {
    // Initialize defaults
    labels_out.best_next_channel_direction = -1;
    labels_out.best_next_channel_bars_away = -1;
    labels_out.best_next_channel_duration = -1;
    labels_out.best_next_channel_r_squared = 0.0;
    labels_out.best_next_channel_bounce_count = 0;
    labels_out.shortest_next_channel_direction = -1;
    labels_out.shortest_next_channel_bars_away = -1;
    labels_out.shortest_next_channel_duration = -1;
    labels_out.small_channels_before_best = 0;

    if (!channels || current_idx < 0 || current_idx >= n_channels) {
        return;
    }

    const Channel& current = channels[current_idx];

    // Collect up to 2 next channels
    struct NextChannelInfo {
        int position;           // 0 or 1
        const Channel* channel;
        int bars_away;
        int duration;
    };
    std::vector<NextChannelInfo> next_channels;

    for (int offset = 1; offset <= 2; ++offset) {
        int idx = current_idx + offset;
        if (idx < n_channels) {
            const Channel& next = channels[idx];
            NextChannelInfo info;
            info.position = offset - 1;
            info.channel = &next;
            info.bars_away = next.start_idx - current.end_idx;
            info.duration = next.end_idx - next.start_idx;
            next_channels.push_back(info);
        }
    }

    if (next_channels.empty()) {
        return;
    }

    // Find best channel (highest sort key)
    int best_position = 0;
    const NextChannelInfo* best_info = &next_channels[0];
    double best_key = channel_sort_key(*best_info->channel);

    for (size_t i = 1; i < next_channels.size(); ++i) {
        double key = channel_sort_key(*next_channels[i].channel);
        if (key > best_key) {
            best_key = key;
            best_info = &next_channels[i];
            best_position = i;
        }
    }

    // Populate best channel fields
    labels_out.best_next_channel_direction = static_cast<int>(best_info->channel->direction);
    labels_out.best_next_channel_bars_away = best_info->bars_away;
    labels_out.best_next_channel_duration = best_info->duration;
    labels_out.best_next_channel_r_squared = best_info->channel->r_squared;
    labels_out.best_next_channel_bounce_count = best_info->channel->bounce_count;
    labels_out.small_channels_before_best = best_position;

    // Find shortest channel
    const NextChannelInfo* shortest_info = &next_channels[0];
    int shortest_duration = shortest_info->duration;

    for (size_t i = 1; i < next_channels.size(); ++i) {
        if (next_channels[i].duration < shortest_duration) {
            shortest_duration = next_channels[i].duration;
            shortest_info = &next_channels[i];
        }
    }

    labels_out.shortest_next_channel_direction = static_cast<int>(shortest_info->channel->direction);
    labels_out.shortest_next_channel_bars_away = shortest_info->bars_away;
    labels_out.shortest_next_channel_duration = shortest_info->duration;
}

// =============================================================================
// CROSS-CORRELATION
// =============================================================================

CrossCorrelationLabels LabelGenerator::compute_cross_correlation_labels(
    const ChannelLabels& tsla_labels,
    const ChannelLabels& spy_labels
) const {
    CrossCorrelationLabels cross;

    // Check validity
    if (!tsla_labels.break_scan_valid || !spy_labels.break_scan_valid) {
        cross.cross_valid = false;
        return cross;
    }

    cross.cross_valid = true;

    // 1. FIRST break direction alignment
    cross.break_direction_aligned =
        (tsla_labels.break_direction == spy_labels.break_direction);

    // 2. Who broke first?
    cross.tsla_broke_first =
        (tsla_labels.bars_to_first_break < spy_labels.bars_to_first_break);
    cross.spy_broke_first =
        (spy_labels.bars_to_first_break < tsla_labels.bars_to_first_break);
    cross.break_lag_bars = std::abs(
        tsla_labels.bars_to_first_break - spy_labels.bars_to_first_break
    );

    // 3. Magnitude spread
    cross.magnitude_spread =
        tsla_labels.break_magnitude - spy_labels.break_magnitude;

    // 4. PERMANENT break alignment
    bool tsla_has_perm = (tsla_labels.permanent_break_direction >= 0);
    bool spy_has_perm = (spy_labels.permanent_break_direction >= 0);

    if (tsla_has_perm && spy_has_perm) {
        cross.permanent_cross_valid = true;
        cross.permanent_direction_aligned =
            (tsla_labels.permanent_break_direction == spy_labels.permanent_break_direction);
        cross.tsla_permanent_first =
            (tsla_labels.bars_to_permanent_break < spy_labels.bars_to_permanent_break);
        cross.spy_permanent_first =
            (spy_labels.bars_to_permanent_break < tsla_labels.bars_to_permanent_break);
        cross.permanent_break_lag_bars = std::abs(
            tsla_labels.bars_to_permanent_break - spy_labels.bars_to_permanent_break
        );
        cross.permanent_magnitude_spread =
            tsla_labels.permanent_break_magnitude - spy_labels.permanent_break_magnitude;
    }

    // 5. Return pattern alignment
    cross.both_returned =
        (tsla_labels.returned_to_channel && spy_labels.returned_to_channel);
    cross.both_permanent =
        (tsla_labels.permanent_break && spy_labels.permanent_break);
    cross.return_pattern_aligned =
        (tsla_labels.returned_to_channel == spy_labels.returned_to_channel);
    cross.continuation_aligned =
        (tsla_labels.channel_continued == spy_labels.channel_continued);

    // 6. Permanent duration dynamics
    if (tsla_has_perm && spy_has_perm) {
        cross.permanent_dynamics_valid = true;
        cross.permanent_duration_lag_bars =
            tsla_labels.bars_to_permanent_break - spy_labels.bars_to_permanent_break;
        cross.permanent_duration_spread = std::abs(cross.permanent_duration_lag_bars);
    }

    // 7. Durability comparison
    cross.durability_spread =
        tsla_labels.durability_score - spy_labels.durability_score;
    cross.avg_bars_outside_spread =
        tsla_labels.avg_bars_outside - spy_labels.avg_bars_outside;
    cross.total_bars_outside_spread =
        tsla_labels.total_bars_outside - spy_labels.total_bars_outside;

    constexpr double HIGH_DURABILITY = 0.7;
    constexpr double LOW_DURABILITY = 0.3;

    cross.both_high_durability =
        (tsla_labels.durability_score > HIGH_DURABILITY &&
         spy_labels.durability_score > HIGH_DURABILITY);
    cross.both_low_durability =
        (tsla_labels.durability_score < LOW_DURABILITY &&
         spy_labels.durability_score < LOW_DURABILITY);
    cross.durability_aligned =
        (cross.both_high_durability || cross.both_low_durability);
    cross.tsla_more_durable =
        (tsla_labels.durability_score > spy_labels.durability_score);
    cross.spy_more_durable =
        (spy_labels.durability_score > tsla_labels.durability_score);

    // 8. Exit verification cross-correlation
    cross.exit_verification_valid = true;
    cross.exit_return_rate_spread =
        tsla_labels.exit_return_rate - spy_labels.exit_return_rate;

    constexpr double ALIGNED_THRESHOLD = 0.2;
    cross.exit_return_rate_aligned =
        (std::abs(cross.exit_return_rate_spread) < ALIGNED_THRESHOLD);

    cross.tsla_more_resilient = (tsla_labels.exit_return_rate > spy_labels.exit_return_rate);
    cross.spy_more_resilient = (spy_labels.exit_return_rate > tsla_labels.exit_return_rate);

    cross.exits_returned_spread =
        tsla_labels.exits_returned_count - spy_labels.exits_returned_count;
    cross.exits_stayed_out_spread =
        tsla_labels.exits_stayed_out_count - spy_labels.exits_stayed_out_count;
    cross.total_exits_spread =
        (tsla_labels.exits_returned_count + tsla_labels.exits_stayed_out_count) -
        (spy_labels.exits_returned_count + spy_labels.exits_stayed_out_count);

    cross.both_scan_timed_out =
        (tsla_labels.scan_timed_out && spy_labels.scan_timed_out);
    cross.scan_timeout_aligned =
        (tsla_labels.scan_timed_out == spy_labels.scan_timed_out);

    cross.bars_verified_spread =
        tsla_labels.bars_verified_permanent - spy_labels.bars_verified_permanent;

    cross.both_first_returned_then_permanent =
        (tsla_labels.first_break_returned && tsla_has_perm &&
         spy_labels.first_break_returned && spy_has_perm);
    cross.both_never_returned =
        (!tsla_labels.returned_to_channel && !spy_labels.returned_to_channel);

    // 9. Next channel alignment
    if (tsla_labels.next_channel_valid && spy_labels.next_channel_valid) {
        cross.next_channel_direction_aligned =
            (tsla_labels.next_channel_direction == spy_labels.next_channel_direction);
    }

    // 10. RSI cross-correlation
    cross.rsi_aligned_at_break =
        (std::abs(tsla_labels.rsi_at_first_break - spy_labels.rsi_at_first_break) < 10.0);
    cross.rsi_divergence_aligned =
        (tsla_labels.rsi_divergence_at_break == spy_labels.rsi_divergence_at_break);
    cross.tsla_rsi_higher_at_break =
        (tsla_labels.rsi_at_first_break > spy_labels.rsi_at_first_break);
    cross.rsi_spread_at_break =
        tsla_labels.rsi_at_first_break - spy_labels.rsi_at_first_break;

    // Predictive patterns
    cross.overbought_predicts_down_break =
        (tsla_labels.rsi_overbought_at_break && tsla_labels.break_direction == 0);
    cross.oversold_predicts_up_break =
        (tsla_labels.rsi_oversold_at_break && tsla_labels.break_direction == 1);

    return cross;
}

} // namespace v15
