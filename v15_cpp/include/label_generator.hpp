#pragma once

#include "channel.hpp"
#include "labels.hpp"
#include "types.hpp"
#include <vector>
#include <cstdint>
#include <memory>

namespace v15 {

// =============================================================================
// BREAK SCANNER STRUCTURES
// =============================================================================

/**
 * Record of a channel exit event.
 * Tracks when price exits channel bounds, whether it returned, and metrics.
 */
struct ExitEvent {
    int bar_index;          // Bar index when exit occurred (relative to scan start)
    int exit_type;          // 0=lower breach, 1=upper breach
    double exit_price;      // Price at exit
    double magnitude;       // How far outside bounds (in std devs)
    bool returned;          // Whether price returned to channel
    int bars_outside;       // Bars spent outside before return
    int return_bar;         // Bar index where price returned (-1 if didn't return)

    ExitEvent()
        : bar_index(0)
        , exit_type(0)
        , exit_price(0.0)
        , magnitude(0.0)
        , returned(false)
        , bars_outside(0)
        , return_bar(-1)
    {}
};

/**
 * Complete result of forward break scanning.
 * Tracks first break, permanent break, exit events, and durability metrics.
 */
struct BreakResult {
    // Core break info (FIRST break)
    bool break_detected;
    int break_bar;              // Bar index of FIRST break
    int break_direction;        // 0=DOWN, 1=UP
    double break_magnitude;     // Distance from bound (std devs)
    double break_price;         // Price at break

    // First touch tracking (visual break point before magnitude check)
    int first_touch_bar;
    int first_touch_direction;  // 0=DOWN, 1=UP
    double first_touch_price;

    // Permanence of FIRST break
    bool is_permanent;          // First break never returned
    bool is_false_break;        // First break returned to channel
    int bars_until_return;      // Bars outside before return
    int return_bar;             // Bar where returned (-1 if didn't)

    // PERMANENT/lasting break tracking
    int permanent_break_direction;  // -1=none, 0=DOWN, 1=UP
    int permanent_break_bar;        // Bar index of permanent break (-1 if none)
    double permanent_break_magnitude;

    // Exit event tracking
    std::vector<ExitEvent> all_exit_events;
    int false_break_count;      // Count of exits that returned
    double false_break_rate;    // Ratio of returns to total exits

    // Metadata
    int scan_bars_used;
    double projected_upper;
    double projected_lower;

    // Exit verification
    bool scan_timed_out;        // Hit max_scan without confirming permanence
    int bars_verified_permanent;
    int exits_returned_count;
    int exits_stayed_out_count;
    double exit_return_rate;

    // Round-trip bounces
    int round_trip_bounces;     // Count of upper->lower or lower->upper alternations

    BreakResult()
        : break_detected(false)
        , break_bar(-1)
        , break_direction(0)
        , break_magnitude(0.0)
        , break_price(0.0)
        , first_touch_bar(-1)
        , first_touch_direction(0)
        , first_touch_price(0.0)
        , is_permanent(false)
        , is_false_break(false)
        , bars_until_return(0)
        , return_bar(-1)
        , permanent_break_direction(-1)
        , permanent_break_bar(-1)
        , permanent_break_magnitude(0.0)
        , false_break_count(0)
        , false_break_rate(0.0)
        , scan_bars_used(0)
        , projected_upper(0.0)
        , projected_lower(0.0)
        , scan_timed_out(false)
        , bars_verified_permanent(0)
        , exits_returned_count(0)
        , exits_stayed_out_count(0)
        , exit_return_rate(0.0)
        , round_trip_bounces(0)
    {}
};

// =============================================================================
// LABEL GENERATOR CLASS
// =============================================================================

/**
 * Label generation for channel break prediction.
 *
 * Implements sophisticated forward scanning to detect:
 * - First break outside channel bounds (using high/low for exits)
 * - Break magnitude in standard deviations
 * - Permanent vs false breaks (return tracking)
 * - Multiple exit events and durability scoring
 * - Next channel detection and direction classification
 * - RSI labels at key break moments
 * - Cross-correlation between TSLA and SPY
 *
 * Thread-safe for parallel processing with OpenMP.
 */
class LabelGenerator {
public:
    /**
     * Configuration for label generation.
     */
    struct Config {
        double min_break_magnitude;     // Minimum magnitude to count as break (default 0.5)
        int return_threshold_bars;      // Bars to confirm permanence (default 5)
        int rsi_period;                 // RSI period (default 14)
        double rsi_overbought;          // RSI overbought threshold (default 70)
        double rsi_oversold;            // RSI oversold threshold (default 30)
        double rsi_trend_threshold;     // RSI trend detection threshold (default 5.0)

        Config()
            : min_break_magnitude(0.5)
            , return_threshold_bars(5)
            , rsi_period(14)
            , rsi_overbought(70.0)
            , rsi_oversold(30.0)
            , rsi_trend_threshold(5.0)
        {}
    };

    /**
     * Constructor.
     */
    explicit LabelGenerator(const Config& config = Config());

    /**
     * Scan forward from channel end to detect break.
     *
     * Projects channel bounds forward and detects first bar where price
     * breaks outside bounds (using close prices). Tracks return behavior
     * and all exit events for durability analysis.
     *
     * Args:
     *   channel: Channel to scan from
     *   forward_high: High prices after channel end
     *   forward_low: Low prices after channel end
     *   forward_close: Close prices after channel end
     *   n_forward: Number of forward bars available
     *   max_scan: Maximum bars to scan
     *
     * Returns:
     *   BreakResult with complete break analysis
     */
    BreakResult scan_for_break(
        const Channel& channel,
        const double* forward_high,
        const double* forward_low,
        const double* forward_close,
        int n_forward,
        int max_scan
    ) const;

    /**
     * Generate labels for a single channel using forward scanning.
     *
     * This is the primary labeling method that:
     * - Scans forward to find first break
     * - Detects permanent vs false breaks
     * - Computes break magnitude and timing
     * - Tracks exit events for durability
     * - Computes RSI labels at break moments
     *
     * Args:
     *   channel: Channel to label
     *   channel_end_idx: Index where channel ends (in timeframe data)
     *   forward_high: High prices after channel end
     *   forward_low: Low prices after channel end
     *   forward_close: Close prices after channel end
     *   n_forward: Number of forward bars available
     *   max_scan: Maximum bars to scan forward
     *   next_channel_direction: Optional next channel direction (-1 if unknown)
     *   full_close_prices: Full close price array for RSI computation (optional, can be nullptr)
     *   full_close_size: Size of full_close_prices array
     *
     * Returns:
     *   ChannelLabels with all break scan fields populated
     */
    ChannelLabels generate_labels_forward_scan(
        const Channel& channel,
        int channel_end_idx,
        const double* forward_high,
        const double* forward_low,
        const double* forward_close,
        int n_forward,
        int max_scan,
        int next_channel_direction = -1,
        const double* full_close_prices = nullptr,
        int full_close_size = 0
    ) const;

    /**
     * Compute RSI labels at key break moments.
     *
     * Calculates RSI at:
     * - Channel end
     * - First break
     * - Permanent break
     *
     * And detects:
     * - Overbought/oversold conditions
     * - Divergence patterns
     * - RSI trend in channel
     *
     * Args:
     *   close_prices: Close prices (includes lookback for RSI calc)
     *   close_array_size: Size of close_prices array
     *   channel_end_idx: Index of channel end in close_prices
     *   first_break_bar: Bars after channel end when first break occurred
     *   permanent_break_bar: Bars after channel end when permanent break occurred (-1 if none)
     *   channel_window: Window size of channel
     *   rsi_out: Output struct for RSI labels (fields without prefix)
     *
     * Note: Caller adds tsla_/spy_ prefix to field names.
     */
    void compute_rsi_labels(
        const double* close_prices,
        int close_array_size,
        int channel_end_idx,
        int first_break_bar,
        int permanent_break_bar,
        int channel_window,
        ChannelLabels& labels_out
    ) const;

    /**
     * Compute next channel labels by looking ahead at next 2 channels.
     *
     * Finds:
     * - Best next channel (highest bounce_count, r_squared tiebreaker)
     * - Shortest next channel (minimum duration)
     * - Small channels before best
     *
     * Args:
     *   channels: Array of channels for this timeframe/window
     *   n_channels: Number of channels in array
     *   current_idx: Index of current channel
     *   labels_out: ChannelLabels to populate with next channel fields
     */
    void compute_next_channel_labels(
        const Channel* channels,
        int n_channels,
        int current_idx,
        ChannelLabels& labels_out
    ) const;

    /**
     * Compute cross-correlation labels comparing TSLA and SPY.
     *
     * Analyzes alignment patterns:
     * - Break direction alignment
     * - Lead/lag relationships
     * - Magnitude spreads
     * - Return pattern alignment
     * - Durability comparison
     *
     * Args:
     *   tsla_labels: TSLA channel labels
     *   spy_labels: SPY channel labels
     *
     * Returns:
     *   CrossCorrelationLabels with comparison metrics
     */
    CrossCorrelationLabels compute_cross_correlation_labels(
        const ChannelLabels& tsla_labels,
        const ChannelLabels& spy_labels
    ) const;

private:
    Config config_;

    // Helper: Project channel bounds forward
    void project_bounds(
        const Channel& channel,
        int bars_forward,
        double& center_out,
        double& upper_out,
        double& lower_out
    ) const;

    // Helper: Calculate durability score from exit events
    double calculate_durability_score(
        int false_break_count,
        int total_exits,
        double avg_bars_outside
    ) const;

    // Helper: Compute RSI for price array
    void compute_rsi(
        const double* prices,
        int n_prices,
        int period,
        double* rsi_out
    ) const;

    // Helper: Channel sort key for ranking (bounce_count, r_squared tiebreaker)
    double channel_sort_key(const Channel& channel) const;
};

} // namespace v15
