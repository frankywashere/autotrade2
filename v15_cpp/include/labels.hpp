#pragma once

#include "types.hpp"
#include <vector>
#include <cstdint>

namespace v15 {

// =============================================================================
// CHANNEL LABELS STRUCTURE
// =============================================================================

/**
 * Labels for a single channel at a specific window size.
 *
 * ARCHITECTURE NOTE: Samples are ONLY created at channel end positions.
 * Therefore, "sample position" and "channel end" are equivalent.
 *
 * BAR TIMING SEMANTICS:
 *   All bar-based fields use 0-based counting from the NEXT bar after sample.
 *   - Value of 0 means event happens on NEXT bar (sample+1)
 *   - Value of N means event happens N bars after sample (sample+N+1)
 *
 * Memory layout optimized for:
 *   - Prediction targets grouped first (hot data)
 *   - TSLA features grouped together
 *   - SPY features grouped together
 *   - Validity flags at end (cold data)
 */
struct ChannelLabels {
    // =========================================================================
    // PREDICTION TARGETS (hot data - accessed most frequently)
    // =========================================================================
    int duration_bars;              // PRIMARY: Bars until channel breaks
    int next_channel_direction;     // SECONDARY: Direction after break (0=BEAR, 1=SIDEWAYS, 2=BULL)
    bool permanent_break;           // SECONDARY: Whether break sticks

    // Metadata
    Timeframe timeframe;            // Which timeframe these labels are for

    // =========================================================================
    // TSLA BREAK SCAN FEATURES (for model input, NOT prediction targets)
    // =========================================================================

    // FIRST break dynamics
    int break_direction;            // Which bound breached FIRST (0=DOWN, 1=UP)
    double break_magnitude;         // How far outside on FIRST break (std devs)
    int bars_to_first_break;        // When FIRST break occurred
    bool returned_to_channel;       // Did price return after first break
    int bounces_after_return;       // False breaks before final exit
    int round_trip_bounces;         // Alternating upper/lower exits
    bool channel_continued;         // Did pattern resume after return

    // PERMANENT break dynamics
    int permanent_break_direction;  // Direction of FINAL break (-1=none, 0=DOWN, 1=UP)
    double permanent_break_magnitude;  // Magnitude of permanent break (std devs)
    int bars_to_permanent_break;    // When permanent break occurred (-1 if none)

    // Exit dynamics
    int duration_to_permanent;      // Bars until PERMANENT break (-1 if none)
    double avg_bars_outside;        // Average duration of each exit
    int total_bars_outside;         // Sum of all exit durations
    double durability_score;        // Weighted resilience score (0.0-1.5+)

    // Exit verification tracking
    bool first_break_returned;      // Alias for returned_to_channel
    double exit_return_rate;        // exits_returned / total_exits
    int exits_returned_count;       // Count of exits that returned
    int exits_stayed_out_count;     // Count of exits that stayed out
    bool scan_timed_out;            // Did scan hit TF_MAX_SCAN?
    int bars_verified_permanent;    // Bars outside before declaring permanent

    // Individual exit events (use vectors for variable-length data)
    std::vector<int> exit_bars;         // Bar indices when each exit occurred
    std::vector<double> exit_magnitudes; // Magnitude of each exit (std devs)
    std::vector<int> exit_durations;     // Bars outside before return (-1 if no return)
    std::vector<int> exit_types;         // 0=lower breach, 1=upper breach
    std::vector<bool> exit_returned;     // Whether each exit returned

    // =========================================================================
    // SPY BREAK SCAN FEATURES (mirrored for cross-asset analysis)
    // =========================================================================

    // SPY FIRST break dynamics
    int spy_break_direction;
    double spy_break_magnitude;
    int spy_bars_to_first_break;
    bool spy_returned_to_channel;
    int spy_bounces_after_return;
    int spy_round_trip_bounces;
    bool spy_channel_continued;

    // SPY PERMANENT break dynamics
    int spy_permanent_break_direction;
    double spy_permanent_break_magnitude;
    int spy_bars_to_permanent_break;

    // SPY Exit dynamics
    int spy_duration_to_permanent;
    double spy_avg_bars_outside;
    int spy_total_bars_outside;
    double spy_durability_score;

    // SPY Exit verification
    bool spy_first_break_returned;
    double spy_exit_return_rate;
    int spy_exits_returned_count;
    int spy_exits_stayed_out_count;
    bool spy_scan_timed_out;
    int spy_bars_verified_permanent;

    // SPY Individual exit events
    std::vector<int> spy_exit_bars;
    std::vector<double> spy_exit_magnitudes;
    std::vector<int> spy_exit_durations;
    std::vector<int> spy_exit_types;
    std::vector<bool> spy_exit_returned;

    // =========================================================================
    // SOURCE CHANNEL PARAMETERS (for visualization reconstruction)
    // =========================================================================

    // TSLA source channel
    double source_channel_slope;
    double source_channel_intercept;
    double source_channel_std_dev;
    double source_channel_r_squared;
    int source_channel_direction;
    int source_channel_bounce_count;
    int64_t source_channel_start_ts;    // Timestamp in ms since epoch
    int64_t source_channel_end_ts;

    // SPY source channel
    double spy_source_channel_slope;
    double spy_source_channel_intercept;
    double spy_source_channel_std_dev;
    double spy_source_channel_r_squared;
    int spy_source_channel_direction;
    int spy_source_channel_bounce_count;
    int64_t spy_source_channel_start_ts;
    int64_t spy_source_channel_end_ts;

    // =========================================================================
    // NEXT CHANNEL LABELS
    // =========================================================================

    // Best next channel (ranked by bounce_count)
    int best_next_channel_direction;
    int best_next_channel_bars_away;
    int best_next_channel_duration;
    double best_next_channel_r_squared;
    int best_next_channel_bounce_count;

    // Shortest next channel (by duration)
    int shortest_next_channel_direction;
    int shortest_next_channel_bars_away;
    int shortest_next_channel_duration;

    // Pattern info
    int small_channels_before_best;

    // SPY next channel labels
    int spy_best_next_channel_direction;
    int spy_best_next_channel_bars_away;
    int spy_best_next_channel_duration;
    double spy_best_next_channel_r_squared;
    int spy_best_next_channel_bounce_count;
    int spy_shortest_next_channel_direction;
    int spy_shortest_next_channel_bars_away;
    int spy_shortest_next_channel_duration;
    int spy_small_channels_before_best;

    // =========================================================================
    // RSI LABELS
    // =========================================================================

    // TSLA RSI
    double rsi_at_first_break;
    double rsi_at_permanent_break;
    double rsi_at_channel_end;
    bool rsi_overbought_at_break;
    bool rsi_oversold_at_break;
    int rsi_divergence_at_break;    // -1=bearish, 0=none, 1=bullish
    int rsi_trend_in_channel;       // -1=falling, 0=flat, 1=rising
    double rsi_range_in_channel;

    // SPY RSI
    double spy_rsi_at_first_break;
    double spy_rsi_at_permanent_break;
    double spy_rsi_at_channel_end;
    bool spy_rsi_overbought_at_break;
    bool spy_rsi_oversold_at_break;
    int spy_rsi_divergence_at_break;
    int spy_rsi_trend_in_channel;
    double spy_rsi_range_in_channel;

    // =========================================================================
    // VALIDITY FLAGS (cold data - accessed less frequently)
    // =========================================================================
    bool duration_valid;
    bool direction_valid;
    bool next_channel_valid;
    bool break_scan_valid;

    // =========================================================================
    // CONSTRUCTORS
    // =========================================================================

    ChannelLabels()
        : duration_bars(0)
        , next_channel_direction(1)  // SIDEWAYS default
        , permanent_break(false)
        , timeframe(Timeframe::INVALID)
        , break_direction(0)
        , break_magnitude(0.0)
        , bars_to_first_break(0)
        , returned_to_channel(false)
        , bounces_after_return(0)
        , round_trip_bounces(0)
        , channel_continued(false)
        , permanent_break_direction(-1)
        , permanent_break_magnitude(0.0)
        , bars_to_permanent_break(-1)
        , duration_to_permanent(-1)
        , avg_bars_outside(0.0)
        , total_bars_outside(0)
        , durability_score(0.0)
        , first_break_returned(false)
        , exit_return_rate(0.0)
        , exits_returned_count(0)
        , exits_stayed_out_count(0)
        , scan_timed_out(false)
        , bars_verified_permanent(0)
        , spy_break_direction(0)
        , spy_break_magnitude(0.0)
        , spy_bars_to_first_break(0)
        , spy_returned_to_channel(false)
        , spy_bounces_after_return(0)
        , spy_round_trip_bounces(0)
        , spy_channel_continued(false)
        , spy_permanent_break_direction(-1)
        , spy_permanent_break_magnitude(0.0)
        , spy_bars_to_permanent_break(-1)
        , spy_duration_to_permanent(-1)
        , spy_avg_bars_outside(0.0)
        , spy_total_bars_outside(0)
        , spy_durability_score(0.0)
        , spy_first_break_returned(false)
        , spy_exit_return_rate(0.0)
        , spy_exits_returned_count(0)
        , spy_exits_stayed_out_count(0)
        , spy_scan_timed_out(false)
        , spy_bars_verified_permanent(0)
        , source_channel_slope(0.0)
        , source_channel_intercept(0.0)
        , source_channel_std_dev(0.0)
        , source_channel_r_squared(0.0)
        , source_channel_direction(-1)
        , source_channel_bounce_count(0)
        , source_channel_start_ts(0)
        , source_channel_end_ts(0)
        , spy_source_channel_slope(0.0)
        , spy_source_channel_intercept(0.0)
        , spy_source_channel_std_dev(0.0)
        , spy_source_channel_r_squared(0.0)
        , spy_source_channel_direction(-1)
        , spy_source_channel_bounce_count(0)
        , spy_source_channel_start_ts(0)
        , spy_source_channel_end_ts(0)
        , best_next_channel_direction(-1)
        , best_next_channel_bars_away(-1)
        , best_next_channel_duration(-1)
        , best_next_channel_r_squared(0.0)
        , best_next_channel_bounce_count(0)
        , shortest_next_channel_direction(-1)
        , shortest_next_channel_bars_away(-1)
        , shortest_next_channel_duration(-1)
        , small_channels_before_best(0)
        , spy_best_next_channel_direction(-1)
        , spy_best_next_channel_bars_away(-1)
        , spy_best_next_channel_duration(-1)
        , spy_best_next_channel_r_squared(0.0)
        , spy_best_next_channel_bounce_count(0)
        , spy_shortest_next_channel_direction(-1)
        , spy_shortest_next_channel_bars_away(-1)
        , spy_shortest_next_channel_duration(-1)
        , spy_small_channels_before_best(0)
        , rsi_at_first_break(50.0)
        , rsi_at_permanent_break(50.0)
        , rsi_at_channel_end(50.0)
        , rsi_overbought_at_break(false)
        , rsi_oversold_at_break(false)
        , rsi_divergence_at_break(0)
        , rsi_trend_in_channel(0)
        , rsi_range_in_channel(0.0)
        , spy_rsi_at_first_break(50.0)
        , spy_rsi_at_permanent_break(50.0)
        , spy_rsi_at_channel_end(50.0)
        , spy_rsi_overbought_at_break(false)
        , spy_rsi_oversold_at_break(false)
        , spy_rsi_divergence_at_break(0)
        , spy_rsi_trend_in_channel(0)
        , spy_rsi_range_in_channel(0.0)
        , duration_valid(false)
        , direction_valid(false)
        , next_channel_valid(false)
        , break_scan_valid(false)
    {}
};

// =============================================================================
// CROSS-CORRELATION LABELS
// =============================================================================

/**
 * Labels comparing TSLA and SPY channel break behavior.
 * Captures lead/lag relationships and correlated movements.
 */
struct CrossCorrelationLabels {
    // FIRST break cross-correlation
    bool break_direction_aligned;
    bool tsla_broke_first;
    bool spy_broke_first;
    int break_lag_bars;
    double magnitude_spread;

    // PERMANENT break cross-correlation
    bool permanent_direction_aligned;
    bool tsla_permanent_first;
    bool spy_permanent_first;
    int permanent_break_lag_bars;
    double permanent_magnitude_spread;

    // Direction transition patterns
    bool tsla_direction_diverged;
    bool spy_direction_diverged;
    bool both_direction_diverged;
    bool direction_divergence_aligned;

    // Return/permanence patterns
    bool both_returned;
    bool both_permanent;
    bool return_pattern_aligned;
    bool continuation_aligned;

    // Exit dynamics cross-correlation
    int permanent_duration_lag_bars;
    int permanent_duration_spread;

    // Resilience comparison
    double durability_spread;
    double avg_bars_outside_spread;
    int total_bars_outside_spread;

    // Resilience alignment flags
    bool both_high_durability;
    bool both_low_durability;
    bool durability_aligned;
    bool tsla_more_durable;
    bool spy_more_durable;

    // Exit verification cross-correlation
    double exit_return_rate_spread;
    bool exit_return_rate_aligned;
    bool tsla_more_resilient;
    bool spy_more_resilient;
    int exits_returned_spread;
    int exits_stayed_out_spread;
    int total_exits_spread;
    bool both_scan_timed_out;
    bool scan_timeout_aligned;
    int bars_verified_spread;
    bool both_first_returned_then_permanent;
    bool both_never_returned;

    // Individual exit event cross-correlation
    double exit_timing_correlation;
    double exit_timing_lag_mean;
    double exit_direction_agreement;
    int exit_count_spread;
    int lead_lag_exits;
    double exit_magnitude_correlation;
    double mean_magnitude_spread;
    double exit_duration_correlation;
    double mean_duration_spread;
    int simultaneous_exit_count;

    // Next channel cross-correlation
    bool divergence_predicts_reversal;
    bool permanent_break_matches_next;
    bool next_channel_direction_aligned;
    bool next_channel_quality_aligned;
    int best_next_channel_tsla_vs_spy;  // -1=SPY better, 0=equal, 1=TSLA better

    // RSI cross-correlation
    bool rsi_aligned_at_break;
    bool rsi_divergence_aligned;
    bool tsla_rsi_higher_at_break;
    double rsi_spread_at_break;
    bool overbought_predicts_down_break;
    bool oversold_predicts_up_break;

    // Validity flags
    bool cross_valid;
    bool permanent_cross_valid;
    bool permanent_dynamics_valid;
    bool exit_verification_valid;
    bool exit_cross_correlation_valid;

    // Default constructor
    CrossCorrelationLabels()
        : break_direction_aligned(false)
        , tsla_broke_first(false)
        , spy_broke_first(false)
        , break_lag_bars(0)
        , magnitude_spread(0.0)
        , permanent_direction_aligned(false)
        , tsla_permanent_first(false)
        , spy_permanent_first(false)
        , permanent_break_lag_bars(0)
        , permanent_magnitude_spread(0.0)
        , tsla_direction_diverged(false)
        , spy_direction_diverged(false)
        , both_direction_diverged(false)
        , direction_divergence_aligned(false)
        , both_returned(false)
        , both_permanent(false)
        , return_pattern_aligned(false)
        , continuation_aligned(false)
        , permanent_duration_lag_bars(0)
        , permanent_duration_spread(0)
        , durability_spread(0.0)
        , avg_bars_outside_spread(0.0)
        , total_bars_outside_spread(0)
        , both_high_durability(false)
        , both_low_durability(false)
        , durability_aligned(false)
        , tsla_more_durable(false)
        , spy_more_durable(false)
        , exit_return_rate_spread(0.0)
        , exit_return_rate_aligned(false)
        , tsla_more_resilient(false)
        , spy_more_resilient(false)
        , exits_returned_spread(0)
        , exits_stayed_out_spread(0)
        , total_exits_spread(0)
        , both_scan_timed_out(false)
        , scan_timeout_aligned(false)
        , bars_verified_spread(0)
        , both_first_returned_then_permanent(false)
        , both_never_returned(false)
        , exit_timing_correlation(0.0)
        , exit_timing_lag_mean(0.0)
        , exit_direction_agreement(0.0)
        , exit_count_spread(0)
        , lead_lag_exits(0)
        , exit_magnitude_correlation(0.0)
        , mean_magnitude_spread(0.0)
        , exit_duration_correlation(0.0)
        , mean_duration_spread(0.0)
        , simultaneous_exit_count(0)
        , divergence_predicts_reversal(false)
        , permanent_break_matches_next(false)
        , next_channel_direction_aligned(false)
        , next_channel_quality_aligned(false)
        , best_next_channel_tsla_vs_spy(0)
        , rsi_aligned_at_break(false)
        , rsi_divergence_aligned(false)
        , tsla_rsi_higher_at_break(false)
        , rsi_spread_at_break(0.0)
        , overbought_predicts_down_break(false)
        , oversold_predicts_up_break(false)
        , cross_valid(false)
        , permanent_cross_valid(false)
        , permanent_dynamics_valid(false)
        , exit_verification_valid(false)
        , exit_cross_correlation_valid(false)
    {}
};

} // namespace v15
