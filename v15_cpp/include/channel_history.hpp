#pragma once

#include <cstdint>

namespace v15 {

/**
 * ChannelHistoryEntry - Stores historical channel information
 *
 * Used to track the last 5 channels for each asset (TSLA/SPY) per timeframe.
 * Contains channel metrics and exit behavior data from ChannelLabels.
 *
 * This struct is in its own header to avoid circular dependencies between
 * feature_extractor.hpp and scanner.hpp.
 */
struct ChannelHistoryEntry {
    int64_t end_timestamp = 0;      // For chronological ordering (milliseconds)
    double duration = 50.0;         // Duration of the channel in bars
    double slope = 0.0;             // Channel slope
    int direction = 1;              // 0=bear, 1=sideways, 2=bull
    int break_direction = 0;        // -1=down, 0=no break, 1=up
    double r_squared = 0.0;         // Channel quality (R-squared)
    double bounce_count = 0.0;      // Number of bounces off channel bounds

    // Exit metrics from ChannelLabels
    double exit_count = 0.0;        // Number of exits from channel
    double avg_exit_magnitude = 0.0; // Average magnitude of exits
    double avg_bars_outside = 0.0;  // Average bars spent outside channel
    double exit_return_rate = 0.0;  // Rate at which exits return to channel
    double durability_score = 0.0;  // Channel durability (resistance to breaks)
    double false_break_count = 0.0; // Number of false breaks (bounces after return)

    ChannelHistoryEntry() = default;
};

} // namespace v15
