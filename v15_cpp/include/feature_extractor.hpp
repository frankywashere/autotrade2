#pragma once

#include "types.hpp"
#include "channel.hpp"
#include "indicators.hpp"
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

namespace v15 {

/**
 * ChannelHistoryEntry - Stores historical channel information
 *
 * Used to track the last 5 channels for each asset (TSLA/SPY) per timeframe.
 * Contains channel metrics and exit behavior data from ChannelLabels.
 */
struct ChannelHistoryEntry {
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

/**
 * FeatureExtractor - Complete multi-timeframe feature extraction system
 *
 * Extracts 14,190 features from TSLA, SPY, and VIX data:
 * - Multi-timeframe resampling (5min → all 10 TFs)
 * - TSLA price features (58 features per TF) = 580
 * - Technical indicators (59 features per TF) = 590
 * - SPY features (117 features per TF) = 1,170
 * - VIX features (25 features per TF) = 250
 * - Cross-asset features (59 features per TF) = 590
 * - Channel features per window (116 features per window * 8 windows * 10 TFs) = 9,280
 * - Window scores and aggregated features (50 per TF) = 500
 * - Channel history features (67 per TF) = 670
 * - Event features (30 TF-independent) = 30
 * - Bar metadata (30 features, 3 per TF) = 30
 *
 * Total: 14,190 features
 */
class FeatureExtractor {
public:
    /**
     * Extract all 14,190 features for a given timestamp
     *
     * @param tsla_5min Base 5-min TSLA OHLCV data
     * @param spy_5min Base 5-min SPY OHLCV data
     * @param vix_5min Base 5-min VIX OHLCV data
     * @param timestamp Current timestamp for event features
     * @param source_bar_count Number of 5min bars from start (for partial bar calculation)
     * @param include_bar_metadata Include 30 bar metadata features
     * @return Map of feature names to values
     */
    static std::unordered_map<std::string, double> extract_all_features(
        const std::vector<OHLCV>& tsla_5min,
        const std::vector<OHLCV>& spy_5min,
        const std::vector<OHLCV>& vix_5min,
        int64_t timestamp,
        int source_bar_count = -1,
        bool include_bar_metadata = true
    );

    /**
     * Get expected total feature count (14,190)
     */
    static int get_total_feature_count() { return 14190; }

    /**
     * Get all feature names in consistent order
     */
    static std::vector<std::string> get_all_feature_names();

    /**
     * Get the 67 channel history feature names (without TF prefix)
     */
    static std::vector<std::string> get_channel_history_feature_names();

    /**
     * Extract 67 channel history features from TSLA and SPY channel histories
     *
     * @param tsla_history Last 5 channel entries for TSLA
     * @param spy_history Last 5 channel entries for SPY
     * @return Map of 67 feature names to values
     */
    static std::unordered_map<std::string, double> extract_channel_history_features(
        const std::vector<ChannelHistoryEntry>& tsla_history,
        const std::vector<ChannelHistoryEntry>& spy_history
    );

private:
    // Resampling utilities
    struct ResampleMetadata {
        double bar_completion_pct;
        int bars_in_partial;
        int expected_bars;
        bool is_partial;
        int total_bars;
        int source_bars;
    };

    static std::pair<std::vector<OHLCV>, ResampleMetadata> resample_to_tf(
        const std::vector<OHLCV>& data_5min,
        Timeframe target_tf,
        int source_bar_count
    );

    // TSLA Price Features (58 per TF)
    static std::unordered_map<std::string, double> extract_tsla_price_features(
        const std::vector<OHLCV>& tsla_data
    );

    // SPY Features (117 per TF)
    static std::unordered_map<std::string, double> extract_spy_features(
        const std::vector<OHLCV>& spy_data
    );

    // VIX Features (25 per TF)
    static std::unordered_map<std::string, double> extract_vix_features(
        const std::vector<OHLCV>& vix_data
    );

    // Cross-Asset Features (59 per TF)
    static std::unordered_map<std::string, double> extract_cross_asset_features(
        const std::vector<OHLCV>& tsla_data,
        const std::vector<OHLCV>& spy_data,
        const std::vector<OHLCV>& vix_data,
        double tsla_rsi_14 = 50.0,
        double spy_rsi_14 = 50.0,
        double position_in_channel = 0.5,
        double spy_position_in_channel = 0.5,
        double vix_level = 20.0
    );

    // Channel Features (58 per window)
    static std::unordered_map<std::string, double> extract_channel_features(
        const Channel& channel,
        const std::vector<OHLCV>& data
    );

    // Default channel features (58 features) - returned for invalid channels
    static std::unordered_map<std::string, double> get_default_channel_features();

    // SPY Channel Features (58 per window)
    static std::unordered_map<std::string, double> extract_spy_channel_features(
        const Channel& channel,
        const std::vector<OHLCV>& spy_data,
        int window
    );

    // Window Score Features (50 per TF)
    static std::unordered_map<std::string, double> extract_window_score_features(
        const std::unordered_map<int, std::shared_ptr<Channel>>& channels_by_window,
        int best_window
    );

    // Channel Correlation Features (50 per TF)
    // Cross-correlation features between TSLA and SPY channel features at the same timeframe
    static std::unordered_map<std::string, double> extract_channel_correlation_features(
        const std::unordered_map<std::string, double>& tsla_channel_features,
        const std::unordered_map<std::string, double>& spy_channel_features
    );

    // Event Features (30 TF-independent)
    static std::unordered_map<std::string, double> extract_event_features(
        int64_t timestamp,
        const std::vector<OHLCV>& tsla_data
    );

    // Bar Metadata Features (30 total: 3 per TF)
    static std::unordered_map<std::string, double> extract_bar_metadata_features(
        const std::unordered_map<Timeframe, ResampleMetadata>& metadata_by_tf
    );

    // Prefix utilities
    static void prefix_features(
        std::unordered_map<std::string, double>& features,
        const std::string& prefix
    );

    static std::unordered_map<std::string, double> prefix_features_copy(
        const std::unordered_map<std::string, double>& features,
        const std::string& prefix
    );

    // Helper functions
    static double safe_divide(double numerator, double denominator, double default_val = 0.0);
    static double safe_float(double value, double default_val = 0.0);
    static double get_last_valid(const std::vector<double>& arr, double default_val = 0.0);
    static double pct_change(double current, double previous, double default_val = 0.0);

    // Indicator helpers (reuse from TechnicalIndicators)
    static std::vector<double> sma(const std::vector<double>& values, int period);
    static std::vector<double> ema(const std::vector<double>& values, int period);
    static std::vector<double> rsi(const std::vector<double>& values, int period);
    static std::vector<double> atr(
        const std::vector<double>& high,
        const std::vector<double>& low,
        const std::vector<double>& close,
        int period
    );

    // Extract OHLCV components
    static void extract_ohlcv_arrays(
        const std::vector<OHLCV>& data,
        std::vector<double>& open,
        std::vector<double>& high,
        std::vector<double>& low,
        std::vector<double>& close,
        std::vector<double>& volume
    );

    // Sanitize features (ensure all are finite)
    static void sanitize_features(std::unordered_map<std::string, double>& features);

    // Calculate correlations
    static double calculate_correlation(
        const std::vector<double>& series1,
        const std::vector<double>& series2,
        int window,
        double default_val = 0.0
    );

    // Calculate beta
    static double calculate_beta(
        const std::vector<double>& asset_returns,
        const std::vector<double>& market_returns,
        int window
    );

    // Channel correlation helper functions
    static double safe_spread(double tsla_val, double spy_val);
    static double safe_ratio(double tsla_val, double spy_val, double default_val = 1.0);
    static double safe_aligned(double tsla_val, double spy_val, double threshold = 0.5);
    static double direction_aligned(double tsla_val, double spy_val);

    // =============================================================================
    // Channel History Helper Functions
    // =============================================================================

    // Extract features from a single asset's channel history (40 features)
    static std::unordered_map<std::string, double> extract_single_history_features(
        const std::vector<ChannelHistoryEntry>& history,
        const std::string& prefix
    );

    // Encode direction sequence to single value (0-4)
    // 0=all bear, 1=mostly bear, 2=mixed, 3=mostly bull, 4=all bull
    static int encode_direction_sequence(const std::vector<int>& directions);

    // Encode break direction sequence to single value (0-4)
    // 0=all down, 1=mostly down, 2=mixed, 3=mostly up, 4=all up
    static int encode_break_sequence(const std::vector<int>& break_directions);

    // Calculate alternating score (0-1, how often direction changes)
    static double calculate_alternating_score(const std::vector<int>& directions);

    // Calculate trend in a sequence of values (normalized slope)
    static double calculate_trend(const std::vector<double>& values);

    // Calculate momentum from direction sequence
    static double calculate_momentum(const std::vector<int>& directions);

    // Calculate regime shift score from history
    static double calculate_regime_shift(const std::vector<ChannelHistoryEntry>& history);

    // Safe statistical helpers for channel history
    static double safe_mean(const std::vector<double>& values, double default_val = 0.0);
    static double safe_std(const std::vector<double>& values, double default_val = 0.0);
    static double safe_min(const std::vector<double>& values, double default_val = 0.0);
    static double safe_max(const std::vector<double>& values, double default_val = 0.0);
};

} // namespace v15
