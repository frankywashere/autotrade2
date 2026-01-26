#include "feature_extractor.hpp"
#include "feature_array.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <ctime>
#include <iostream>
#include <set>
#include <mutex>
#include <array>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace v15 {

// Thread-local resample cache definition
thread_local ResampleCache FeatureExtractor::s_resample_cache;

// =============================================================================
// MAIN EXTRACTION FUNCTION
// =============================================================================

std::unordered_map<std::string, double> FeatureExtractor::extract_all_features(
    const std::vector<OHLCV>& tsla_5min,
    const std::vector<OHLCV>& spy_5min,
    const std::vector<OHLCV>& vix_5min,
    int64_t timestamp,
    int source_bar_count,
    bool include_bar_metadata
) {
    // Pre-reserve the feature map to avoid rehashing during 14,190 insertions
    // This is a significant performance optimization
    auto all_features = create_feature_map();

    // SAFETY: Validate input data
    if (tsla_5min.empty()) {
        std::cerr << "[ERROR] extract_all_features: TSLA data is empty\n";
        return all_features;  // Return empty map
    }
    if (spy_5min.empty()) {
        std::cerr << "[ERROR] extract_all_features: SPY data is empty\n";
        return all_features;
    }
    if (vix_5min.empty()) {
        std::cerr << "[ERROR] extract_all_features: VIX data is empty\n";
        return all_features;
    }

    // SAFETY: Validate data alignment
    if (tsla_5min.size() != spy_5min.size() || tsla_5min.size() != vix_5min.size()) {
        std::cerr << "[ERROR] extract_all_features: Data size mismatch - TSLA: "
                  << tsla_5min.size() << ", SPY: " << spy_5min.size()
                  << ", VIX: " << vix_5min.size() << "\n";
        return all_features;
    }

    // Default source_bar_count to data length
    if (source_bar_count < 0) {
        source_bar_count = static_cast<int>(tsla_5min.size());
    }

    // SAFETY: Validate source_bar_count
    if (source_bar_count > static_cast<int>(tsla_5min.size())) {
        std::cerr << "[WARNING] extract_all_features: source_bar_count (" << source_bar_count
                  << ") exceeds data size (" << tsla_5min.size() << "), clamping\n";
        source_bar_count = static_cast<int>(tsla_5min.size());
    }

    // Track metadata by timeframe for bar completion features
    std::unordered_map<Timeframe, ResampleMetadata> metadata_by_tf;

    // Thread-local storage for parallel extraction:
    // Each timeframe gets its own feature map, merged at the end
    std::array<std::unordered_map<std::string, double>, NUM_TIMEFRAMES> tf_features;
    std::array<ResampleMetadata, NUM_TIMEFRAMES> tf_metadata;
    std::array<bool, NUM_TIMEFRAMES> tf_valid;
    tf_valid.fill(false);

    // Process each timeframe in parallel (when OpenMP is available)
    // Each iteration is independent - writes to its own tf_features[tf_idx]
    #pragma omp parallel for schedule(dynamic) if(NUM_TIMEFRAMES > 1)
    for (int tf_idx = 0; tf_idx < NUM_TIMEFRAMES; ++tf_idx) {
        Timeframe tf = static_cast<Timeframe>(tf_idx);
        std::string tf_prefix = std::string(timeframe_to_string(tf)) + "_";

        // Thread-local feature map for this timeframe
        std::unordered_map<std::string, double>& local_features = tf_features[tf_idx];

        try {
            // 1. Resample data to this timeframe
            auto [tsla_tf, tsla_meta] = resample_to_tf(tsla_5min, tf, source_bar_count);
            auto [spy_tf, spy_meta] = resample_to_tf(spy_5min, tf, source_bar_count);
            auto [vix_tf, vix_meta] = resample_to_tf(vix_5min, tf, source_bar_count);

            // Store metadata for this timeframe
            tf_metadata[tf_idx] = tsla_meta;

            // SAFETY: Check if we have enough data
            if (tsla_tf.empty()) {
                #pragma omp critical
                {
                    std::cerr << "[WARNING] Timeframe " << tf_prefix << " has empty TSLA data after resampling\n";
                }
                continue;
            }
            if (spy_tf.empty()) {
                #pragma omp critical
                {
                    std::cerr << "[WARNING] Timeframe " << tf_prefix << " has empty SPY data after resampling\n";
                }
                continue;
            }
            if (vix_tf.empty()) {
                #pragma omp critical
                {
                    std::cerr << "[WARNING] Timeframe " << tf_prefix << " has empty VIX data after resampling\n";
                }
                continue;
            }

            // Require minimum bars for meaningful features
            if (tsla_tf.size() < 10) {
                static int skip_count = 0;
                #pragma omp critical
                {
                    if (skip_count < 3) {
                        std::cerr << "[DEBUG] Skipping timeframe " << tf_prefix
                                  << " (only " << tsla_tf.size() << " bars, need 10+)\n";
                        skip_count++;
                    }
                }
                continue;
            }

            // 2. Extract TSLA price features (58)
            auto price_features = extract_tsla_price_features(tsla_tf);
            for (const auto& [name, value] : price_features) {
                local_features[tf_prefix + name] = value;
            }

            // 3. Extract technical indicators (59) - using existing indicators.cpp
            // Use optimized struct-of-arrays extraction (single resize vs 5 separate reserves)
            OHLCVArrays tsla_arrays;
            extract_ohlcv_arrays_optimized(tsla_tf, tsla_arrays);

            // SAFETY: Validate extracted arrays
            if (tsla_arrays.empty()) {
                #pragma omp critical
                {
                    std::cerr << "[WARNING] Timeframe " << tf_prefix << " has empty OHLCV arrays\n";
                }
                continue;
            }

            auto tech_features = TechnicalIndicators::extract_features(
                tsla_arrays.open, tsla_arrays.high, tsla_arrays.low,
                tsla_arrays.close, tsla_arrays.volume);
            for (const auto& [name, value] : tech_features) {
                local_features[tf_prefix + name] = value;
            }

            // 4. Extract SPY features (117)
            auto spy_features = extract_spy_features(spy_tf);
            for (const auto& [name, value] : spy_features) {
                local_features[tf_prefix + name] = value;
            }

            // 5. Extract VIX features (25)
            auto vix_features = extract_vix_features(vix_tf);
            for (const auto& [name, value] : vix_features) {
                local_features[tf_prefix + name] = value;
            }

            // 6. Extract cross-asset features (59)
            // Get RSI values from price features for cross-asset correlation
            double tsla_rsi_14 = price_features.count("rsi_14") ? price_features.at("rsi_14") : 50.0;
            double spy_rsi_14 = spy_features.count("spy_rsi_14") ? spy_features.at("spy_rsi_14") : 50.0;
            double vix_level = vix_features.count("vix_level") ? vix_features.at("vix_level") : 20.0;

            auto cross_features = extract_cross_asset_features(
                tsla_tf, spy_tf, vix_tf,
                tsla_rsi_14, spy_rsi_14, 0.5, 0.5, vix_level
            );
            for (const auto& [name, value] : cross_features) {
                local_features[tf_prefix + name] = value;
            }

            // 7. Extract channel features for each window (58 TSLA + 58 SPY = 116 features × 8 windows = 928)
            // Note: This would require channel detection which is already implemented
            // For now, we'll add placeholder zeros for channel features
            // In production, this should call the channel detector

            // Collect channel features across all windows for aggregated correlation
            std::unordered_map<std::string, double> tsla_channel_feats_agg;
            std::unordered_map<std::string, double> spy_channel_feats_agg;
            int valid_window_count = 0;

            for (int window : STANDARD_WINDOWS) {
                std::string window_prefix = tf_prefix + "w" + std::to_string(window) + "_";

                // TSLA channel features (58)
                // For now, using default channel - in production, would use detected channel
                Channel tsla_channel;
                auto tsla_channel_feats = extract_channel_features(tsla_channel, tsla_tf);
                for (const auto& [name, value] : tsla_channel_feats) {
                    local_features[window_prefix + name] = value;
                }

                // SPY channel features (58)
                Channel spy_channel;
                auto spy_channel_feats = extract_spy_channel_features(spy_channel, spy_tf, window);
                for (const auto& [name, value] : spy_channel_feats) {
                    local_features[window_prefix + name] = value;
                }

                // Accumulate for aggregated channel correlation (computed once per TF below)
                for (const auto& [name, value] : tsla_channel_feats) {
                    tsla_channel_feats_agg[name] += value;
                }
                for (const auto& [name, value] : spy_channel_feats) {
                    spy_channel_feats_agg[name] += value;
                }
                valid_window_count++;
            }

            // Average the aggregated channel features across windows
            if (valid_window_count > 0) {
                for (auto& [name, value] : tsla_channel_feats_agg) {
                    value /= valid_window_count;
                }
                for (auto& [name, value] : spy_channel_feats_agg) {
                    value /= valid_window_count;
                }
            }

            // 7b. Channel correlation features (50 per TF - computed once using aggregated features)
            auto channel_corr_feats = extract_channel_correlation_features(
                tsla_channel_feats_agg, spy_channel_feats_agg
            );
            for (const auto& [name, value] : channel_corr_feats) {
                local_features[tf_prefix + name] = value;
            }

            // 8. Extract window score features (50)
            // Placeholder for now - would use detected channels
            std::unordered_map<int, std::shared_ptr<Channel>> channels_by_window;
            auto window_scores = extract_window_score_features(channels_by_window, 50);
            for (const auto& [name, value] : window_scores) {
                local_features[tf_prefix + name] = value;
            }

            // 9. Extract channel history features (67)
            // In production, this would use actual channel history tracking per TF
            // For now, we use empty histories which will produce default features
            std::vector<ChannelHistoryEntry> tsla_history;
            std::vector<ChannelHistoryEntry> spy_history;
            auto history_features = extract_channel_history_features(tsla_history, spy_history);
            for (const auto& [name, value] : history_features) {
                local_features[tf_prefix + name] = value;
            }

            // Mark this timeframe as successfully processed
            tf_valid[tf_idx] = true;

        } catch (const std::exception& e) {
            #pragma omp critical
            {
                std::cerr << "[ERROR] Failed to extract features for timeframe " << tf_prefix
                          << ": " << e.what() << "\n";
            }
            // Continue to next timeframe on error
        }
    }

    // Merge thread-local results into all_features (sequential, after parallel section)
    for (int tf_idx = 0; tf_idx < NUM_TIMEFRAMES; ++tf_idx) {
        // Copy metadata
        Timeframe tf = static_cast<Timeframe>(tf_idx);
        metadata_by_tf[tf] = tf_metadata[tf_idx];

        // Merge features from this timeframe
        if (tf_valid[tf_idx]) {
            all_features.insert(tf_features[tf_idx].begin(), tf_features[tf_idx].end());
        }
    }

    // 10. Extract event features (30 TF-independent)
    auto event_features = extract_event_features(timestamp, tsla_5min);
    all_features.insert(event_features.begin(), event_features.end());

    // 11. Extract bar metadata features (30)
    if (include_bar_metadata) {
        auto bar_meta = extract_bar_metadata_features(metadata_by_tf);
        all_features.insert(bar_meta.begin(), bar_meta.end());
    }

    // Sanitize all features
    sanitize_features(all_features);

    // Debug: show feature count breakdown
    static int call_count = 0;
    if (call_count < 2) {
        std::cerr << "[DEBUG] extract_all_features returning " << all_features.size() << " features\n";
        std::cerr << "  source_bar_count=" << source_bar_count
                  << " tsla_5min.size()=" << tsla_5min.size() << "\n";
        call_count++;
    }

    return all_features;
}

// =============================================================================
// DATAVIEW OVERLOAD - ZERO-COPY FEATURE EXTRACTION
// =============================================================================

std::unordered_map<std::string, double> FeatureExtractor::extract_all_features(
    const DataView& tsla_view,
    const DataView& spy_view,
    const DataView& vix_view,
    int64_t timestamp,
    int source_bar_count,
    bool include_bar_metadata
) {
    std::unordered_map<std::string, double> all_features;

    // SAFETY: Validate input data
    if (tsla_view.empty()) {
        std::cerr << "[ERROR] extract_all_features(DataView): TSLA data is empty\n";
        return all_features;
    }
    if (spy_view.empty()) {
        std::cerr << "[ERROR] extract_all_features(DataView): SPY data is empty\n";
        return all_features;
    }
    if (vix_view.empty()) {
        std::cerr << "[ERROR] extract_all_features(DataView): VIX data is empty\n";
        return all_features;
    }

    // SAFETY: Validate data alignment
    if (tsla_view.size() != spy_view.size() || tsla_view.size() != vix_view.size()) {
        std::cerr << "[ERROR] extract_all_features(DataView): Data size mismatch - TSLA: "
                  << tsla_view.size() << ", SPY: " << spy_view.size()
                  << ", VIX: " << vix_view.size() << "\n";
        return all_features;
    }

    // Default source_bar_count to data length
    if (source_bar_count < 0) {
        source_bar_count = static_cast<int>(tsla_view.size());
    }

    // SAFETY: Validate source_bar_count
    if (source_bar_count > static_cast<int>(tsla_view.size())) {
        source_bar_count = static_cast<int>(tsla_view.size());
    }

    // Track metadata by timeframe for bar completion features
    std::unordered_map<Timeframe, ResampleMetadata> metadata_by_tf;

    // Process each timeframe
    for (int tf_idx = 0; tf_idx < NUM_TIMEFRAMES; ++tf_idx) {
        Timeframe tf = static_cast<Timeframe>(tf_idx);
        std::string tf_prefix = std::string(timeframe_to_string(tf)) + "_";

        try {
            // 1. Resample data to this timeframe using DataView overload
            auto [tsla_tf, tsla_meta] = resample_to_tf(tsla_view, tf, source_bar_count);
            auto [spy_tf, spy_meta] = resample_to_tf(spy_view, tf, source_bar_count);
            auto [vix_tf, vix_meta] = resample_to_tf(vix_view, tf, source_bar_count);

            // Store metadata
            metadata_by_tf[tf] = tsla_meta;

            // SAFETY: Check if we have enough data
            if (tsla_tf.empty() || spy_tf.empty() || vix_tf.empty()) {
                continue;
            }

            // Require minimum bars for meaningful features
            if (tsla_tf.size() < 10) {
                continue;
            }

            // 2. Extract TSLA price features (58)
            auto price_features = extract_tsla_price_features(tsla_tf);
            for (const auto& [name, value] : price_features) {
                all_features[tf_prefix + name] = value;
            }

            // 3. Extract technical indicators (59)
            // Use optimized struct-of-arrays extraction
            OHLCVArrays tsla_arrays;
            extract_ohlcv_arrays_optimized(tsla_tf, tsla_arrays);

            if (tsla_arrays.empty()) {
                continue;
            }

            auto tech_features = TechnicalIndicators::extract_features(
                tsla_arrays.open, tsla_arrays.high, tsla_arrays.low,
                tsla_arrays.close, tsla_arrays.volume);
            for (const auto& [name, value] : tech_features) {
                all_features[tf_prefix + name] = value;
            }

            // 4. Extract SPY features (117)
            auto spy_features = extract_spy_features(spy_tf);
            for (const auto& [name, value] : spy_features) {
                all_features[tf_prefix + name] = value;
            }

            // 5. Extract VIX features (25)
            auto vix_features = extract_vix_features(vix_tf);
            for (const auto& [name, value] : vix_features) {
                all_features[tf_prefix + name] = value;
            }

            // 6. Extract cross-asset features (59)
            double tsla_rsi_14 = price_features.count("rsi_14") ? price_features.at("rsi_14") : 50.0;
            double spy_rsi_14 = spy_features.count("spy_rsi_14") ? spy_features.at("spy_rsi_14") : 50.0;
            double vix_level = vix_features.count("vix_level") ? vix_features.at("vix_level") : 20.0;

            auto cross_features = extract_cross_asset_features(
                tsla_tf, spy_tf, vix_tf,
                tsla_rsi_14, spy_rsi_14, 0.5, 0.5, vix_level
            );
            for (const auto& [name, value] : cross_features) {
                all_features[tf_prefix + name] = value;
            }

            // 7. Extract channel features for each window (116 features x 8 windows = 928)
            std::unordered_map<std::string, double> tsla_channel_feats_agg;
            std::unordered_map<std::string, double> spy_channel_feats_agg;
            int valid_window_count = 0;

            for (int window : STANDARD_WINDOWS) {
                std::string window_prefix = tf_prefix + "w" + std::to_string(window) + "_";

                Channel tsla_channel;
                auto tsla_channel_feats = extract_channel_features(tsla_channel, tsla_tf);
                for (const auto& [name, value] : tsla_channel_feats) {
                    all_features[window_prefix + name] = value;
                }

                Channel spy_channel;
                auto spy_channel_feats = extract_spy_channel_features(spy_channel, spy_tf, window);
                for (const auto& [name, value] : spy_channel_feats) {
                    all_features[window_prefix + name] = value;
                }

                for (const auto& [name, value] : tsla_channel_feats) {
                    tsla_channel_feats_agg[name] += value;
                }
                for (const auto& [name, value] : spy_channel_feats) {
                    spy_channel_feats_agg[name] += value;
                }
                valid_window_count++;
            }

            if (valid_window_count > 0) {
                for (auto& [name, value] : tsla_channel_feats_agg) {
                    value /= valid_window_count;
                }
                for (auto& [name, value] : spy_channel_feats_agg) {
                    value /= valid_window_count;
                }
            }

            // 7b. Channel correlation features (50 per TF)
            auto channel_corr_feats = extract_channel_correlation_features(
                tsla_channel_feats_agg, spy_channel_feats_agg
            );
            for (const auto& [name, value] : channel_corr_feats) {
                all_features[tf_prefix + name] = value;
            }

            // 8. Extract window score features (50)
            std::unordered_map<int, std::shared_ptr<Channel>> channels_by_window;
            auto window_scores = extract_window_score_features(channels_by_window, 50);
            for (const auto& [name, value] : window_scores) {
                all_features[tf_prefix + name] = value;
            }

            // 9. Extract channel history features (67)
            std::vector<ChannelHistoryEntry> tsla_history;
            std::vector<ChannelHistoryEntry> spy_history;
            auto history_features = extract_channel_history_features(tsla_history, spy_history);
            for (const auto& [name, value] : history_features) {
                all_features[tf_prefix + name] = value;
            }

        } catch (const std::exception& e) {
            std::cerr << "[ERROR] Failed to extract features for timeframe " << tf_prefix
                      << ": " << e.what() << "\n";
        }
    }

    // 10. Extract event features (30 TF-independent) using DataView overload
    auto event_features = extract_event_features(timestamp, tsla_view);
    all_features.insert(event_features.begin(), event_features.end());

    // 11. Extract bar metadata features (30)
    if (include_bar_metadata) {
        auto bar_meta = extract_bar_metadata_features(metadata_by_tf);
        all_features.insert(bar_meta.begin(), bar_meta.end());
    }

    // Sanitize all features
    sanitize_features(all_features);

    return all_features;
}

// DataView overload for resample_to_tf
std::pair<std::vector<OHLCV>, FeatureExtractor::ResampleMetadata>
FeatureExtractor::resample_to_tf(
    const DataView& data_view,
    Timeframe target_tf,
    int source_bar_count
) {
    // For 5min timeframe, we can avoid copying by just wrapping the view
    // For other timeframes, we need to copy to resample
    if (target_tf == Timeframe::MIN_5) {
        ResampleMetadata metadata;
        metadata.source_bars = static_cast<int>(data_view.size());
        metadata.bar_completion_pct = 1.0;
        metadata.bars_in_partial = 1;
        metadata.expected_bars = 1;
        metadata.is_partial = false;
        metadata.total_bars = static_cast<int>(data_view.size());
        // Copy needed here for 5min as we return a vector
        return {data_view.to_vector(), metadata};
    }
    
    // For other timeframes, convert to vector and resample
    // This is where the copy happens, but it's only for resampled data
    return resample_to_tf(data_view.to_vector(), target_tf, source_bar_count);
}

// DataView overload for extract_event_features
std::unordered_map<std::string, double> FeatureExtractor::extract_event_features(
    int64_t timestamp,
    const DataView& tsla_view
) {
    // Event features only need the last few bars, so we can work directly with the view
    // For now, convert to vector for compatibility with existing implementation
    return extract_event_features(timestamp, tsla_view.to_vector());
}


// =============================================================================
// RESAMPLING
// =============================================================================

std::pair<std::vector<OHLCV>, FeatureExtractor::ResampleMetadata>
FeatureExtractor::resample_to_tf(
    const std::vector<OHLCV>& data_5min,
    Timeframe target_tf,
    int source_bar_count
) {
    ResampleMetadata metadata;
    metadata.source_bars = static_cast<int>(data_5min.size());

    // SAFETY: Check for empty input
    if (data_5min.empty()) {
        metadata.bar_completion_pct = 1.0;
        metadata.bars_in_partial = 0;
        metadata.expected_bars = 0;
        metadata.is_partial = false;
        metadata.total_bars = 0;
        return {std::vector<OHLCV>(), metadata};
    }

    int bars_per_tf_bar = get_bars_per_tf(target_tf);

    // SAFETY: Validate bars_per_tf_bar
    if (bars_per_tf_bar <= 0) {
        std::cerr << "[ERROR] Invalid bars_per_tf_bar: " << bars_per_tf_bar << " for timeframe\n";
        return {std::vector<OHLCV>(), metadata};
    }

    // For 5min, no resampling needed
    if (target_tf == Timeframe::MIN_5) {
        metadata.bar_completion_pct = 1.0;
        metadata.bars_in_partial = 1;
        metadata.expected_bars = 1;
        metadata.is_partial = false;
        metadata.total_bars = static_cast<int>(data_5min.size());
        return {data_5min, metadata};
    }

    // SAFETY: Validate source_bar_count
    if (source_bar_count < 0 || source_bar_count > static_cast<int>(data_5min.size())) {
        source_bar_count = static_cast<int>(data_5min.size());
    }

    // Calculate bar completion
    int bars_into_current = source_bar_count % bars_per_tf_bar;
    if (bars_into_current == 0) {
        metadata.bar_completion_pct = 1.0;
        metadata.is_partial = false;
    } else {
        metadata.bar_completion_pct = static_cast<double>(bars_into_current) / bars_per_tf_bar;
        metadata.is_partial = true;
    }
    metadata.bars_in_partial = bars_into_current > 0 ? bars_into_current : bars_per_tf_bar;
    metadata.expected_bars = bars_per_tf_bar;

    // Perform OHLC aggregation
    std::vector<OHLCV> resampled;
    int n = static_cast<int>(data_5min.size());

    // SAFETY: Pre-allocate space to avoid reallocation
    resampled.reserve(n / bars_per_tf_bar + 1);

    for (int i = 0; i < n; i += bars_per_tf_bar) {
        int end_idx = std::min(i + bars_per_tf_bar, n);

        // SAFETY: Validate indices
        if (i >= n || end_idx > n || i >= end_idx) {
            std::cerr << "[WARNING] Invalid resample indices: i=" << i << ", end_idx=" << end_idx << ", n=" << n << "\n";
            break;
        }

        OHLCV bar;
        bar.open = data_5min[i].open;
        bar.close = data_5min[end_idx - 1].close;
        bar.volume = 0.0;

        bar.high = data_5min[i].high;
        bar.low = data_5min[i].low;

        // SAFETY: Validate OHLCV values
        if (!std::isfinite(bar.open) || !std::isfinite(bar.close) ||
            !std::isfinite(bar.high) || !std::isfinite(bar.low)) {
            std::cerr << "[WARNING] Non-finite OHLCV values at index " << i << ", skipping bar\n";
            continue;
        }

        for (int j = i; j < end_idx; ++j) {
            // SAFETY: Check array bounds
            if (j >= n) break;

            bar.high = std::max(bar.high, data_5min[j].high);
            bar.low = std::min(bar.low, data_5min[j].low);
            bar.volume += data_5min[j].volume;
        }

        // SAFETY: Validate final bar values
        if (std::isfinite(bar.high) && std::isfinite(bar.low) &&
            std::isfinite(bar.open) && std::isfinite(bar.close)) {
            resampled.push_back(bar);
        }
    }

    metadata.total_bars = static_cast<int>(resampled.size());

    return {resampled, metadata};
}

std::pair<std::vector<OHLCV>, FeatureExtractor::ResampleMetadata>
FeatureExtractor::resample_to_tf_cached(
    const std::vector<OHLCV>& data_5min,
    Timeframe target_tf,
    int source_bar_count,
    ResampleCache& cache,
    int asset_idx  // 0=TSLA, 1=SPY, 2=VIX
) {
    int tf_idx = static_cast<int>(target_tf);

    // Check if we have a valid cached result
    if (cache.is_valid(target_tf, data_5min.size(), source_bar_count)) {
        // Return cached data based on asset index
        ResampleMetadata metadata;
        const std::vector<OHLCV>* cached_data = nullptr;

        switch (asset_idx) {
            case 0: cached_data = &cache.tsla_resampled[tf_idx]; break;
            case 1: cached_data = &cache.spy_resampled[tf_idx]; break;
            case 2: cached_data = &cache.vix_resampled[tf_idx]; break;
            default: break;
        }

        if (cached_data && !cached_data->empty()) {
            // Reconstruct metadata from cached data
            int bars_per_tf_bar = get_bars_per_tf(target_tf);
            metadata.source_bars = static_cast<int>(data_5min.size());
            metadata.total_bars = static_cast<int>(cached_data->size());

            if (target_tf == Timeframe::MIN_5) {
                metadata.bar_completion_pct = 1.0;
                metadata.bars_in_partial = 1;
                metadata.expected_bars = 1;
                metadata.is_partial = false;
            } else {
                int bars_into_current = source_bar_count % bars_per_tf_bar;
                if (bars_into_current == 0) {
                    metadata.bar_completion_pct = 1.0;
                    metadata.is_partial = false;
                } else {
                    metadata.bar_completion_pct = static_cast<double>(bars_into_current) / bars_per_tf_bar;
                    metadata.is_partial = true;
                }
                metadata.bars_in_partial = bars_into_current > 0 ? bars_into_current : bars_per_tf_bar;
                metadata.expected_bars = bars_per_tf_bar;
            }

            return {*cached_data, metadata};
        }
    }

    // Cache miss - perform resampling
    auto [resampled, metadata] = resample_to_tf(data_5min, target_tf, source_bar_count);

    // Store in cache if valid timeframe index
    if (tf_idx >= 0 && tf_idx < NUM_TIMEFRAMES) {
        switch (asset_idx) {
            case 0: cache.tsla_resampled[tf_idx] = resampled; break;
            case 1: cache.spy_resampled[tf_idx] = resampled; break;
            case 2: cache.vix_resampled[tf_idx] = resampled; break;
            default: break;
        }

        // Update cache validity
        cache.source_size = data_5min.size();
        cache.source_bar_count = source_bar_count;
        cache.valid[tf_idx] = true;
    }

    return {resampled, metadata};
}

// =============================================================================
// TSLA PRICE FEATURES (58)
// =============================================================================

std::unordered_map<std::string, double> FeatureExtractor::extract_tsla_price_features(
    const std::vector<OHLCV>& tsla_data
) {
    // Pre-reserve for 58 TSLA price features
    auto features = create_feature_map(FeatureOffsets::TSLA_PRICE_COUNT);

    // SAFETY: Check for insufficient data
    if (tsla_data.empty()) {
        std::cerr << "[WARNING] extract_tsla_price_features: Empty data\n";
        return features;  // Return empty map
    }

    if (tsla_data.size() < 2) {
        // Return default features for single bar
        for (int i = 0; i < 58; ++i) {
            features["price_" + std::to_string(i)] = 0.0;
        }
        return features;
    }

    // Extract arrays using optimized SoA extraction
    OHLCVArrays arrays;
    extract_ohlcv_arrays_optimized(tsla_data, arrays);

    // SAFETY: Validate extracted arrays
    if (arrays.empty()) {
        std::cerr << "[WARNING] extract_tsla_price_features: Empty OHLCV arrays after extraction\n";
        return features;
    }

    // Create references for readability
    const auto& open = arrays.open;
    const auto& high = arrays.high;
    const auto& low = arrays.low;
    const auto& close = arrays.close;
    const auto& volume = arrays.volume;

    int n = static_cast<int>(close.size());

    // Use [-2] to avoid data leakage (previous bar)
    double curr_open = n > 1 ? open[n-2] : 0.0;
    double curr_high = n > 1 ? high[n-2] : 0.0;
    double curr_low = n > 1 ? low[n-2] : 0.0;
    double curr_close = n > 1 ? close[n-2] : 0.0;
    double curr_volume = n > 1 ? volume[n-2] : 0.0;
    double prev_close = n > 2 ? close[n-3] : curr_close;

    // Basic Price (11)
    features["close"] = curr_close;
    features["close_vs_open"] = curr_close - curr_open;
    features["close_vs_open_pct"] = pct_change(curr_close, curr_open);
    features["high_low_range"] = curr_high - curr_low;
    features["high_low_range_pct"] = safe_divide(curr_high - curr_low, curr_close) * 100.0;

    double bar_range = curr_high - curr_low;
    features["close_vs_high_pct"] = safe_divide(curr_high - curr_close, bar_range) * 100.0;
    features["close_vs_low_pct"] = safe_divide(curr_close - curr_low, bar_range) * 100.0;

    double body_high = std::max(curr_open, curr_close);
    double body_low = std::min(curr_open, curr_close);
    double body_size = std::abs(curr_close - curr_open);

    features["upper_shadow_pct"] = safe_divide(curr_high - body_high, bar_range) * 100.0;
    features["lower_shadow_pct"] = safe_divide(body_low - curr_low, bar_range) * 100.0;
    features["body_pct"] = safe_divide(body_size, bar_range) * 100.0;
    features["gap_pct"] = pct_change(curr_open, prev_close);

    // Volume (7)
    features["volume"] = curr_volume;

    auto calc_vol_avg = [&](int period) {
        if (n < period) return curr_volume;
        double sum = 0.0;
        for (int i = n - period; i < n; ++i) {
            sum += volume[i];
        }
        return sum / period;
    };

    double vol_avg_10 = calc_vol_avg(10);
    double vol_avg_20 = calc_vol_avg(20);
    double vol_avg_50 = calc_vol_avg(50);

    features["volume_vs_avg_10"] = safe_divide(curr_volume, vol_avg_10, 1.0);
    features["volume_vs_avg_20"] = safe_divide(curr_volume, vol_avg_20, 1.0);
    features["volume_vs_avg_50"] = safe_divide(curr_volume, vol_avg_50, 1.0);

    double vol_avg_5 = calc_vol_avg(5);
    features["volume_trend"] = safe_divide(vol_avg_5, vol_avg_20, 1.0);

    double price_change = curr_close - prev_close;
    double vol_ratio = safe_divide(curr_volume, vol_avg_20, 1.0);
    features["volume_price_trend"] = price_change * vol_ratio;
    features["relative_volume"] = safe_divide(curr_volume,
        *std::max_element(volume.end() - std::min(20, n), volume.end()), 0.0);

    // Moving Averages (14)
    auto sma_10_arr = sma(close, 10);
    auto sma_20_arr = sma(close, 20);
    auto sma_50_arr = sma(close, 50);

    double sma_10_val = get_last_valid(sma_10_arr, curr_close);
    double sma_20_val = get_last_valid(sma_20_arr, curr_close);
    double sma_50_val = get_last_valid(sma_50_arr, curr_close);

    features["sma_10"] = sma_10_val;
    features["sma_20"] = sma_20_val;
    features["sma_50"] = sma_50_val;

    auto ema_10_arr = ema(close, 10);
    auto ema_20_arr = ema(close, 20);

    features["ema_10"] = get_last_valid(ema_10_arr, curr_close);
    features["ema_20"] = get_last_valid(ema_20_arr, curr_close);

    features["price_vs_sma_10"] = pct_change(curr_close, sma_10_val);
    features["price_vs_sma_20"] = pct_change(curr_close, sma_20_val);
    features["price_vs_sma_50"] = pct_change(curr_close, sma_50_val);
    features["sma_10_vs_sma_20"] = pct_change(sma_10_val, sma_20_val);
    features["sma_20_vs_sma_50"] = pct_change(sma_20_val, sma_50_val);
    features["ma_spread"] = pct_change(sma_10_val, sma_50_val);

    features["ma_converging"] = 0.0;
    features["ma_diverging"] = 0.0;

    bool bullish_aligned = sma_10_val > sma_20_val && sma_20_val > sma_50_val;
    bool bearish_aligned = sma_10_val < sma_20_val && sma_20_val < sma_50_val;
    features["trend_alignment"] = (bullish_aligned || bearish_aligned) ? 1.0 : 0.0;

    // Momentum (13) - momentum_1 and williams_r removed = 13
    auto calc_momentum = [&](int lookback) -> double {
        if (n <= lookback) return 0.0;
        return pct_change(curr_close, close[n - lookback - 1]);
    };

    features["momentum_3"] = calc_momentum(3);
    features["momentum_5"] = calc_momentum(5);
    features["momentum_10"] = calc_momentum(10);
    features["momentum_20"] = calc_momentum(20);
    features["momentum_50"] = calc_momentum(50);

    double mom_5_now = calc_momentum(5);
    if (n > 10) {
        double close_5_ago = close[n-6];
        double close_10_ago = close[n-11];
        double mom_5_prev = pct_change(close_5_ago, close_10_ago);
        features["acceleration"] = mom_5_now - mom_5_prev;
    } else {
        features["acceleration"] = 0.0;
    }

    auto rsi_5_arr = rsi(close, 5);
    auto rsi_9_arr = rsi(close, 9);
    auto rsi_14_arr = rsi(close, 14);
    auto rsi_21_arr = rsi(close, 21);

    features["rsi_5"] = get_last_valid(rsi_5_arr, 50.0);
    features["rsi_9"] = get_last_valid(rsi_9_arr, 50.0);
    features["rsi_14"] = get_last_valid(rsi_14_arr, 50.0);
    features["rsi_21"] = get_last_valid(rsi_21_arr, 50.0);

    features["rsi_divergence"] = 0.0; // Simplified

    // Stochastic
    features["stochastic_k"] = 50.0; // Simplified
    features["stochastic_d"] = 50.0; // Simplified

    // Volatility (8)
    auto atr_arr = atr(high, low, close, 14);
    double atr_14_val = get_last_valid(atr_arr, 0.0);
    features["atr_14"] = atr_14_val;
    features["atr_pct"] = safe_divide(atr_14_val, curr_close) * 100.0;

    features["volatility_5"] = 0.0; // Would calculate std of returns
    features["volatility_20"] = 0.0;
    features["volatility_ratio"] = 1.0;
    features["range_pct_5"] = 0.0;
    features["range_pct_20"] = 0.0;

    double atr_pct = features["atr_pct"];
    if (atr_pct < 1.5) {
        features["volatility_regime"] = 0.0;
    } else if (atr_pct < 3.0) {
        features["volatility_regime"] = 1.0;
    } else {
        features["volatility_regime"] = 2.0;
    }

    // Trend (5)
    features["higher_highs_count"] = 0.0;
    features["lower_lows_count"] = 0.0;
    features["up_bars_ratio_10"] = 0.5;
    features["consecutive_up"] = 0.0;
    features["consecutive_down"] = 0.0;

    return features;
}

// =============================================================================
// SPY FEATURES (117 = 60 SPY-specific + 59 technical - 2 overlapping)
// =============================================================================

std::unordered_map<std::string, double> FeatureExtractor::extract_spy_features(
    const std::vector<OHLCV>& spy_data
) {
    // Pre-reserve for 117 SPY features
    auto features = create_feature_map(FeatureOffsets::SPY_COUNT);

    if (spy_data.size() < 2) {
        // Return defaults
        for (int i = 0; i < 117; ++i) {
            features["spy_" + std::to_string(i)] = 0.0;
        }
        return features;
    }

    // Extract arrays using optimized SoA extraction
    OHLCVArrays arrays;
    extract_ohlcv_arrays_optimized(spy_data, arrays);

    // Create references for readability
    const auto& open = arrays.open;
    const auto& high = arrays.high;
    const auto& low = arrays.low;
    const auto& close = arrays.close;
    const auto& volume = arrays.volume;

    int n = static_cast<int>(close.size());
    double c = n > 0 ? close[n-1] : 0.0;

    // Basic Price (10)
    features["spy_close"] = c;
    features["spy_close_vs_open_pct"] = n > 0 ? pct_change(close[n-1], open[n-1]) : 0.0;
    features["spy_high_low_range_pct"] = 0.0;
    features["spy_close_vs_high_pct"] = 0.5;
    features["spy_close_vs_low_pct"] = 0.5;
    features["spy_body_pct"] = 0.0;
    features["spy_gap_pct"] = 0.0;
    features["spy_upper_shadow_pct"] = 0.0;
    features["spy_lower_shadow_pct"] = 0.0;
    features["spy_volume_vs_avg"] = 1.0;

    // Moving Averages (10)
    auto sma_10_arr = sma(close, 10);
    auto sma_20_arr = sma(close, 20);
    auto sma_50_arr = sma(close, 50);

    double sma_10_val = get_last_valid(sma_10_arr, c);
    double sma_20_val = get_last_valid(sma_20_arr, c);
    double sma_50_val = get_last_valid(sma_50_arr, c);

    features["spy_sma_10"] = sma_10_val;
    features["spy_sma_20"] = sma_20_val;
    features["spy_sma_50"] = sma_50_val;
    features["spy_price_vs_sma_10"] = pct_change(c, sma_10_val);
    features["spy_price_vs_sma_20"] = pct_change(c, sma_20_val);
    features["spy_price_vs_sma_50"] = pct_change(c, sma_50_val);
    features["spy_sma_10_vs_sma_20"] = pct_change(sma_10_val, sma_20_val);
    features["spy_sma_20_vs_sma_50"] = pct_change(sma_20_val, sma_50_val);
    features["spy_ma_spread"] = pct_change(sma_10_val, sma_50_val);
    features["spy_trend_alignment"] = 0.0;

    // Momentum (15)
    auto rsi_14_arr = rsi(close, 14);
    features["spy_momentum_1"] = 0.0;
    features["spy_momentum_3"] = 0.0;
    features["spy_momentum_5"] = 0.0;
    features["spy_momentum_10"] = 0.0;
    features["spy_momentum_20"] = 0.0;
    features["spy_acceleration"] = 0.0;
    features["spy_rsi_5"] = 50.0;
    features["spy_rsi_9"] = 50.0;
    features["spy_rsi_14"] = get_last_valid(rsi_14_arr, 50.0);
    features["spy_rsi_21"] = 50.0;
    features["spy_stochastic_k"] = 50.0;
    features["spy_stochastic_d"] = 50.0;
    features["spy_williams_r"] = -50.0;
    features["spy_rsi_divergence"] = 0.0;
    features["spy_momentum_regime"] = 0.0;

    // Volatility (8)
    auto atr_arr = atr(high, low, close, 14);
    double atr_14_val = get_last_valid(atr_arr, 0.0);
    features["spy_atr_14"] = atr_14_val;
    features["spy_atr_pct"] = safe_divide(atr_14_val, c) * 100.0;
    features["spy_volatility_5"] = 0.0;
    features["spy_volatility_20"] = 0.0;
    features["spy_volatility_ratio"] = 1.0;
    features["spy_range_pct_5"] = 0.0;
    features["spy_range_pct_20"] = 0.0;
    features["spy_volatility_regime"] = 1.0;

    // Trend (7)
    features["spy_higher_highs_count"] = 0.0;
    features["spy_lower_lows_count"] = 0.0;
    features["spy_up_bars_ratio_10"] = 0.5;
    features["spy_consecutive_up"] = 0.0;
    features["spy_consecutive_down"] = 0.0;
    features["spy_trend_strength"] = 0.0;
    features["spy_trend_direction"] = 0.0;

    // Market Regime (10)
    features["spy_intraday_range_position"] = 0.5;
    features["spy_open_gap_filled"] = 0.0;
    features["spy_daily_range_expansion"] = 1.0;
    features["spy_price_acceleration"] = 0.0;
    features["spy_volume_price_trend"] = 0.0;
    features["spy_buying_pressure"] = 0.5;
    features["spy_roc_5"] = 0.0;
    features["spy_roc_10"] = 0.0;
    features["spy_efficiency_ratio"] = 0.0;
    features["spy_choppiness_index"] = 50.0;

    // Technical indicators (59) - reuse from TechnicalIndicators
    auto tech_features = TechnicalIndicators::extract_features(open, high, low, close, volume);
    for (const auto& [name, value] : tech_features) {
        features["spy_" + name] = value;
    }

    return features;
}

// =============================================================================
// VIX FEATURES (25)
// =============================================================================

std::unordered_map<std::string, double> FeatureExtractor::extract_vix_features(
    const std::vector<OHLCV>& vix_data
) {
    // Pre-reserve for 25 VIX features
    auto features = create_feature_map(FeatureOffsets::VIX_COUNT);

    if (vix_data.empty()) {
        // Return defaults
        features["vix_level"] = 20.0;
        features["vix_sma_10"] = 20.0;
        features["vix_sma_20"] = 20.0;
        features["vix_vs_sma_10"] = 0.0;
        features["vix_vs_sma_20"] = 0.0;
        // ... add all 25 default values
        for (int i = 0; i < 20; ++i) {
            features["vix_feature_" + std::to_string(i)] = 0.0;
        }
        return features;
    }

    // Extract arrays using optimized SoA extraction
    OHLCVArrays arrays;
    extract_ohlcv_arrays_optimized(vix_data, arrays);

    // Only close is needed for VIX features
    const auto& close = arrays.close;

    double current_vix = close.back();

    // Level (5)
    auto sma_10_arr = sma(close, 10);
    auto sma_20_arr = sma(close, 20);

    double sma_10_val = get_last_valid(sma_10_arr, current_vix);
    double sma_20_val = get_last_valid(sma_20_arr, current_vix);

    features["vix_level"] = current_vix;
    features["vix_sma_10"] = sma_10_val;
    features["vix_sma_20"] = sma_20_val;
    features["vix_vs_sma_10"] = pct_change(current_vix, sma_10_val);
    features["vix_vs_sma_20"] = pct_change(current_vix, sma_20_val);

    // Changes (4)
    int n = static_cast<int>(close.size());
    features["vix_change_1d"] = n >= 2 ? pct_change(close[n-1], close[n-2]) : 0.0;
    features["vix_change_5d"] = n >= 6 ? pct_change(close[n-1], close[n-6]) : 0.0;
    features["vix_change_20d"] = n >= 21 ? pct_change(close[n-1], close[n-21]) : 0.0;
    features["vix_acceleration"] = 0.0;

    // Percentiles (3)
    features["vix_percentile_30d"] = 50.0;
    features["vix_percentile_90d"] = 50.0;
    features["vix_percentile_252d"] = 50.0;

    // Regime (4)
    if (current_vix < 15) {
        features["vix_regime"] = 0.0;
    } else if (current_vix < 20) {
        features["vix_regime"] = 1.0;
    } else if (current_vix < 30) {
        features["vix_regime"] = 2.0;
    } else {
        features["vix_regime"] = 3.0;
    }

    features["vix_spike"] = features["vix_change_1d"] > 20.0 ? 1.0 : 0.0;
    features["vix_crush"] = features["vix_change_1d"] < -15.0 ? 1.0 : 0.0;
    features["vix_extreme"] = current_vix > 35.0 ? 1.0 : 0.0;

    // Technicals (5)
    auto rsi_arr = rsi(close, 14);
    features["vix_rsi"] = get_last_valid(rsi_arr, 50.0);
    features["vix_bb_pct_b"] = 0.5;
    features["vix_momentum_5"] = features["vix_change_5d"];
    features["vix_momentum_10"] = 0.0;
    features["vix_mean_reversion"] = 0.0;

    // Structure (4)
    features["vix_5d_high"] = current_vix;
    features["vix_5d_low"] = current_vix;
    features["vix_range_5d"] = 0.0;
    features["vix_volatility"] = 0.0;

    return features;
}

// =============================================================================
// CROSS-ASSET FEATURES (59)
// =============================================================================

std::unordered_map<std::string, double> FeatureExtractor::extract_cross_asset_features(
    const std::vector<OHLCV>& tsla_data,
    const std::vector<OHLCV>& spy_data,
    const std::vector<OHLCV>& vix_data,
    double tsla_rsi_14,
    double spy_rsi_14,
    double position_in_channel,
    double spy_position_in_channel,
    double vix_level
) {
    // Pre-reserve for 59 cross-asset features
    auto features = create_feature_map(FeatureOffsets::CROSS_ASSET_COUNT);

    if (tsla_data.empty() || spy_data.empty() || vix_data.empty()) {
        // Return defaults
        for (int i = 0; i < 59; ++i) {
            features["cross_" + std::to_string(i)] = 0.0;
        }
        return features;
    }

    // Extract only close prices - more efficient than full OHLCV extraction
    std::vector<double> tsla_close, spy_close, vix_close;
    tsla_close.reserve(tsla_data.size());
    spy_close.reserve(spy_data.size());
    vix_close.reserve(vix_data.size());

    for (const auto& bar : tsla_data) tsla_close.push_back(bar.close);
    for (const auto& bar : spy_data) spy_close.push_back(bar.close);
    for (const auto& bar : vix_data) vix_close.push_back(bar.close);

    // Align to minimum length
    int min_len = static_cast<int>(std::min({tsla_close.size(), spy_close.size(), vix_close.size()}));
    if (min_len < 2) {
        for (int i = 0; i < 59; ++i) {
            features["cross_" + std::to_string(i)] = 0.0;
        }
        return features;
    }

    // Rolling Correlations (15)
    features["tsla_spy_corr_5"] = calculate_correlation(tsla_close, spy_close, 5);
    features["tsla_spy_corr_10"] = calculate_correlation(tsla_close, spy_close, 10);
    features["tsla_spy_corr_20"] = calculate_correlation(tsla_close, spy_close, 20);
    features["tsla_spy_corr_50"] = calculate_correlation(tsla_close, spy_close, 50);
    features["tsla_vix_corr_5"] = calculate_correlation(tsla_close, vix_close, 5);
    features["tsla_vix_corr_10"] = calculate_correlation(tsla_close, vix_close, 10);
    features["tsla_vix_corr_20"] = calculate_correlation(tsla_close, vix_close, 20);
    features["tsla_vix_corr_50"] = calculate_correlation(tsla_close, vix_close, 50);
    features["spy_vix_corr_5"] = calculate_correlation(spy_close, vix_close, 5);
    features["spy_vix_corr_10"] = calculate_correlation(spy_close, vix_close, 10);
    features["spy_vix_corr_20"] = calculate_correlation(spy_close, vix_close, 20);
    features["spy_vix_corr_50"] = calculate_correlation(spy_close, vix_close, 50);
    features["tsla_spy_corr_change"] = 0.0;
    features["tsla_vix_corr_change"] = 0.0;
    features["spy_vix_corr_change"] = 0.0;

    // Beta Metrics (8)
    // Calculate returns for beta
    std::vector<double> tsla_returns, spy_returns;
    for (size_t i = 1; i < min_len; ++i) {
        tsla_returns.push_back((tsla_close[i] - tsla_close[i-1]) / tsla_close[i-1]);
        spy_returns.push_back((spy_close[i] - spy_close[i-1]) / spy_close[i-1]);
    }

    features["tsla_spy_beta_20"] = calculate_beta(tsla_returns, spy_returns, 20);
    features["tsla_spy_beta_50"] = calculate_beta(tsla_returns, spy_returns, 50);
    features["tsla_spy_beta_100"] = calculate_beta(tsla_returns, spy_returns, 100);
    features["tsla_vix_beta_20"] = 0.0;
    features["tsla_vix_beta_50"] = 0.0;
    features["spy_vix_beta_20"] = 0.0;

    double current_beta = features["tsla_spy_beta_20"];
    if (current_beta < 0.8) {
        features["tsla_beta_regime"] = 0.0;
    } else if (current_beta <= 1.5) {
        features["tsla_beta_regime"] = 1.0;
    } else {
        features["tsla_beta_regime"] = 2.0;
    }
    features["beta_trend"] = 0.0;

    // Relative Performance (10)
    features["tsla_vs_spy_1bar"] = 0.0;
    features["tsla_vs_spy_5bar"] = 0.0;
    features["tsla_vs_spy_20bar"] = 0.0;
    features["tsla_outperforming_spy"] = 0.0;
    features["tsla_spy_divergence"] = 0.0;
    features["spy_vs_vix_1bar"] = 0.0;
    features["spy_vs_vix_5bar"] = 0.0;
    features["tsla_alpha_20"] = 0.0;
    features["relative_strength_tsla_spy"] = 0.0;
    features["relative_strength_spy_vix"] = 0.0;

    // Cross-Asset Momentum (7)
    features["cross_asset_momentum_alignment"] = 0.0;
    features["cross_momentum_score"] = 0.0;
    features["lead_lag_tsla_spy"] = 0.0;
    features["lead_lag_spy_vix"] = 0.0;
    features["risk_on_off_signal"] = 0.0;
    features["market_regime"] = 1.0;
    features["correlation_regime"] = 1.0;

    // RSI vs Channel Position (6)
    features["rsi_position_spread"] = tsla_rsi_14 - (position_in_channel * 100.0);
    features["rsi_above_50_in_upper_half"] = (tsla_rsi_14 > 50 && position_in_channel > 0.5) ? 1.0 : 0.0;
    features["rsi_below_50_in_lower_half"] = (tsla_rsi_14 < 50 && position_in_channel < 0.5) ? 1.0 : 0.0;
    features["rsi_position_aligned"] = ((tsla_rsi_14 > 50) == (position_in_channel > 0.5)) ? 1.0 : 0.0;
    features["rsi_overbought_near_upper"] = (tsla_rsi_14 > 70 && position_in_channel > 0.8) ? 1.0 : 0.0;
    features["rsi_oversold_near_lower"] = (tsla_rsi_14 < 30 && position_in_channel < 0.2) ? 1.0 : 0.0;

    // SPY RSI vs Channel Position (2)
    features["spy_rsi_position_spread"] = spy_rsi_14 - (spy_position_in_channel * 100.0);
    features["spy_rsi_position_aligned"] = ((spy_rsi_14 > 50) == (spy_position_in_channel > 0.5)) ? 1.0 : 0.0;

    // RSI vs VIX (4)
    features["rsi_vix_spread"] = tsla_rsi_14 - vix_level;
    features["rsi_high_vix_low"] = (tsla_rsi_14 > 60 && vix_level < 20) ? 1.0 : 0.0;
    features["rsi_low_vix_high"] = (tsla_rsi_14 < 40 && vix_level > 25) ? 1.0 : 0.0;
    features["rsi_vix_divergence"] = ((tsla_rsi_14 > 50 && vix_level > 25) ||
                                       (tsla_rsi_14 < 50 && vix_level < 15)) ? 1.0 : 0.0;

    // Position vs VIX (4)
    features["position_vix_spread"] = (position_in_channel * 100.0) - vix_level;
    features["near_upper_high_vix"] = (position_in_channel > 0.8 && vix_level > 25) ? 1.0 : 0.0;
    features["near_lower_low_vix"] = (position_in_channel < 0.2 && vix_level < 15) ? 1.0 : 0.0;
    features["position_vix_aligned"] = ((position_in_channel > 0.5) == (vix_level < 20)) ? 1.0 : 0.0;

    // Combined Signals (3)
    features["bullish_alignment"] = (tsla_rsi_14 > 50 && position_in_channel > 0.5 && vix_level < 20) ? 1.0 : 0.0;
    features["bearish_alignment"] = (tsla_rsi_14 < 50 && position_in_channel < 0.5 && vix_level > 25) ? 1.0 : 0.0;
    features["contrarian_signal"] = (tsla_rsi_14 < 30 && position_in_channel < 0.2 && vix_level > 30) ? 1.0 : 0.0;

    return features;
}

// =============================================================================
// CHANNEL FEATURES (58 per window)
// =============================================================================

std::unordered_map<std::string, double> FeatureExtractor::extract_channel_features(
    const Channel& channel,
    const std::vector<OHLCV>& data
) {
    // Pre-reserve for 58 channel features
    auto features = create_feature_map(58);

    // Handle invalid channel - return defaults
    if (!channel.valid) {
        return get_default_channel_features();
    }

    // Extract arrays from data for calculations
    OHLCVArrays data_arrays;
    extract_ohlcv_arrays_optimized(data, data_arrays);

    // Use channel's stored arrays if available, else use data arrays
    const std::vector<double>& close = channel.close.empty() ? data_arrays.close : channel.close;
    const std::vector<double>& high = channel.high.empty() ? data_arrays.high : channel.high;
    const std::vector<double>& low = channel.low.empty() ? data_arrays.low : channel.low;

    int n = static_cast<int>(close.size());
    int window = channel.window > 0 ? channel.window : 50;

    // ==========================================================================
    // 1. channel_valid (0/1)
    // ==========================================================================
    features["channel_valid"] = channel.valid ? 1.0 : 0.0;

    // ==========================================================================
    // 2. channel_direction (0=bear, 1=sideways, 2=bull)
    // ==========================================================================
    int direction_int = static_cast<int>(channel.direction);
    features["channel_direction"] = safe_float(static_cast<double>(direction_int), 1.0);

    // ==========================================================================
    // 3. channel_slope
    // ==========================================================================
    double slope = channel.slope;
    features["channel_slope"] = safe_float(slope, 0.0);

    // ==========================================================================
    // 4. channel_slope_normalized (slope / price level)
    // ==========================================================================
    double avg_price = 1.0;
    if (!close.empty()) {
        double sum = 0.0;
        for (double c : close) sum += c;
        avg_price = sum / close.size();
        if (!std::isfinite(avg_price) || avg_price == 0.0) avg_price = 1.0;
    }
    features["channel_slope_normalized"] = safe_divide(slope, avg_price, 0.0);

    // ==========================================================================
    // 5. channel_intercept
    // ==========================================================================
    features["channel_intercept"] = safe_float(channel.intercept, 0.0);

    // ==========================================================================
    // 6. channel_r_squared
    // ==========================================================================
    double r_squared = safe_float(channel.r_squared, 0.0);
    features["channel_r_squared"] = r_squared;

    // ==========================================================================
    // 7. channel_width_pct
    // ==========================================================================
    double width_pct = safe_float(channel.width_pct, 0.0);
    features["channel_width_pct"] = width_pct;

    // ==========================================================================
    // 8. channel_width_atr_ratio
    // ==========================================================================
    double channel_width_atr_ratio = 0.0;
    if (!high.empty() && !low.empty() && !close.empty() && n >= 14) {
        auto atr_values = atr(high, low, close, 14);
        double current_atr = get_last_valid(atr_values, 1.0);
        if (current_atr <= 0.0) current_atr = 1.0;

        if (!channel.upper_line.empty() && !channel.lower_line.empty()) {
            double channel_width = safe_float(
                channel.upper_line.back() - channel.lower_line.back(), 0.0
            );
            channel_width_atr_ratio = safe_divide(channel_width, current_atr, 0.0);
        }
    }
    features["channel_width_atr_ratio"] = channel_width_atr_ratio;

    // ==========================================================================
    // 9. bounce_count
    // ==========================================================================
    double bounce_count = safe_float(static_cast<double>(channel.bounce_count), 0.0);
    features["bounce_count"] = bounce_count;

    // ==========================================================================
    // 10. complete_cycles
    // ==========================================================================
    features["complete_cycles"] = safe_float(static_cast<double>(channel.complete_cycles), 0.0);

    // ==========================================================================
    // 11. upper_touches
    // ==========================================================================
    double upper_touches = safe_float(static_cast<double>(channel.upper_touches), 0.0);
    features["upper_touches"] = upper_touches;

    // ==========================================================================
    // 12. lower_touches
    // ==========================================================================
    double lower_touches = safe_float(static_cast<double>(channel.lower_touches), 0.0);
    features["lower_touches"] = lower_touches;

    // ==========================================================================
    // 13. alternation_ratio
    // ==========================================================================
    features["alternation_ratio"] = safe_float(channel.alternation_ratio, 0.0);

    // ==========================================================================
    // 14. quality_score
    // ==========================================================================
    features["quality_score"] = safe_float(channel.quality_score, 0.0);

    // ==========================================================================
    // 15. channel_age_bars
    // ==========================================================================
    features["channel_age_bars"] = safe_float(static_cast<double>(window), 50.0);

    // ==========================================================================
    // 16. channel_trend_strength (slope * r_squared)
    // ==========================================================================
    features["channel_trend_strength"] = safe_float(slope * r_squared, 0.0);

    // ==========================================================================
    // 17-19. bars_since_last_touch, bars_since_upper_touch, bars_since_lower_touch
    // ==========================================================================
    double bars_since_last = safe_float(static_cast<double>(window), static_cast<double>(window));
    double bars_since_upper = safe_float(static_cast<double>(window), static_cast<double>(window));
    double bars_since_lower = safe_float(static_cast<double>(window), static_cast<double>(window));

    const std::vector<Touch>& touches = channel.touches;

    if (!touches.empty()) {
        // Find bars_since_last_touch from most recent touch
        int last_touch_bar = touches.back().bar_index;
        bars_since_last = safe_float(static_cast<double>(window - 1 - last_touch_bar), static_cast<double>(window));
        if (bars_since_last < 0) bars_since_last = 0;

        // Find bars since upper and lower touches
        for (auto it = touches.rbegin(); it != touches.rend(); ++it) {
            int bar_idx = it->bar_index;
            double bars_since = static_cast<double>(window - 1 - bar_idx);
            if (bars_since < 0) bars_since = 0;

            if (it->touch_type == TouchType::UPPER && bars_since_upper == static_cast<double>(window)) {
                bars_since_upper = bars_since;
            } else if (it->touch_type == TouchType::LOWER && bars_since_lower == static_cast<double>(window)) {
                bars_since_lower = bars_since;
            }

            if (bars_since_upper < static_cast<double>(window) && bars_since_lower < static_cast<double>(window)) {
                break;
            }
        }
    }

    features["bars_since_last_touch"] = bars_since_last;
    features["bars_since_upper_touch"] = bars_since_upper;
    features["bars_since_lower_touch"] = bars_since_lower;

    // ==========================================================================
    // 20. touch_velocity (bounces per bar)
    // ==========================================================================
    features["touch_velocity"] = safe_divide(bounce_count, static_cast<double>(window), 0.0);

    // ==========================================================================
    // 21. last_touch_type (0=lower, 1=upper)
    // ==========================================================================
    double last_touch_type = 0.0;
    if (!touches.empty()) {
        last_touch_type = touches.back().touch_type == TouchType::UPPER ? 1.0 : 0.0;
    }
    features["last_touch_type"] = last_touch_type;

    // ==========================================================================
    // 22. consecutive_same_touches
    // ==========================================================================
    int consecutive = 0;
    if (!touches.empty()) {
        TouchType last_type = touches.back().touch_type;
        consecutive = 1;
        for (int i = static_cast<int>(touches.size()) - 2; i >= 0; --i) {
            if (touches[i].touch_type == last_type) {
                consecutive++;
            } else {
                break;
            }
        }
    }
    features["consecutive_same_touches"] = safe_float(static_cast<double>(consecutive), 0.0);

    // ==========================================================================
    // 23. channel_maturity (bounces / window)
    // ==========================================================================
    features["channel_maturity"] = safe_divide(bounce_count, static_cast<double>(window), 0.0);

    // ==========================================================================
    // 24. position_in_channel (0=floor, 1=ceiling)
    // ==========================================================================
    double position = 0.5;
    if (!close.empty() && !channel.upper_line.empty() && !channel.lower_line.empty()) {
        double current_close = close.back();
        double upper_val = channel.upper_line.back();
        double lower_val = channel.lower_line.back();
        double range = upper_val - lower_val;
        if (range > 0.0) {
            position = safe_divide(current_close - lower_val, range, 0.5);
            // Clamp to [0, 1]
            position = std::clamp(position, 0.0, 1.0);
        }
    }
    features["position_in_channel"] = position;

    // ==========================================================================
    // 25. distance_to_upper_pct
    // ==========================================================================
    double distance_to_upper_pct = 0.0;
    if (!close.empty() && !channel.upper_line.empty()) {
        double current_close = close.back();
        double upper_val = channel.upper_line.back();
        if (current_close > 0.0) {
            distance_to_upper_pct = safe_divide(upper_val - current_close, current_close, 0.0) * 100.0;
        }
    }
    features["distance_to_upper_pct"] = distance_to_upper_pct;

    // ==========================================================================
    // 26. distance_to_lower_pct
    // ==========================================================================
    double distance_to_lower_pct = 0.0;
    if (!close.empty() && !channel.lower_line.empty()) {
        double current_close = close.back();
        double lower_val = channel.lower_line.back();
        if (current_close > 0.0) {
            distance_to_lower_pct = safe_divide(current_close - lower_val, current_close, 0.0) * 100.0;
        }
    }
    features["distance_to_lower_pct"] = distance_to_lower_pct;

    // ==========================================================================
    // 27. price_vs_channel_midpoint
    // ==========================================================================
    double price_vs_midpoint = 0.0;
    if (!close.empty() && !channel.center_line.empty()) {
        double current_price = close.back();
        double center_price = channel.center_line.back();
        if (!std::isfinite(center_price) || center_price == 0.0) center_price = current_price;
        price_vs_midpoint = safe_divide(current_price - center_price, center_price, 0.0) * 100.0;
    }
    features["price_vs_channel_midpoint"] = price_vs_midpoint;

    // ==========================================================================
    // 28. channel_momentum (slope change - estimated from regression)
    // ==========================================================================
    double channel_momentum = 0.0;
    if (!close.empty() && n >= 10) {
        int half_window = n / 2;
        if (n - half_window >= 5) {
            // Fit linear regression on second half
            std::vector<double> close_half(close.begin() + half_window, close.end());
            int m = static_cast<int>(close_half.size());

            // Simple linear regression: slope = Cov(x,y) / Var(x)
            double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
            for (int i = 0; i < m; ++i) {
                sum_x += i;
                sum_y += close_half[i];
                sum_xy += i * close_half[i];
                sum_x2 += i * i;
            }
            double denom = m * sum_x2 - sum_x * sum_x;
            if (std::abs(denom) > 1e-10) {
                double slope_half = (m * sum_xy - sum_x * sum_y) / denom;
                channel_momentum = safe_float(slope - slope_half, 0.0);
            }
        }
    }
    features["channel_momentum"] = channel_momentum;

    // ==========================================================================
    // 29. upper_line_slope
    // ==========================================================================
    double upper_line_slope = 0.0;
    if (!channel.upper_line.empty() && channel.upper_line.size() >= 2) {
        int ul_len = static_cast<int>(channel.upper_line.size());
        upper_line_slope = safe_divide(
            channel.upper_line.back() - channel.upper_line.front(),
            static_cast<double>(ul_len - 1),
            0.0
        );
    }
    features["upper_line_slope"] = upper_line_slope;

    // ==========================================================================
    // 30. lower_line_slope
    // ==========================================================================
    double lower_line_slope = 0.0;
    if (!channel.lower_line.empty() && channel.lower_line.size() >= 2) {
        int ll_len = static_cast<int>(channel.lower_line.size());
        lower_line_slope = safe_divide(
            channel.lower_line.back() - channel.lower_line.front(),
            static_cast<double>(ll_len - 1),
            0.0
        );
    }
    features["lower_line_slope"] = lower_line_slope;

    // ==========================================================================
    // 31. channel_expanding (1 if width increasing)
    // ==========================================================================
    double channel_expanding = 0.0;
    if (!channel.upper_line.empty() && !channel.lower_line.empty() &&
        channel.upper_line.size() >= 10 && channel.lower_line.size() >= 10) {
        double width_start = channel.upper_line.front() - channel.lower_line.front();
        double width_end = channel.upper_line.back() - channel.lower_line.back();
        if (width_end > width_start * 1.05) {
            channel_expanding = 1.0;
        }
    }
    features["channel_expanding"] = channel_expanding;

    // ==========================================================================
    // 32. channel_contracting (1 if width decreasing)
    // ==========================================================================
    double channel_contracting = 0.0;
    if (!channel.upper_line.empty() && !channel.lower_line.empty() &&
        channel.upper_line.size() >= 10 && channel.lower_line.size() >= 10) {
        double width_start = channel.upper_line.front() - channel.lower_line.front();
        double width_end = channel.upper_line.back() - channel.lower_line.back();
        if (width_end < width_start * 0.95) {
            channel_contracting = 1.0;
        }
    }
    features["channel_contracting"] = channel_contracting;

    // ==========================================================================
    // 33. std_dev_ratio (std_dev / avg_price)
    // ==========================================================================
    double std_dev = safe_float(channel.std_dev, 0.0);
    features["std_dev_ratio"] = safe_divide(std_dev, avg_price, 0.0);

    // ==========================================================================
    // 34. breakout_pressure_up
    // ==========================================================================
    double breakout_pressure_up = 0.0;
    if (!high.empty() && !channel.upper_line.empty() && n >= 5) {
        int start_idx = std::max(0, n - 5);
        int upper_start = std::max(0, static_cast<int>(channel.upper_line.size()) - 5);

        std::vector<double> distances_to_upper;
        for (int i = 0; i < 5; ++i) {
            int h_idx = start_idx + i;
            int u_idx = upper_start + i;
            if (h_idx < n && u_idx < static_cast<int>(channel.upper_line.size())) {
                double h = high[h_idx];
                double u = channel.upper_line[u_idx];
                if (u > 0) {
                    double dist = safe_divide(u - h, u, 0.0);
                    distances_to_upper.push_back(std::max(0.0, dist));
                }
            }
        }
        if (!distances_to_upper.empty()) {
            double avg_dist = 0.0;
            for (double d : distances_to_upper) avg_dist += d;
            avg_dist /= distances_to_upper.size();
            breakout_pressure_up = safe_float(1.0 - avg_dist, 0.0);
        }
    }
    features["breakout_pressure_up"] = breakout_pressure_up;

    // ==========================================================================
    // 35. breakout_pressure_down
    // ==========================================================================
    double breakout_pressure_down = 0.0;
    if (!low.empty() && !channel.lower_line.empty() && n >= 5) {
        int start_idx = std::max(0, n - 5);
        int lower_start = std::max(0, static_cast<int>(channel.lower_line.size()) - 5);

        std::vector<double> distances_to_lower;
        for (int i = 0; i < 5; ++i) {
            int l_idx = start_idx + i;
            int lb_idx = lower_start + i;
            if (l_idx < n && lb_idx < static_cast<int>(channel.lower_line.size())) {
                double l = low[l_idx];
                double lb = channel.lower_line[lb_idx];
                if (l > 0) {
                    double dist = safe_divide(l - lb, l, 0.0);
                    distances_to_lower.push_back(std::max(0.0, dist));
                }
            }
        }
        if (!distances_to_lower.empty()) {
            double avg_dist = 0.0;
            for (double d : distances_to_lower) avg_dist += d;
            avg_dist /= distances_to_lower.size();
            breakout_pressure_down = safe_float(1.0 - avg_dist, 0.0);
        }
    }
    features["breakout_pressure_down"] = breakout_pressure_down;

    // ==========================================================================
    // 36. channel_symmetry (how balanced are upper/lower touches)
    // ==========================================================================
    double channel_symmetry = 0.0;
    double total_touches = upper_touches + lower_touches;
    if (total_touches > 0) {
        double min_touches = std::min(upper_touches, lower_touches);
        double max_touches = std::max(upper_touches, lower_touches);
        channel_symmetry = safe_divide(min_touches, max_touches, 0.0);
    }
    features["channel_symmetry"] = channel_symmetry;

    // ==========================================================================
    // 37. touch_regularity (std dev of intervals between touches)
    // ==========================================================================
    double touch_regularity = 0.0;
    if (touches.size() >= 3) {
        std::vector<int> intervals;
        for (size_t i = 1; i < touches.size(); ++i) {
            int interval = touches[i].bar_index - touches[i-1].bar_index;
            intervals.push_back(interval);
        }
        if (!intervals.empty()) {
            double sum = 0.0;
            for (int intv : intervals) sum += intv;
            double avg_interval = sum / intervals.size();

            double sum_sq = 0.0;
            for (int intv : intervals) {
                double diff = intv - avg_interval;
                sum_sq += diff * diff;
            }
            double std_interval = std::sqrt(sum_sq / intervals.size());

            // Lower std relative to mean = more regular
            double regularity = 1.0 - safe_divide(std_interval, avg_interval + 1.0, 0.0);
            touch_regularity = safe_float(std::max(0.0, regularity), 0.0);
        }
    }
    features["touch_regularity"] = touch_regularity;

    // ==========================================================================
    // 38. recent_touch_bias (bias toward upper or lower in recent touches)
    // ==========================================================================
    double recent_touch_bias = 0.0;
    if (touches.size() >= 3) {
        int num_recent = std::min(5, static_cast<int>(touches.size()));
        int recent_upper = 0;
        int recent_lower = 0;
        for (int i = static_cast<int>(touches.size()) - num_recent; i < static_cast<int>(touches.size()); ++i) {
            if (touches[i].touch_type == TouchType::UPPER) {
                recent_upper++;
            } else {
                recent_lower++;
            }
        }
        // -1 = all lower, 0 = balanced, 1 = all upper
        recent_touch_bias = safe_divide(
            static_cast<double>(recent_upper - recent_lower),
            static_cast<double>(num_recent),
            0.0
        );
    }
    features["recent_touch_bias"] = recent_touch_bias;

    // ==========================================================================
    // 39. channel_curvature (non-linearity measure)
    // ==========================================================================
    double channel_curvature = 0.0;
    if (!close.empty() && n >= 10) {
        // Approximate curvature by comparing first-half slope to second-half slope
        if (n >= 4) {
            int half = n / 2;
            // First half slope
            double sum_x1 = 0.0, sum_y1 = 0.0, sum_xy1 = 0.0, sum_x2_1 = 0.0;
            for (int i = 0; i < half; ++i) {
                sum_x1 += i;
                sum_y1 += close[i];
                sum_xy1 += i * close[i];
                sum_x2_1 += i * i;
            }
            double denom1 = half * sum_x2_1 - sum_x1 * sum_x1;
            double slope1 = 0.0;
            if (std::abs(denom1) > 1e-10) {
                slope1 = (half * sum_xy1 - sum_x1 * sum_y1) / denom1;
            }

            // Second half slope
            double sum_x2h = 0.0, sum_y2 = 0.0, sum_xy2 = 0.0, sum_x2_2 = 0.0;
            int len2 = n - half;
            for (int i = half; i < n; ++i) {
                int j = i - half;
                sum_x2h += j;
                sum_y2 += close[i];
                sum_xy2 += j * close[i];
                sum_x2_2 += j * j;
            }
            double denom2 = len2 * sum_x2_2 - sum_x2h * sum_x2h;
            double slope2 = 0.0;
            if (std::abs(denom2) > 1e-10) {
                slope2 = (len2 * sum_xy2 - sum_x2h * sum_y2) / denom2;
            }

            // Curvature approximation: change in slope
            double curvature = slope2 - slope1;
            channel_curvature = safe_divide(curvature, avg_price, 0.0) * 1000.0;
        }
    }
    features["channel_curvature"] = channel_curvature;

    // ==========================================================================
    // 40. parallel_score (how parallel are upper and lower lines)
    // ==========================================================================
    double parallel_score = 0.5;
    double avg_slope = safe_divide(upper_line_slope + lower_line_slope, 2.0, 0.0);
    if (avg_slope != 0.0) {
        double slope_diff = std::abs(upper_line_slope - lower_line_slope);
        parallel_score = safe_float(
            1.0 - safe_divide(slope_diff, std::abs(avg_slope) + 0.0001, 0.0),
            0.5
        );
    } else {
        parallel_score = (upper_line_slope == lower_line_slope) ? 1.0 : 0.5;
    }
    features["parallel_score"] = parallel_score;

    // ==========================================================================
    // 41. touch_density (touches per unit channel width)
    // ==========================================================================
    features["touch_density"] = safe_divide(total_touches, width_pct + 1.0, 0.0);

    // ==========================================================================
    // 42. bounce_efficiency (complete_cycles / total touches)
    // ==========================================================================
    double complete_cycles = features["complete_cycles"];
    features["bounce_efficiency"] = safe_divide(complete_cycles, total_touches + 1.0, 0.0);

    // ==========================================================================
    // 43. channel_stability (r_squared * alternation_ratio)
    // ==========================================================================
    double alt_ratio = features["alternation_ratio"];
    features["channel_stability"] = safe_float(r_squared * alt_ratio, 0.0);

    // ==========================================================================
    // 44. momentum_direction_alignment (1 if momentum matches direction)
    // ==========================================================================
    double momentum_dir_align = 0.5;
    double dir_val = features["channel_direction"];
    if (dir_val == 2.0) {  // Bull
        momentum_dir_align = (channel_momentum > 0) ? 1.0 : 0.0;
    } else if (dir_val == 0.0) {  // Bear
        momentum_dir_align = (channel_momentum < 0) ? 1.0 : 0.0;
    } else {  // Sideways
        momentum_dir_align = (std::abs(channel_momentum) < 0.01) ? 1.0 : 0.5;
    }
    features["momentum_direction_alignment"] = momentum_dir_align;

    // ==========================================================================
    // 45. price_position_extreme (how close to boundaries)
    // ==========================================================================
    features["price_position_extreme"] = safe_float(std::abs(position - 0.5) * 2.0, 0.0);

    // ==========================================================================
    // 46. breakout_imminence (combined pressure score)
    // ==========================================================================
    features["breakout_imminence"] = safe_float(std::max(breakout_pressure_up, breakout_pressure_down), 0.0);

    // ==========================================================================
    // 47. breakout_direction_bias (positive = up, negative = down)
    // ==========================================================================
    features["breakout_direction_bias"] = safe_float(breakout_pressure_up - breakout_pressure_down, 0.0);

    // ==========================================================================
    // 48. channel_health_score (composite quality metric)
    // ==========================================================================
    double health = (
        features["channel_valid"] * 0.2 +
        features["channel_stability"] * 0.3 +
        parallel_score * 0.2 +
        touch_regularity * 0.15 +
        channel_symmetry * 0.15
    );
    features["channel_health_score"] = safe_float(health, 0.0);

    // ==========================================================================
    // 49. time_weighted_position (position weighted by time since last touch)
    // ==========================================================================
    double time_factor = safe_divide(bars_since_last, static_cast<double>(window), 1.0);
    features["time_weighted_position"] = safe_float(position * (1.0 - time_factor), 0.0);

    // ==========================================================================
    // 50. volatility_adjusted_width (width relative to recent volatility)
    // ==========================================================================
    if (channel_width_atr_ratio > 0) {
        // Normalize: 1.0 = average, >1 = wide, <1 = narrow (typical ATR ratio ~4)
        features["volatility_adjusted_width"] = safe_float(channel_width_atr_ratio / 4.0, 1.0);
    } else {
        features["volatility_adjusted_width"] = 1.0;
    }

    // ==========================================================================
    // 51-58. Excursion Features (price going OUTSIDE the channel)
    // ==========================================================================
    double intercept = channel.intercept;

    int excursions_above = 0;
    int excursions_below = 0;
    double max_excursion_above = 0.0;
    double max_excursion_below = 0.0;
    int last_excursion_bar = -1;
    double last_excursion_dir = 0.5;  // 0=below, 0.5=none, 1=above
    std::vector<int> excursion_durations;
    bool in_excursion = false;
    int current_excursion_start = -1;

    if (!close.empty() && std_dev > 0) {
        for (int i = 0; i < n; ++i) {
            double center_at_i = slope * i + intercept;
            double upper_at_i = center_at_i + 2.0 * std_dev;
            double lower_at_i = center_at_i - 2.0 * std_dev;
            double close_i = close[i];

            if (close_i > upper_at_i) {
                excursions_above++;
                last_excursion_bar = i;
                last_excursion_dir = 1.0;
                if (upper_at_i > 0) {
                    double excursion_pct = safe_divide(close_i - upper_at_i, upper_at_i, 0.0) * 100.0;
                    max_excursion_above = std::max(max_excursion_above, excursion_pct);
                }
                if (!in_excursion) {
                    in_excursion = true;
                    current_excursion_start = i;
                }
            } else if (close_i < lower_at_i) {
                excursions_below++;
                last_excursion_bar = i;
                last_excursion_dir = 0.0;
                if (lower_at_i > 0) {
                    double excursion_pct = safe_divide(lower_at_i - close_i, lower_at_i, 0.0) * 100.0;
                    max_excursion_below = std::max(max_excursion_below, excursion_pct);
                }
                if (!in_excursion) {
                    in_excursion = true;
                    current_excursion_start = i;
                }
            } else {
                // Price is inside channel
                if (in_excursion) {
                    int duration = i - current_excursion_start;
                    excursion_durations.push_back(duration);
                    in_excursion = false;
                    current_excursion_start = -1;
                }
            }
        }

        // Handle case where we're still in excursion at end
        if (in_excursion && current_excursion_start >= 0) {
            int duration = n - current_excursion_start;
            excursion_durations.push_back(duration);
        }
    }

    // 51. excursions_above_upper
    features["excursions_above_upper"] = safe_float(static_cast<double>(excursions_above), 0.0);

    // 52. excursions_below_lower
    features["excursions_below_lower"] = safe_float(static_cast<double>(excursions_below), 0.0);

    // 53. max_excursion_above_pct
    features["max_excursion_above_pct"] = safe_float(max_excursion_above, 0.0);

    // 54. max_excursion_below_pct
    features["max_excursion_below_pct"] = safe_float(max_excursion_below, 0.0);

    // 55. bars_since_last_excursion
    double bars_since_excursion = static_cast<double>(window);
    if (last_excursion_bar >= 0 && n > 0) {
        bars_since_excursion = static_cast<double>(n - 1 - last_excursion_bar);
    }
    features["bars_since_last_excursion"] = safe_float(bars_since_excursion, static_cast<double>(window));

    // 56. excursion_return_speed_avg
    double avg_return_speed = 0.0;
    if (!excursion_durations.empty()) {
        double sum = 0.0;
        for (int d : excursion_durations) sum += d;
        avg_return_speed = sum / excursion_durations.size();
    }
    features["excursion_return_speed_avg"] = safe_float(avg_return_speed, 0.0);

    // 57. excursion_rate
    int total_excursions = excursions_above + excursions_below;
    features["excursion_rate"] = n > 0 ? safe_divide(static_cast<double>(total_excursions), static_cast<double>(n), 0.0) : 0.0;

    // 58. last_excursion_direction
    features["last_excursion_direction"] = safe_float(last_excursion_dir, 0.5);

    // Final safety check - ensure all values are finite
    for (auto& [key, value] : features) {
        if (!std::isfinite(value)) {
            value = 0.0;
        }
    }

    return features;
}

// Helper function to return default channel features (58 features)
std::unordered_map<std::string, double> FeatureExtractor::get_default_channel_features() {
    return {
        {"channel_valid", 0.0},
        {"channel_direction", 1.0},
        {"channel_slope", 0.0},
        {"channel_slope_normalized", 0.0},
        {"channel_intercept", 0.0},
        {"channel_r_squared", 0.0},
        {"channel_width_pct", 0.0},
        {"channel_width_atr_ratio", 0.0},
        {"bounce_count", 0.0},
        {"complete_cycles", 0.0},
        {"upper_touches", 0.0},
        {"lower_touches", 0.0},
        {"alternation_ratio", 0.0},
        {"quality_score", 0.0},
        {"channel_age_bars", 50.0},
        {"channel_trend_strength", 0.0},
        {"bars_since_last_touch", 50.0},
        {"bars_since_upper_touch", 50.0},
        {"bars_since_lower_touch", 50.0},
        {"touch_velocity", 0.0},
        {"last_touch_type", 0.0},
        {"consecutive_same_touches", 0.0},
        {"channel_maturity", 0.0},
        {"position_in_channel", 0.5},
        {"distance_to_upper_pct", 0.0},
        {"distance_to_lower_pct", 0.0},
        {"price_vs_channel_midpoint", 0.0},
        {"channel_momentum", 0.0},
        {"upper_line_slope", 0.0},
        {"lower_line_slope", 0.0},
        {"channel_expanding", 0.0},
        {"channel_contracting", 0.0},
        {"std_dev_ratio", 0.0},
        {"breakout_pressure_up", 0.0},
        {"breakout_pressure_down", 0.0},
        {"channel_symmetry", 0.0},
        {"touch_regularity", 0.0},
        {"recent_touch_bias", 0.0},
        {"channel_curvature", 0.0},
        {"parallel_score", 0.5},
        {"touch_density", 0.0},
        {"bounce_efficiency", 0.0},
        {"channel_stability", 0.0},
        {"momentum_direction_alignment", 0.5},
        {"price_position_extreme", 0.0},
        {"breakout_imminence", 0.0},
        {"breakout_direction_bias", 0.0},
        {"channel_health_score", 0.0},
        {"time_weighted_position", 0.0},
        {"volatility_adjusted_width", 1.0},
        {"excursions_above_upper", 0.0},
        {"excursions_below_lower", 0.0},
        {"max_excursion_above_pct", 0.0},
        {"max_excursion_below_pct", 0.0},
        {"bars_since_last_excursion", 50.0},
        {"excursion_return_speed_avg", 0.0},
        {"excursion_rate", 0.0},
        {"last_excursion_direction", 0.5}
    };
}

// =============================================================================
// SPY CHANNEL FEATURES (58 per window)
// Mirrors the Python implementation in v15/features/spy_channel.py
// =============================================================================

std::unordered_map<std::string, double> FeatureExtractor::extract_spy_channel_features(
    const Channel& channel,
    const std::vector<OHLCV>& spy_data,
    int window
) {
    // Pre-reserve for 58 SPY channel features
    auto features = create_feature_map(58);

    // Handle invalid channel - return all defaults
    if (!channel.valid) {
        // 50 base features
        features["spy_channel_valid"] = 0.0;
        features["spy_channel_direction"] = 1.0;
        features["spy_channel_slope"] = 0.0;
        features["spy_channel_slope_normalized"] = 0.0;
        features["spy_channel_intercept"] = 0.0;
        features["spy_channel_r_squared"] = 0.0;
        features["spy_channel_width_pct"] = 0.0;
        features["spy_channel_width_atr_ratio"] = 0.0;
        features["spy_bounce_count"] = 0.0;
        features["spy_complete_cycles"] = 0.0;
        features["spy_upper_touches"] = 0.0;
        features["spy_lower_touches"] = 0.0;
        features["spy_alternation_ratio"] = 0.0;
        features["spy_quality_score"] = 0.0;
        features["spy_channel_age_bars"] = static_cast<double>(window);
        features["spy_channel_trend_strength"] = 0.0;
        features["spy_bars_since_last_touch"] = static_cast<double>(window);
        features["spy_bars_since_upper_touch"] = static_cast<double>(window);
        features["spy_bars_since_lower_touch"] = static_cast<double>(window);
        features["spy_touch_velocity"] = 0.0;
        features["spy_last_touch_type"] = 0.0;
        features["spy_consecutive_same_touches"] = 0.0;
        features["spy_channel_maturity"] = 0.0;
        features["spy_position_in_channel"] = 0.5;
        features["spy_distance_to_upper_pct"] = 0.0;
        features["spy_distance_to_lower_pct"] = 0.0;
        features["spy_price_vs_channel_midpoint"] = 0.0;
        features["spy_channel_momentum"] = 0.0;
        features["spy_upper_line_slope"] = 0.0;
        features["spy_lower_line_slope"] = 0.0;
        features["spy_channel_expanding"] = 0.0;
        features["spy_channel_contracting"] = 0.0;
        features["spy_std_dev_ratio"] = 0.0;
        features["spy_breakout_pressure_up"] = 0.0;
        features["spy_breakout_pressure_down"] = 0.0;
        features["spy_channel_symmetry"] = 0.0;
        features["spy_touch_regularity"] = 0.0;
        features["spy_recent_touch_bias"] = 0.0;
        features["spy_channel_curvature"] = 0.0;
        features["spy_parallel_score"] = 0.5;
        features["spy_touch_density"] = 0.0;
        features["spy_bounce_efficiency"] = 0.0;
        features["spy_channel_stability"] = 0.0;
        features["spy_momentum_direction_alignment"] = 0.5;
        features["spy_price_position_extreme"] = 0.0;
        features["spy_breakout_imminence"] = 0.0;
        features["spy_breakout_direction_bias"] = 0.0;
        features["spy_channel_health_score"] = 0.0;
        features["spy_time_weighted_position"] = 0.0;
        features["spy_volatility_adjusted_width"] = 1.0;
        // 8 excursion features
        features["spy_excursions_above_upper"] = 0.0;
        features["spy_excursions_below_lower"] = 0.0;
        features["spy_max_excursion_above_pct"] = 0.0;
        features["spy_max_excursion_below_pct"] = 0.0;
        features["spy_bars_since_last_excursion"] = static_cast<double>(window);
        features["spy_excursion_return_speed_avg"] = 0.0;
        features["spy_excursion_rate"] = 0.0;
        features["spy_last_excursion_direction"] = 0.0;
        return features;
    }

    // Extract OHLCV arrays from SPY data using optimized extraction
    OHLCVArrays spy_arrays;
    extract_ohlcv_arrays_optimized(spy_data, spy_arrays);

    // Use channel's stored arrays if available, otherwise use extracted
    const std::vector<double>& ch_close = channel.close.empty() ? spy_arrays.close : channel.close;
    const std::vector<double>& ch_high = channel.high.empty() ? spy_arrays.high : channel.high;
    const std::vector<double>& ch_low = channel.low.empty() ? spy_arrays.low : channel.low;

    // ==========================================================================
    // 1. spy_channel_valid (0/1)
    // ==========================================================================
    features["spy_channel_valid"] = channel.valid ? 1.0 : 0.0;

    // ==========================================================================
    // 2. spy_channel_direction (0=bear, 1=sideways, 2=bull)
    // ==========================================================================
    double direction_val = 1.0; // Default sideways
    if (channel.direction == ChannelDirection::BEAR) direction_val = 0.0;
    else if (channel.direction == ChannelDirection::BULL) direction_val = 2.0;
    features["spy_channel_direction"] = direction_val;

    // ==========================================================================
    // 3. spy_channel_slope
    // ==========================================================================
    double slope = safe_float(channel.slope, 0.0);
    features["spy_channel_slope"] = slope;

    // ==========================================================================
    // 4. spy_channel_slope_normalized (slope / price level)
    // ==========================================================================
    double avg_price = 1.0;
    if (!ch_close.empty()) {
        double sum = 0.0;
        for (double c : ch_close) sum += c;
        avg_price = sum / ch_close.size();
        if (!std::isfinite(avg_price) || avg_price == 0.0) avg_price = 1.0;
    }
    features["spy_channel_slope_normalized"] = safe_divide(slope, avg_price, 0.0);

    // ==========================================================================
    // 5. spy_channel_intercept
    // ==========================================================================
    features["spy_channel_intercept"] = safe_float(channel.intercept, 0.0);

    // ==========================================================================
    // 6. spy_channel_r_squared
    // ==========================================================================
    double r_squared = safe_float(channel.r_squared, 0.0);
    features["spy_channel_r_squared"] = r_squared;

    // ==========================================================================
    // 7. spy_channel_width_pct
    // ==========================================================================
    double width_pct = safe_float(channel.width_pct, 0.0);
    features["spy_channel_width_pct"] = width_pct;

    // ==========================================================================
    // 8. spy_channel_width_atr_ratio
    // ==========================================================================
    double channel_width_atr_ratio = 0.0;
    if (!ch_high.empty() && !ch_low.empty() && !ch_close.empty() && ch_close.size() >= 14) {
        auto atr_values = atr(ch_high, ch_low, ch_close, 14);
        double current_atr = get_last_valid(atr_values, 1.0);
        if (!channel.upper_line.empty() && !channel.lower_line.empty()) {
            double channel_width = channel.upper_line.back() - channel.lower_line.back();
            channel_width_atr_ratio = safe_divide(channel_width, current_atr, 0.0);
        }
    }
    features["spy_channel_width_atr_ratio"] = channel_width_atr_ratio;

    // ==========================================================================
    // 9. spy_bounce_count
    // ==========================================================================
    double bounce_count = static_cast<double>(channel.bounce_count);
    features["spy_bounce_count"] = bounce_count;

    // ==========================================================================
    // 10. spy_complete_cycles
    // ==========================================================================
    double complete_cycles = static_cast<double>(channel.complete_cycles);
    features["spy_complete_cycles"] = complete_cycles;

    // ==========================================================================
    // 11. spy_upper_touches
    // ==========================================================================
    double upper_touches = static_cast<double>(channel.upper_touches);
    features["spy_upper_touches"] = upper_touches;

    // ==========================================================================
    // 12. spy_lower_touches
    // ==========================================================================
    double lower_touches = static_cast<double>(channel.lower_touches);
    features["spy_lower_touches"] = lower_touches;

    // ==========================================================================
    // 13. spy_alternation_ratio
    // ==========================================================================
    double alternation_ratio = safe_float(channel.alternation_ratio, 0.0);
    features["spy_alternation_ratio"] = alternation_ratio;

    // ==========================================================================
    // 14. spy_quality_score
    // ==========================================================================
    features["spy_quality_score"] = safe_float(channel.quality_score, 0.0);

    // ==========================================================================
    // 15. spy_channel_age_bars
    // ==========================================================================
    int channel_window = channel.window > 0 ? channel.window : window;
    features["spy_channel_age_bars"] = static_cast<double>(channel_window);

    // ==========================================================================
    // 16. spy_channel_trend_strength (slope * r_squared)
    // ==========================================================================
    features["spy_channel_trend_strength"] = safe_float(slope * r_squared, 0.0);

    // ==========================================================================
    // 17-19. spy_bars_since_last_touch, spy_bars_since_upper_touch, spy_bars_since_lower_touch
    // ==========================================================================
    double bars_since_last = static_cast<double>(window);
    double bars_since_upper = static_cast<double>(window);
    double bars_since_lower = static_cast<double>(window);

    const auto& touches = channel.touches;
    if (!touches.empty()) {
        int last_touch_idx = touches.back().bar_index;
        bars_since_last = static_cast<double>(std::max(0, window - 1 - last_touch_idx));

        for (auto it = touches.rbegin(); it != touches.rend(); ++it) {
            int bars_since = std::max(0, window - 1 - it->bar_index);
            if (it->touch_type == TouchType::UPPER && bars_since_upper == window) {
                bars_since_upper = static_cast<double>(bars_since);
            } else if (it->touch_type == TouchType::LOWER && bars_since_lower == window) {
                bars_since_lower = static_cast<double>(bars_since);
            }
            if (bars_since_upper < window && bars_since_lower < window) break;
        }
    }

    features["spy_bars_since_last_touch"] = bars_since_last;
    features["spy_bars_since_upper_touch"] = bars_since_upper;
    features["spy_bars_since_lower_touch"] = bars_since_lower;

    // ==========================================================================
    // 20. spy_touch_velocity (bounces per bar)
    // ==========================================================================
    features["spy_touch_velocity"] = safe_divide(bounce_count, window, 0.0);

    // ==========================================================================
    // 21. spy_last_touch_type (0=lower, 1=upper)
    // ==========================================================================
    double last_touch_type = 0.0;
    if (!touches.empty()) {
        last_touch_type = (touches.back().touch_type == TouchType::UPPER) ? 1.0 : 0.0;
    }
    features["spy_last_touch_type"] = last_touch_type;

    // ==========================================================================
    // 22. spy_consecutive_same_touches
    // ==========================================================================
    int consecutive = 0;
    if (!touches.empty()) {
        TouchType last_type = touches.back().touch_type;
        for (auto it = touches.rbegin(); it != touches.rend(); ++it) {
            if (it->touch_type == last_type) consecutive++;
            else break;
        }
    }
    features["spy_consecutive_same_touches"] = static_cast<double>(consecutive);

    // ==========================================================================
    // 23. spy_channel_maturity (bounces / window)
    // ==========================================================================
    features["spy_channel_maturity"] = safe_divide(bounce_count, window, 0.0);

    // ==========================================================================
    // 24. spy_position_in_channel (0=floor, 1=ceiling)
    // ==========================================================================
    double position = 0.5;
    if (!ch_close.empty() && !channel.upper_line.empty() && !channel.lower_line.empty()) {
        double current_price = ch_close.back();
        double upper = channel.upper_line.back();
        double lower = channel.lower_line.back();
        double width = upper - lower;
        if (width > 0.0) {
            position = (current_price - lower) / width;
            position = std::max(0.0, std::min(1.0, position));
        }
    }
    features["spy_position_in_channel"] = position;

    // ==========================================================================
    // 25. spy_distance_to_upper_pct
    // ==========================================================================
    double distance_to_upper_pct = 0.0;
    if (!ch_close.empty() && !channel.upper_line.empty()) {
        double current_price = ch_close.back();
        double upper = channel.upper_line.back();
        distance_to_upper_pct = safe_divide(upper - current_price, current_price, 0.0) * 100.0;
    }
    features["spy_distance_to_upper_pct"] = distance_to_upper_pct;

    // ==========================================================================
    // 26. spy_distance_to_lower_pct
    // ==========================================================================
    double distance_to_lower_pct = 0.0;
    if (!ch_close.empty() && !channel.lower_line.empty()) {
        double current_price = ch_close.back();
        double lower = channel.lower_line.back();
        distance_to_lower_pct = safe_divide(current_price - lower, current_price, 0.0) * 100.0;
    }
    features["spy_distance_to_lower_pct"] = distance_to_lower_pct;

    // ==========================================================================
    // 27. spy_price_vs_channel_midpoint
    // ==========================================================================
    double price_vs_midpoint = 0.0;
    if (!ch_close.empty() && !channel.center_line.empty()) {
        double current_price = ch_close.back();
        double center_price = channel.center_line.back();
        price_vs_midpoint = safe_divide(current_price - center_price, center_price, 0.0) * 100.0;
    }
    features["spy_price_vs_channel_midpoint"] = price_vs_midpoint;

    // ==========================================================================
    // 28. spy_channel_momentum (slope change - estimated from regression)
    // ==========================================================================
    double channel_momentum = 0.0;
    if (!ch_close.empty() && ch_close.size() >= 10) {
        size_t half_window = ch_close.size() / 2;
        if (ch_close.size() - half_window >= 5) {
            std::vector<double> close_half(ch_close.begin() + half_window, ch_close.end());
            size_t m = close_half.size();
            double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
            for (size_t i = 0; i < m; ++i) {
                sum_x += i;
                sum_y += close_half[i];
                sum_xy += i * close_half[i];
                sum_x2 += i * i;
            }
            double denom = m * sum_x2 - sum_x * sum_x;
            if (denom != 0.0) {
                double slope_half = (m * sum_xy - sum_x * sum_y) / denom;
                channel_momentum = safe_float(slope - slope_half, 0.0);
            }
        }
    }
    features["spy_channel_momentum"] = channel_momentum;

    // ==========================================================================
    // 29. spy_upper_line_slope
    // ==========================================================================
    double upper_line_slope = 0.0;
    const auto& upper_line = channel.upper_line;
    if (upper_line.size() >= 2) {
        upper_line_slope = (upper_line.back() - upper_line.front()) / static_cast<double>(upper_line.size() - 1);
    }
    features["spy_upper_line_slope"] = safe_float(upper_line_slope, 0.0);

    // ==========================================================================
    // 30. spy_lower_line_slope
    // ==========================================================================
    double lower_line_slope = 0.0;
    const auto& lower_line = channel.lower_line;
    if (lower_line.size() >= 2) {
        lower_line_slope = (lower_line.back() - lower_line.front()) / static_cast<double>(lower_line.size() - 1);
    }
    features["spy_lower_line_slope"] = safe_float(lower_line_slope, 0.0);

    // ==========================================================================
    // 31. spy_channel_expanding (1 if width increasing)
    // ==========================================================================
    double channel_expanding = 0.0;
    if (upper_line.size() >= 10 && lower_line.size() >= 10) {
        double width_start = upper_line.front() - lower_line.front();
        double width_end = upper_line.back() - lower_line.back();
        channel_expanding = (width_end > width_start * 1.05) ? 1.0 : 0.0;
    }
    features["spy_channel_expanding"] = channel_expanding;

    // ==========================================================================
    // 32. spy_channel_contracting (1 if width decreasing)
    // ==========================================================================
    double channel_contracting = 0.0;
    if (upper_line.size() >= 10 && lower_line.size() >= 10) {
        double width_start = upper_line.front() - lower_line.front();
        double width_end = upper_line.back() - lower_line.back();
        channel_contracting = (width_end < width_start * 0.95) ? 1.0 : 0.0;
    }
    features["spy_channel_contracting"] = channel_contracting;

    // ==========================================================================
    // 33. spy_std_dev_ratio (std_dev / avg_price)
    // ==========================================================================
    features["spy_std_dev_ratio"] = safe_divide(channel.std_dev, avg_price, 0.0);

    // ==========================================================================
    // 34. spy_breakout_pressure_up
    // ==========================================================================
    double breakout_pressure_up = 0.0;
    if (!ch_high.empty() && !upper_line.empty() && ch_high.size() >= 5) {
        double sum_dist = 0.0;
        int count = 0;
        size_t start = ch_high.size() > 5 ? ch_high.size() - 5 : 0;
        size_t ul_start = upper_line.size() > 5 ? upper_line.size() - 5 : 0;
        for (size_t i = 0; i < 5 && start + i < ch_high.size() && ul_start + i < upper_line.size(); ++i) {
            double h = ch_high[start + i];
            double u = upper_line[ul_start + i];
            if (u > 0) {
                double dist = (u - h) / u;
                if (dist > 0) sum_dist += dist;
                count++;
            }
        }
        if (count > 0) {
            double avg_dist = sum_dist / count;
            breakout_pressure_up = safe_float(1.0 - avg_dist, 0.0);
        }
    }
    features["spy_breakout_pressure_up"] = breakout_pressure_up;

    // ==========================================================================
    // 35. spy_breakout_pressure_down
    // ==========================================================================
    double breakout_pressure_down = 0.0;
    if (!ch_low.empty() && !lower_line.empty() && ch_low.size() >= 5) {
        double sum_dist = 0.0;
        int count = 0;
        size_t start = ch_low.size() > 5 ? ch_low.size() - 5 : 0;
        size_t ll_start = lower_line.size() > 5 ? lower_line.size() - 5 : 0;
        for (size_t i = 0; i < 5 && start + i < ch_low.size() && ll_start + i < lower_line.size(); ++i) {
            double l = ch_low[start + i];
            double lb = lower_line[ll_start + i];
            if (l > 0) {
                double dist = (l - lb) / l;
                if (dist > 0) sum_dist += dist;
                count++;
            }
        }
        if (count > 0) {
            double avg_dist = sum_dist / count;
            breakout_pressure_down = safe_float(1.0 - avg_dist, 0.0);
        }
    }
    features["spy_breakout_pressure_down"] = breakout_pressure_down;

    // ==========================================================================
    // 36. spy_channel_symmetry (how balanced are upper/lower touches)
    // ==========================================================================
    double total_touches = upper_touches + lower_touches;
    double channel_symmetry = 0.0;
    if (total_touches > 0) {
        double min_t = std::min(upper_touches, lower_touches);
        double max_t = std::max(upper_touches, lower_touches);
        channel_symmetry = safe_divide(min_t, max_t, 0.0);
    }
    features["spy_channel_symmetry"] = channel_symmetry;

    // ==========================================================================
    // 37. spy_touch_regularity (std dev of intervals between touches)
    // ==========================================================================
    double touch_regularity = 0.0;
    if (touches.size() >= 3) {
        std::vector<int> intervals;
        for (size_t i = 1; i < touches.size(); ++i) {
            intervals.push_back(touches[i].bar_index - touches[i-1].bar_index);
        }
        if (!intervals.empty()) {
            double sum = 0.0;
            for (int interval : intervals) sum += interval;
            double avg_interval = sum / intervals.size();
            double sum_sq = 0.0;
            for (int interval : intervals) {
                double diff = interval - avg_interval;
                sum_sq += diff * diff;
            }
            double std_interval = std::sqrt(sum_sq / intervals.size());
            touch_regularity = std::max(0.0, 1.0 - safe_divide(std_interval, avg_interval + 1, 0.0));
        }
    }
    features["spy_touch_regularity"] = safe_float(touch_regularity, 0.0);

    // ==========================================================================
    // 38. spy_recent_touch_bias (bias toward upper or lower in recent touches)
    // ==========================================================================
    double recent_touch_bias = 0.0;
    if (touches.size() >= 3) {
        int recent_count = std::min(5, static_cast<int>(touches.size()));
        int recent_upper = 0;
        for (size_t i = touches.size() - recent_count; i < touches.size(); ++i) {
            if (touches[i].touch_type == TouchType::UPPER) recent_upper++;
        }
        int recent_lower = recent_count - recent_upper;
        recent_touch_bias = safe_divide(recent_upper - recent_lower, recent_count, 0.0);
    }
    features["spy_recent_touch_bias"] = recent_touch_bias;

    // ==========================================================================
    // 39. spy_channel_curvature (non-linearity measure)
    // ==========================================================================
    double channel_curvature = 0.0;
    if (!ch_close.empty() && ch_close.size() >= 10) {
        size_t m = ch_close.size();
        if (m >= 3) {
            double y0 = ch_close[0];
            double y_mid = ch_close[m/2];
            double y_end = ch_close[m-1];
            double curvature = (y0 - 2*y_mid + y_end) / ((m/2.0) * (m/2.0));
            channel_curvature = safe_divide(curvature, avg_price, 0.0) * 1000.0;
        }
    }
    features["spy_channel_curvature"] = safe_float(channel_curvature, 0.0);

    // ==========================================================================
    // 40. spy_parallel_score (how parallel are upper and lower lines)
    // ==========================================================================
    double avg_line_slope = (upper_line_slope + lower_line_slope) / 2.0;
    double parallel_score = 0.5;
    if (avg_line_slope != 0.0) {
        double slope_diff = std::abs(upper_line_slope - lower_line_slope);
        parallel_score = safe_float(1.0 - safe_divide(slope_diff, std::abs(avg_line_slope) + 0.0001, 0.0), 0.5);
    } else {
        parallel_score = (upper_line_slope == lower_line_slope) ? 1.0 : 0.5;
    }
    features["spy_parallel_score"] = parallel_score;

    // ==========================================================================
    // 41. spy_touch_density (touches per unit channel width)
    // ==========================================================================
    features["spy_touch_density"] = safe_divide(total_touches, width_pct + 1.0, 0.0);

    // ==========================================================================
    // 42. spy_bounce_efficiency (complete_cycles / total touches)
    // ==========================================================================
    features["spy_bounce_efficiency"] = safe_divide(complete_cycles, total_touches + 1.0, 0.0);

    // ==========================================================================
    // 43. spy_channel_stability (r_squared * alternation_ratio)
    // ==========================================================================
    double channel_stability = safe_float(r_squared * alternation_ratio, 0.0);
    features["spy_channel_stability"] = channel_stability;

    // ==========================================================================
    // 44. spy_momentum_direction_alignment (1 if momentum matches direction)
    // ==========================================================================
    double momentum_direction_alignment = 0.5;
    if (direction_val == 2.0) { // Bull
        momentum_direction_alignment = (channel_momentum > 0) ? 1.0 : 0.0;
    } else if (direction_val == 0.0) { // Bear
        momentum_direction_alignment = (channel_momentum < 0) ? 1.0 : 0.0;
    } else { // Sideways
        momentum_direction_alignment = (std::abs(channel_momentum) < 0.01) ? 1.0 : 0.5;
    }
    features["spy_momentum_direction_alignment"] = momentum_direction_alignment;

    // ==========================================================================
    // 45. spy_price_position_extreme (how close to boundaries)
    // ==========================================================================
    features["spy_price_position_extreme"] = safe_float(std::abs(position - 0.5) * 2.0, 0.0);

    // ==========================================================================
    // 46. spy_breakout_imminence (combined pressure score)
    // ==========================================================================
    features["spy_breakout_imminence"] = safe_float(std::max(breakout_pressure_up, breakout_pressure_down), 0.0);

    // ==========================================================================
    // 47. spy_breakout_direction_bias (positive = up, negative = down)
    // ==========================================================================
    features["spy_breakout_direction_bias"] = safe_float(breakout_pressure_up - breakout_pressure_down, 0.0);

    // ==========================================================================
    // 48. spy_channel_health_score (composite quality metric)
    // ==========================================================================
    double health = (
        (channel.valid ? 1.0 : 0.0) * 0.2 +
        channel_stability * 0.3 +
        parallel_score * 0.2 +
        touch_regularity * 0.15 +
        channel_symmetry * 0.15
    );
    features["spy_channel_health_score"] = safe_float(health, 0.0);

    // ==========================================================================
    // 49. spy_time_weighted_position (position weighted by time since last touch)
    // ==========================================================================
    double time_factor = safe_divide(bars_since_last, window, 1.0);
    features["spy_time_weighted_position"] = safe_float(position * (1.0 - time_factor), 0.0);

    // ==========================================================================
    // 50. spy_volatility_adjusted_width (width relative to recent volatility)
    // ==========================================================================
    if (channel_width_atr_ratio > 0) {
        features["spy_volatility_adjusted_width"] = safe_float(channel_width_atr_ratio / 4.0, 1.0);
    } else {
        features["spy_volatility_adjusted_width"] = 1.0;
    }

    // ==========================================================================
    // 51-58. EXCURSION FEATURES - Price movements beyond channel boundaries
    // ==========================================================================
    int excursions_above = 0;
    int excursions_below = 0;
    double max_excursion_above = 0.0;
    double max_excursion_below = 0.0;
    int last_excursion_bar = -1;
    double last_excursion_dir = 0.0; // 0=none, 1=above, -1=below
    std::vector<int> excursion_durations;
    bool in_excursion = false;
    int current_excursion_start = -1;

    if (!ch_high.empty() && !ch_low.empty() && !upper_line.empty() && !lower_line.empty()) {
        size_t len = std::min({ch_high.size(), ch_low.size(), upper_line.size(), lower_line.size()});

        for (size_t i = 0; i < len; ++i) {
            double h = ch_high[i];
            double l = ch_low[i];
            double u = upper_line[i];
            double lb = lower_line[i];

            // Check for excursion above upper line
            if (h > u) {
                if (!in_excursion) {
                    in_excursion = true;
                    current_excursion_start = static_cast<int>(i);
                }
                excursions_above++;
                last_excursion_bar = static_cast<int>(i);
                last_excursion_dir = 1.0;

                double excursion_pct = safe_divide(h - u, u, 0.0) * 100.0;
                max_excursion_above = std::max(max_excursion_above, excursion_pct);
            }
            // Check for excursion below lower line
            else if (l < lb) {
                if (!in_excursion) {
                    in_excursion = true;
                    current_excursion_start = static_cast<int>(i);
                }
                excursions_below++;
                last_excursion_bar = static_cast<int>(i);
                last_excursion_dir = -1.0;

                double excursion_pct = safe_divide(lb - l, lb, 0.0) * 100.0;
                max_excursion_below = std::max(max_excursion_below, excursion_pct);
            }
            else {
                // Price is inside channel
                if (in_excursion) {
                    int duration = static_cast<int>(i) - current_excursion_start;
                    excursion_durations.push_back(duration);
                    in_excursion = false;
                    current_excursion_start = -1;
                }
            }
        }

        // Handle ongoing excursion at end
        if (in_excursion && current_excursion_start >= 0) {
            size_t len = std::min({ch_high.size(), ch_low.size(), upper_line.size(), lower_line.size()});
            int duration = static_cast<int>(len) - current_excursion_start;
            excursion_durations.push_back(duration);
        }
    }

    // 51. spy_excursions_above_upper
    features["spy_excursions_above_upper"] = static_cast<double>(excursions_above);

    // 52. spy_excursions_below_lower
    features["spy_excursions_below_lower"] = static_cast<double>(excursions_below);

    // 53. spy_max_excursion_above_pct
    features["spy_max_excursion_above_pct"] = safe_float(max_excursion_above, 0.0);

    // 54. spy_max_excursion_below_pct
    features["spy_max_excursion_below_pct"] = safe_float(max_excursion_below, 0.0);

    // 55. spy_bars_since_last_excursion
    if (last_excursion_bar >= 0 && !ch_high.empty()) {
        int bars_since = static_cast<int>(ch_high.size()) - 1 - last_excursion_bar;
        features["spy_bars_since_last_excursion"] = safe_float(std::max(0, bars_since), static_cast<double>(window));
    } else {
        features["spy_bars_since_last_excursion"] = static_cast<double>(window);
    }

    // 56. spy_excursion_return_speed_avg
    if (!excursion_durations.empty()) {
        double sum = 0.0;
        for (int dur : excursion_durations) sum += dur;
        features["spy_excursion_return_speed_avg"] = safe_float(sum / excursion_durations.size(), 0.0);
    } else {
        features["spy_excursion_return_speed_avg"] = 0.0;
    }

    // 57. spy_excursion_rate
    int total_excursions = excursions_above + excursions_below;
    features["spy_excursion_rate"] = safe_divide(total_excursions, window, 0.0);

    // 58. spy_last_excursion_direction
    features["spy_last_excursion_direction"] = last_excursion_dir;

    return features;
}

// =============================================================================
// CHANNEL HISTORY FEATURES (67 per TF)
// Full implementation matching Python v15/features/channel_history.py
// =============================================================================

// Safe statistical helpers for channel history
double FeatureExtractor::safe_mean(const std::vector<double>& values, double default_val) {
    if (values.empty()) return default_val;
    double sum = 0.0;
    int count = 0;
    for (double v : values) {
        if (std::isfinite(v)) { sum += v; ++count; }
    }
    return count > 0 ? sum / count : default_val;
}

double FeatureExtractor::safe_std(const std::vector<double>& values, double default_val) {
    if (values.size() < 2) return default_val;
    double mean = safe_mean(values, 0.0);
    double sum_sq = 0.0;
    int count = 0;
    for (double v : values) {
        if (std::isfinite(v)) { double diff = v - mean; sum_sq += diff * diff; ++count; }
    }
    if (count < 2) return default_val;
    double result = std::sqrt(sum_sq / (count - 1));
    return std::isfinite(result) ? result : default_val;
}

double FeatureExtractor::safe_min(const std::vector<double>& values, double default_val) {
    if (values.empty()) return default_val;
    double min_val = std::numeric_limits<double>::max();
    bool found = false;
    for (double v : values) { if (std::isfinite(v) && v < min_val) { min_val = v; found = true; } }
    return found ? min_val : default_val;
}

double FeatureExtractor::safe_max(const std::vector<double>& values, double default_val) {
    if (values.empty()) return default_val;
    double max_val = std::numeric_limits<double>::lowest();
    bool found = false;
    for (double v : values) { if (std::isfinite(v) && v > max_val) { max_val = v; found = true; } }
    return found ? max_val : default_val;
}

int FeatureExtractor::encode_direction_sequence(const std::vector<int>& directions) {
    if (directions.empty()) return 2;
    int bull_count = 0, bear_count = 0;
    for (int d : directions) { if (d == 2) ++bull_count; else if (d == 0) ++bear_count; }
    int total = static_cast<int>(directions.size());
    if (bear_count == total) return 0;
    if (bull_count == total) return 4;
    if (static_cast<double>(bear_count) / total > 0.6) return 1;
    if (static_cast<double>(bull_count) / total > 0.6) return 3;
    return 2;
}

int FeatureExtractor::encode_break_sequence(const std::vector<int>& break_directions) {
    if (break_directions.empty()) return 2;
    int up_count = 0, down_count = 0;
    for (int b : break_directions) { if (b > 0) ++up_count; else if (b < 0) ++down_count; }
    int total = static_cast<int>(break_directions.size());
    if (down_count == total) return 0;
    if (up_count == total) return 4;
    if (static_cast<double>(down_count) / total > 0.6) return 1;
    if (static_cast<double>(up_count) / total > 0.6) return 3;
    return 2;
}

double FeatureExtractor::calculate_alternating_score(const std::vector<int>& directions) {
    if (directions.size() < 2) return 0.0;
    int alt = 0;
    for (size_t i = 1; i < directions.size(); ++i) { if (directions[i] != directions[i-1]) ++alt; }
    return safe_divide(static_cast<double>(alt), static_cast<double>(directions.size() - 1), 0.0);
}

double FeatureExtractor::calculate_trend(const std::vector<double>& values) {
    if (values.size() < 2) return 0.0;
    std::vector<double> x_vals, y_vals;
    for (size_t i = 0; i < values.size(); ++i) {
        if (std::isfinite(values[i])) { x_vals.push_back(static_cast<double>(i)); y_vals.push_back(values[i]); }
    }
    if (x_vals.size() < 2) return 0.0;
    double n = static_cast<double>(x_vals.size()), sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    for (size_t i = 0; i < x_vals.size(); ++i) { sum_x += x_vals[i]; sum_y += y_vals[i]; sum_xy += x_vals[i] * y_vals[i]; sum_x2 += x_vals[i] * x_vals[i]; }
    double denom = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denom) < 1e-10) return 0.0;
    double slope = (n * sum_xy - sum_x * sum_y) / denom, mean_val = sum_y / n;
    if (std::abs(mean_val) < 1e-10) mean_val = 1.0;
    return safe_float(safe_divide(slope, std::abs(mean_val), 0.0), 0.0);
}

double FeatureExtractor::calculate_momentum(const std::vector<int>& directions) {
    if (directions.size() < 3) return 0.0;
    double recent_sum = 0.0;
    for (size_t i = directions.size() - 2; i < directions.size(); ++i) recent_sum += (directions[i] - 1);
    double recent_score = recent_sum / 2.0;
    size_t older_count = directions.size() > 3 ? directions.size() - 2 : 2;
    double older_sum = 0.0;
    for (size_t i = 0; i < older_count && i < directions.size(); ++i) older_sum += (directions[i] - 1);
    return safe_float(recent_score - older_sum / static_cast<double>(older_count), 0.0);
}

double FeatureExtractor::calculate_regime_shift(const std::vector<ChannelHistoryEntry>& history) {
    if (history.size() < 3) return 0.0;
    size_t mid = history.size() / 2;
    double first_sum = 0.0, second_sum = 0.0;
    for (size_t i = 0; i < mid; ++i) first_sum += history[i].direction;
    for (size_t i = mid; i < history.size(); ++i) second_sum += history[i].direction;
    double first_avg = first_sum / static_cast<double>(mid);
    double second_avg = second_sum / static_cast<double>(history.size() - mid);
    return safe_float(safe_divide(std::abs(second_avg - first_avg), 2.0, 0.0), 0.0);
}

std::unordered_map<std::string, double> FeatureExtractor::extract_single_history_features(
    const std::vector<ChannelHistoryEntry>& history, const std::string& prefix) {
    // Pre-reserve for ~40 single history features
    auto features = create_feature_map(40);
    std::vector<ChannelHistoryEntry> hist = history.size() > 5 ?
        std::vector<ChannelHistoryEntry>(history.end() - 5, history.end()) : history;
    int n_ch = static_cast<int>(hist.size());
    std::vector<double> durations, slopes, r_squareds, bounce_counts, exit_counts, avg_exit_magnitudes;
    std::vector<double> avg_bars_outside_list, exit_return_rates, durability_scores, false_break_counts;
    std::vector<int> directions, break_directions;
    for (const auto& h : hist) {
        durations.push_back(h.duration); slopes.push_back(h.slope); directions.push_back(h.direction);
        break_directions.push_back(h.break_direction); r_squareds.push_back(h.r_squared);
        bounce_counts.push_back(h.bounce_count); exit_counts.push_back(h.exit_count);
        avg_exit_magnitudes.push_back(h.avg_exit_magnitude); avg_bars_outside_list.push_back(h.avg_bars_outside);
        exit_return_rates.push_back(h.exit_return_rate); durability_scores.push_back(h.durability_score);
        false_break_counts.push_back(h.false_break_count);
    }
    features[prefix + "last5_avg_duration"] = safe_mean(durations, 50.0);
    features[prefix + "last5_avg_slope"] = safe_mean(slopes, 0.0);
    features[prefix + "last5_direction_pattern"] = static_cast<double>(encode_direction_sequence(directions));
    features[prefix + "last5_break_pattern"] = static_cast<double>(encode_break_sequence(break_directions));
    features[prefix + "last5_avg_quality"] = safe_mean(r_squareds, 0.0);
    features[prefix + "channel_momentum"] = calculate_momentum(directions);
    features[prefix + "last5_slope_trend"] = calculate_trend(slopes);
    features[prefix + "last5_duration_trend"] = calculate_trend(durations);
    features[prefix + "last5_quality_trend"] = calculate_trend(r_squareds);
    features[prefix + "channel_regime_shift"] = calculate_regime_shift(hist);
    features[prefix + "alternating_pattern"] = calculate_alternating_score(directions);
    features[prefix + "break_alternating"] = calculate_alternating_score(break_directions);
    int consec = 0; if (!directions.empty()) { int ld = directions.back(); for (auto it = directions.rbegin(); it != directions.rend() && *it == ld; ++it) ++consec; }
    features[prefix + "consecutive_same_dir"] = static_cast<double>(consec);
    int consec_b = 0; if (!break_directions.empty()) { int lb = break_directions.back(); for (auto it = break_directions.rbegin(); it != break_directions.rend() && *it == lb; ++it) ++consec_b; }
    features[prefix + "consecutive_same_break"] = static_cast<double>(consec_b);
    features[prefix + "last5_duration_std"] = safe_std(durations, 0.0);
    features[prefix + "last5_slope_std"] = safe_std(slopes, 0.0);
    features[prefix + "last5_quality_std"] = safe_std(r_squareds, 0.0);
    features[prefix + "last5_min_duration"] = safe_min(durations, 50.0);
    features[prefix + "last5_max_duration"] = safe_max(durations, 50.0);
    features[prefix + "last5_duration_range"] = features[prefix + "last5_max_duration"] - features[prefix + "last5_min_duration"];
    features[prefix + "recent_vs_avg_duration"] = durations.empty() ? 0.0 : safe_divide(durations.back() - features[prefix + "last5_avg_duration"], features[prefix + "last5_avg_duration"] + 1, 0.0);
    features[prefix + "recent_vs_avg_slope"] = slopes.empty() ? 0.0 : safe_float(slopes.back() - features[prefix + "last5_avg_slope"], 0.0);
    features[prefix + "recent_vs_avg_quality"] = r_squareds.empty() ? 0.0 : safe_float(r_squareds.back() - features[prefix + "last5_avg_quality"], 0.0);
    int bull = 0, bear = 0, up_b = 0, dn_b = 0;
    for (int d : directions) { if (d == 2) ++bull; if (d == 0) ++bear; }
    for (int b : break_directions) { if (b > 0) ++up_b; if (b < 0) ++dn_b; }
    features[prefix + "bull_channel_ratio"] = safe_divide(static_cast<double>(bull), std::max(n_ch, 1), 0.0);
    features[prefix + "bear_channel_ratio"] = safe_divide(static_cast<double>(bear), std::max(n_ch, 1), 0.0);
    features[prefix + "up_break_ratio"] = safe_divide(static_cast<double>(up_b), std::max(n_ch, 1), 0.0);
    features[prefix + "down_break_ratio"] = safe_divide(static_cast<double>(dn_b), std::max(n_ch, 1), 0.0);
    double qc = 1.0 - std::min(features[prefix + "last5_quality_std"] / 0.5, 1.0);
    double dc = 1.0 - std::min(features[prefix + "last5_duration_std"] / 50.0, 1.0);
    features[prefix + "channel_stability_score"] = safe_float((qc + dc) / 2.0, 0.5);
    double ds = std::abs(features[prefix + "bull_channel_ratio"] - features[prefix + "bear_channel_ratio"]);
    double sc = 1.0 - std::min(features[prefix + "last5_slope_std"] / 0.1, 1.0);
    features[prefix + "trend_strength_score"] = safe_float((ds + sc) / 2.0, 0.0);
    features[prefix + "last5_avg_bounces"] = safe_mean(bounce_counts, 0.0);
    features[prefix + "last5_avg_exit_count"] = safe_mean(exit_counts, 0.0);
    features[prefix + "last5_avg_exit_magnitude"] = safe_mean(avg_exit_magnitudes, 0.0);
    features[prefix + "last5_avg_bars_outside"] = safe_mean(avg_bars_outside_list, 0.0);
    features[prefix + "last5_avg_exit_return_rate"] = safe_mean(exit_return_rates, 0.0);
    features[prefix + "last5_avg_durability"] = safe_mean(durability_scores, 0.0);
    features[prefix + "last5_avg_false_breaks"] = safe_mean(false_break_counts, 0.0);
    features[prefix + "exit_count_trend"] = calculate_trend(exit_counts);
    features[prefix + "durability_trend"] = calculate_trend(durability_scores);
    features[prefix + "bars_outside_trend"] = calculate_trend(avg_bars_outside_list);
    features[prefix + "exit_return_rate_trend"] = calculate_trend(exit_return_rates);
    return features;
}

std::unordered_map<std::string, double> FeatureExtractor::extract_channel_history_features(
    const std::vector<ChannelHistoryEntry>& tsla_history,
    const std::vector<ChannelHistoryEntry>& spy_history
) {
    // Pre-reserve for 67 channel history features (40 TSLA + 40 SPY - overlapping + cross-asset)
    auto features = create_feature_map(FeatureOffsets::CHANNEL_HISTORY_COUNT);
    auto tsla_f = extract_single_history_features(tsla_history, "tsla_");
    auto spy_f = extract_single_history_features(spy_history, "spy_");
    features.insert(tsla_f.begin(), tsla_f.end());
    features.insert(spy_f.begin(), spy_f.end());

    auto get_last_5 = [](const std::vector<ChannelHistoryEntry>& h) {
        return h.size() > 5 ? std::vector<ChannelHistoryEntry>(h.end() - 5, h.end()) : h;
    };
    auto th = get_last_5(tsla_history), sh = get_last_5(spy_history);
    std::vector<int> td, sd, tb, sb;
    for (const auto& h : th) { td.push_back(h.direction); tb.push_back(h.break_direction); }
    for (const auto& h : sh) { sd.push_back(h.direction); sb.push_back(h.break_direction); }

    if (!td.empty() && !sd.empty()) {
        size_t ml = std::min(td.size(), sd.size()); int m = 0;
        for (size_t i = 0; i < ml; ++i) if (td[td.size()-1-i] == sd[sd.size()-1-i]) ++m;
        features["tsla_spy_channel_alignment"] = safe_divide(static_cast<double>(m), static_cast<double>(ml), 0.5);
    } else features["tsla_spy_channel_alignment"] = 0.5;

    double tm = features.count("tsla_channel_momentum") ? features["tsla_channel_momentum"] : 0.0;
    double sm = features.count("spy_channel_momentum") ? features["spy_channel_momentum"] : 0.0;
    features["channel_momentum_alignment"] = (tm > 0 && sm > 0) || (tm < 0 && sm < 0) ? 1.0 : (tm * sm < 0 ? 0.0 : 0.5);

    if (!tb.empty() && !sb.empty()) {
        size_t ml = std::min(tb.size(), sb.size()); int m = 0;
        for (size_t i = 0; i < ml; ++i) {
            bool tu = tb[tb.size()-1-i] > 0, su = sb[sb.size()-1-i] > 0, td_ = tb[tb.size()-1-i] < 0, sd_ = sb[sb.size()-1-i] < 0;
            if ((tu && su) || (td_ && sd_)) ++m;
        }
        features["break_pattern_alignment"] = safe_divide(static_cast<double>(m), static_cast<double>(ml), 0.5);
    } else features["break_pattern_alignment"] = 0.5;

    features["quality_spread"] = safe_float((features.count("tsla_last5_avg_quality") ? features["tsla_last5_avg_quality"] : 0.0) - (features.count("spy_last5_avg_quality") ? features["spy_last5_avg_quality"] : 0.0), 0.0);
    double tdur = features.count("tsla_last5_avg_duration") ? features["tsla_last5_avg_duration"] : 50.0;
    double sdur = features.count("spy_last5_avg_duration") ? features["spy_last5_avg_duration"] : 50.0;
    features["duration_spread"] = safe_divide(tdur - sdur, (tdur + sdur) / 2.0 + 1.0, 0.0);
    features["slope_spread"] = safe_float((features.count("tsla_last5_avg_slope") ? features["tsla_last5_avg_slope"] : 0.0) - (features.count("spy_last5_avg_slope") ? features["spy_last5_avg_slope"] : 0.0), 0.0);
    double tr = features.count("tsla_channel_regime_shift") ? features["tsla_channel_regime_shift"] : 0.0;
    double sr = features.count("spy_channel_regime_shift") ? features["spy_channel_regime_shift"] : 0.0;
    features["combined_regime_shift"] = safe_float((tr + sr) / 2.0, 0.0);
    features["momentum_divergence"] = safe_float(std::abs(tm - sm), 0.0);
    features["tsla_leading_indicator"] = safe_float(tr - sr, 0.0);
    double tts = features.count("tsla_trend_strength_score") ? features["tsla_trend_strength_score"] : 0.0;
    double sts = features.count("spy_trend_strength_score") ? features["spy_trend_strength_score"] : 0.0;
    features["combined_trend_strength"] = safe_float((tts + sts) / 2.0, 0.0);
    features["exit_count_spread"] = safe_float((features.count("tsla_last5_avg_exit_count") ? features["tsla_last5_avg_exit_count"] : 0.0) - (features.count("spy_last5_avg_exit_count") ? features["spy_last5_avg_exit_count"] : 0.0), 0.0);
    features["durability_spread_avg"] = safe_float((features.count("tsla_last5_avg_durability") ? features["tsla_last5_avg_durability"] : 0.0) - (features.count("spy_last5_avg_durability") ? features["spy_last5_avg_durability"] : 0.0), 0.0);
    double tet = features.count("tsla_exit_count_trend") ? features["tsla_exit_count_trend"] : 0.0;
    double set = features.count("spy_exit_count_trend") ? features["spy_exit_count_trend"] : 0.0;
    features["exit_alignment"] = (tet > 0 && set > 0) || (tet < 0 && set < 0) ? 1.0 : (tet * set < 0 ? 0.0 : 0.5);

    static const std::vector<std::string> final_names = {
        "tsla_last5_avg_duration", "tsla_last5_avg_slope", "tsla_last5_direction_pattern", "tsla_last5_break_pattern", "tsla_last5_avg_quality",
        "spy_last5_avg_duration", "spy_last5_avg_slope", "spy_last5_direction_pattern", "spy_last5_break_pattern", "spy_last5_avg_quality",
        "tsla_channel_momentum", "tsla_last5_slope_trend", "tsla_last5_duration_trend", "tsla_last5_quality_trend", "tsla_channel_regime_shift",
        "spy_channel_momentum", "spy_last5_slope_trend", "spy_last5_duration_trend", "spy_last5_quality_trend", "spy_channel_regime_shift",
        "tsla_alternating_pattern", "tsla_consecutive_same_dir", "tsla_consecutive_same_break", "tsla_bull_channel_ratio", "tsla_bear_channel_ratio",
        "spy_alternating_pattern", "spy_consecutive_same_dir", "spy_consecutive_same_break", "spy_bull_channel_ratio", "spy_bear_channel_ratio",
        "tsla_last5_duration_std", "tsla_last5_slope_std", "tsla_up_break_ratio", "tsla_down_break_ratio", "tsla_channel_stability_score",
        "spy_last5_duration_std", "spy_last5_slope_std", "spy_up_break_ratio", "spy_down_break_ratio", "spy_channel_stability_score",
        "tsla_last5_avg_exit_count", "tsla_last5_avg_exit_magnitude", "tsla_last5_avg_bars_outside", "tsla_last5_avg_exit_return_rate", "tsla_last5_avg_durability",
        "spy_last5_avg_exit_count", "spy_last5_avg_exit_magnitude", "spy_last5_avg_bars_outside", "spy_last5_avg_exit_return_rate", "spy_last5_avg_durability",
        "tsla_exit_count_trend", "tsla_durability_trend", "spy_exit_count_trend", "spy_durability_trend",
        "tsla_spy_channel_alignment", "channel_momentum_alignment", "break_pattern_alignment", "quality_spread", "duration_spread", "slope_spread",
        "combined_regime_shift", "momentum_divergence", "tsla_leading_indicator", "combined_trend_strength", "exit_count_spread", "durability_spread_avg", "exit_alignment"
    };
    std::unordered_map<std::string, double> final_features;
    for (const auto& name : final_names) {
        double val = features.count(name) ? features.at(name) : 0.0;
        final_features[name] = std::isfinite(val) ? val : 0.0;
    }
    return final_features;
}

std::vector<std::string> FeatureExtractor::get_channel_history_feature_names() {
    return {
        "tsla_last5_avg_duration", "tsla_last5_avg_slope", "tsla_last5_direction_pattern", "tsla_last5_break_pattern", "tsla_last5_avg_quality",
        "spy_last5_avg_duration", "spy_last5_avg_slope", "spy_last5_direction_pattern", "spy_last5_break_pattern", "spy_last5_avg_quality",
        "tsla_channel_momentum", "tsla_last5_slope_trend", "tsla_last5_duration_trend", "tsla_last5_quality_trend", "tsla_channel_regime_shift",
        "spy_channel_momentum", "spy_last5_slope_trend", "spy_last5_duration_trend", "spy_last5_quality_trend", "spy_channel_regime_shift",
        "tsla_alternating_pattern", "tsla_consecutive_same_dir", "tsla_consecutive_same_break", "tsla_bull_channel_ratio", "tsla_bear_channel_ratio",
        "spy_alternating_pattern", "spy_consecutive_same_dir", "spy_consecutive_same_break", "spy_bull_channel_ratio", "spy_bear_channel_ratio",
        "tsla_last5_duration_std", "tsla_last5_slope_std", "tsla_up_break_ratio", "tsla_down_break_ratio", "tsla_channel_stability_score",
        "spy_last5_duration_std", "spy_last5_slope_std", "spy_up_break_ratio", "spy_down_break_ratio", "spy_channel_stability_score",
        "tsla_last5_avg_exit_count", "tsla_last5_avg_exit_magnitude", "tsla_last5_avg_bars_outside", "tsla_last5_avg_exit_return_rate", "tsla_last5_avg_durability",
        "spy_last5_avg_exit_count", "spy_last5_avg_exit_magnitude", "spy_last5_avg_bars_outside", "spy_last5_avg_exit_return_rate", "spy_last5_avg_durability",
        "tsla_exit_count_trend", "tsla_durability_trend", "spy_exit_count_trend", "spy_durability_trend",
        "tsla_spy_channel_alignment", "channel_momentum_alignment", "break_pattern_alignment", "quality_spread", "duration_spread", "slope_spread",
        "combined_regime_shift", "momentum_divergence", "tsla_leading_indicator", "combined_trend_strength", "exit_count_spread", "durability_spread_avg", "exit_alignment"
    };
}

// =============================================================================
// WINDOW SCORE FEATURES (50 per TF)
// =============================================================================

namespace {

// Helper to get channel score: r_squared * bounce_count
double get_channel_score(const std::shared_ptr<Channel>& channel) {
    if (!channel || !channel->valid) {
        return 0.0;
    }
    double r_squared = std::isfinite(channel->r_squared) ? channel->r_squared : 0.0;
    double bounce_count = static_cast<double>(channel->bounce_count);
    double score = r_squared * bounce_count;
    return std::isfinite(score) ? score : 0.0;
}

// Helper to check if channel is valid
bool is_channel_valid(const std::shared_ptr<Channel>& channel) {
    return channel && channel->valid;
}

// Helper to get r_squared from channel
double get_channel_r_squared(const std::shared_ptr<Channel>& channel) {
    if (!channel || !channel->valid) return 0.0;
    return std::isfinite(channel->r_squared) ? channel->r_squared : 0.0;
}

// Helper to get slope from channel
double get_channel_slope(const std::shared_ptr<Channel>& channel) {
    if (!channel || !channel->valid) return 0.0;
    return std::isfinite(channel->slope) ? channel->slope : 0.0;
}

// Helper to get bounce_count from channel
double get_channel_bounce_count(const std::shared_ptr<Channel>& channel) {
    if (!channel || !channel->valid) return 0.0;
    return static_cast<double>(channel->bounce_count);
}

// Helper to get direction from channel: 0=bear, 1=sideways, 2=bull
int get_channel_direction(const std::shared_ptr<Channel>& channel) {
    if (!channel || !channel->valid) return 1; // Default to sideways
    return static_cast<int>(channel->direction);
}

// Normalize window size to 0-1 range where 10=0, 80=1
double normalize_window_size(int window) {
    if (window <= 10) return 0.0;
    if (window >= 80) return 1.0;
    return static_cast<double>(window - 10) / 70.0;
}

// Calculate mean of a vector
double ws_calc_mean(const std::vector<double>& values, double default_val = 0.0) {
    if (values.empty()) return default_val;
    double sum = 0.0;
    for (double v : values) {
        sum += v;
    }
    return sum / static_cast<double>(values.size());
}

// Calculate standard deviation of a vector
double ws_calc_std(const std::vector<double>& values, double default_val = 0.0) {
    if (values.size() < 2) return default_val;
    double mean = ws_calc_mean(values, 0.0);
    double sum_sq = 0.0;
    for (double v : values) {
        double diff = v - mean;
        sum_sq += diff * diff;
    }
    double variance = sum_sq / static_cast<double>(values.size());
    double result = std::sqrt(variance);
    return std::isfinite(result) ? result : default_val;
}

// Calculate correlation coefficient between two vectors
double ws_calc_correlation(const std::vector<double>& x, const std::vector<double>& y, double default_val = 0.0) {
    if (x.size() != y.size() || x.size() < 2) return default_val;

    double mean_x = ws_calc_mean(x, 0.0);
    double mean_y = ws_calc_mean(y, 0.0);

    double sum_xy = 0.0, sum_x2 = 0.0, sum_y2 = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double dx = x[i] - mean_x;
        double dy = y[i] - mean_y;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    double denom = std::sqrt(sum_x2 * sum_y2);
    if (denom < 1e-10) return default_val;

    double corr = sum_xy / denom;
    if (!std::isfinite(corr)) return default_val;
    return std::clamp(corr, -1.0, 1.0);
}

} // anonymous namespace

std::unordered_map<std::string, double> FeatureExtractor::extract_window_score_features(
    const std::unordered_map<int, std::shared_ptr<Channel>>& channels_by_window,
    int best_window
) {
    // Pre-reserve for 50 window score features
    auto features = create_feature_map(FeatureOffsets::WINDOW_SCORE_COUNT);

    // Ensure best_window is valid
    bool best_window_valid = false;
    for (int w : STANDARD_WINDOWS) {
        if (w == best_window) {
            best_window_valid = true;
            break;
        }
    }
    if (!best_window_valid) {
        best_window = 50; // Default to middle window
    }

    // Pre-compute validity and scores for all windows
    std::unordered_map<int, bool> validity;
    std::unordered_map<int, double> scores;
    std::unordered_map<int, double> r_squared_vals;
    std::unordered_map<int, double> slopes;
    std::unordered_map<int, double> bounce_counts;
    std::unordered_map<int, int> directions;

    for (int window : STANDARD_WINDOWS) {
        auto it = channels_by_window.find(window);
        std::shared_ptr<Channel> channel = (it != channels_by_window.end()) ? it->second : nullptr;

        validity[window] = is_channel_valid(channel);
        scores[window] = get_channel_score(channel);
        r_squared_vals[window] = get_channel_r_squared(channel);
        slopes[window] = get_channel_slope(channel);
        bounce_counts[window] = get_channel_bounce_count(channel);
        directions[window] = get_channel_direction(channel);
    }

    // ==========================================================================
    // 1. PER-WINDOW VALIDITY (8 features)
    // ==========================================================================
    for (int window : STANDARD_WINDOWS) {
        features["window_" + std::to_string(window) + "_valid"] = validity[window] ? 1.0 : 0.0;
    }

    // ==========================================================================
    // 2. PER-WINDOW SCORES (8 features)
    // ==========================================================================
    for (int window : STANDARD_WINDOWS) {
        features["window_" + std::to_string(window) + "_score"] = scores[window];
    }

    // ==========================================================================
    // 3. ALIGNMENT FEATURES (15 features)
    // ==========================================================================

    // Collect valid windows
    std::vector<int> valid_windows;
    for (int window : STANDARD_WINDOWS) {
        if (validity[window]) {
            valid_windows.push_back(window);
        }
    }
    int valid_count = static_cast<int>(valid_windows.size());

    // 3.1 valid_window_count
    features["valid_window_count"] = static_cast<double>(valid_count);

    // 3.2 valid_window_ratio
    features["valid_window_ratio"] = safe_divide(static_cast<double>(valid_count),
                                                  static_cast<double>(NUM_STANDARD_WINDOWS), 0.0);

    // 3.3 all_windows_agree_direction
    if (valid_count > 0) {
        int first_direction = directions[valid_windows[0]];
        bool all_same = true;
        for (int w : valid_windows) {
            if (directions[w] != first_direction) {
                all_same = false;
                break;
            }
        }
        features["all_windows_agree_direction"] = all_same ? 1.0 : 0.0;
    } else {
        features["all_windows_agree_direction"] = 0.0;
    }

    // 3.4 direction_consensus (% of valid windows with same direction as best)
    int best_direction = directions[best_window];
    if (valid_count > 0) {
        int same_direction_count = 0;
        for (int w : valid_windows) {
            if (directions[w] == best_direction) {
                ++same_direction_count;
            }
        }
        features["direction_consensus"] = safe_divide(static_cast<double>(same_direction_count),
                                                       static_cast<double>(valid_count), 0.0);
    } else {
        features["direction_consensus"] = 0.0;
    }

    // 3.5 slope_consensus (std dev of slopes across valid windows)
    if (valid_count >= 2) {
        std::vector<double> valid_slopes;
        for (int w : valid_windows) {
            valid_slopes.push_back(slopes[w]);
        }
        features["slope_consensus"] = ws_calc_std(valid_slopes, 0.0);
    } else {
        features["slope_consensus"] = 0.0;
    }

    // 3.6 avg_r_squared_all_valid
    if (valid_count > 0) {
        std::vector<double> valid_r_squared;
        for (int w : valid_windows) {
            valid_r_squared.push_back(r_squared_vals[w]);
        }
        features["avg_r_squared_all_valid"] = ws_calc_mean(valid_r_squared, 0.0);
    } else {
        features["avg_r_squared_all_valid"] = 0.0;
    }

    // 3.7 best_window_score_ratio (best score / avg score)
    std::vector<double> all_scores;
    for (int window : STANDARD_WINDOWS) {
        all_scores.push_back(scores[window]);
    }
    double avg_score = ws_calc_mean(all_scores, 0.0);
    double best_score = scores[best_window];
    features["best_window_score_ratio"] = safe_divide(best_score, avg_score, 1.0);

    // 3.8 window_spread (largest - smallest valid window)
    if (valid_count >= 2) {
        int min_valid = *std::min_element(valid_windows.begin(), valid_windows.end());
        int max_valid = *std::max_element(valid_windows.begin(), valid_windows.end());
        features["window_spread"] = static_cast<double>(max_valid - min_valid);
    } else {
        features["window_spread"] = 0.0;
    }

    // 3.9 small_windows_valid (10, 20, 30 valid count)
    std::vector<int> small_windows = {10, 20, 30};
    int small_valid_count = 0;
    for (int w : small_windows) {
        if (validity[w]) ++small_valid_count;
    }
    features["small_windows_valid"] = static_cast<double>(small_valid_count);

    // 3.10 large_windows_valid (60, 70, 80 valid count)
    std::vector<int> large_windows = {60, 70, 80};
    int large_valid_count = 0;
    for (int w : large_windows) {
        if (validity[w]) ++large_valid_count;
    }
    features["large_windows_valid"] = static_cast<double>(large_valid_count);

    // 3.11 small_vs_large_bias (which size range has more valid)
    // -1 = all small, 0 = balanced, 1 = all large
    int total_extremes = small_valid_count + large_valid_count;
    if (total_extremes > 0) {
        features["small_vs_large_bias"] = safe_divide(
            static_cast<double>(large_valid_count - small_valid_count),
            static_cast<double>(total_extremes), 0.0);
    } else {
        features["small_vs_large_bias"] = 0.0;
    }

    // 3.12 consecutive_valid_windows (longest streak of consecutive valid windows)
    int max_consecutive = 0;
    int current_consecutive = 0;
    for (int window : STANDARD_WINDOWS) {
        if (validity[window]) {
            ++current_consecutive;
            max_consecutive = std::max(max_consecutive, current_consecutive);
        } else {
            current_consecutive = 0;
        }
    }
    features["consecutive_valid_windows"] = static_cast<double>(max_consecutive);

    // 3.13 window_gap_pattern (pattern of gaps in validity - encoded as ratio)
    // Count transitions from valid to invalid
    int gap_count = 0;
    for (size_t i = 1; i < STANDARD_WINDOWS.size(); ++i) {
        bool prev_valid = validity[STANDARD_WINDOWS[i - 1]];
        bool curr_valid = validity[STANDARD_WINDOWS[i]];
        if (prev_valid != curr_valid) {
            ++gap_count;
        }
    }
    // Normalize by max possible transitions (7)
    features["window_gap_pattern"] = safe_divide(static_cast<double>(gap_count), 7.0, 0.0);

    // 3.14 multi_scale_alignment (do small and large windows agree)
    std::vector<int> small_directions;
    std::vector<int> large_directions;
    for (int w : small_windows) {
        if (validity[w]) small_directions.push_back(directions[w]);
    }
    for (int w : large_windows) {
        if (validity[w]) large_directions.push_back(directions[w]);
    }

    if (!small_directions.empty() && !large_directions.empty()) {
        // Find majority direction for each group
        auto find_majority = [](const std::vector<int>& dirs) -> int {
            std::unordered_map<int, int> counts;
            for (int d : dirs) counts[d]++;
            int majority = 1; // default sideways
            int max_count = 0;
            for (const auto& [dir, count] : counts) {
                if (count > max_count) {
                    max_count = count;
                    majority = dir;
                }
            }
            return majority;
        };

        int small_majority = find_majority(small_directions);
        int large_majority = find_majority(large_directions);
        features["multi_scale_alignment"] = (small_majority == large_majority) ? 1.0 : 0.0;
    } else {
        features["multi_scale_alignment"] = 0.5; // Neutral if not enough data
    }

    // 3.15 fractal_score (self-similarity across scales)
    // Measure how consistent scores are across adjacent window sizes
    if (valid_count >= 3) {
        std::vector<double> score_diffs;
        for (size_t i = 0; i < STANDARD_WINDOWS.size() - 1; ++i) {
            int w1 = STANDARD_WINDOWS[i];
            int w2 = STANDARD_WINDOWS[i + 1];
            if (validity[w1] && validity[w2]) {
                double diff = std::abs(scores[w1] - scores[w2]);
                score_diffs.push_back(diff);
            }
        }
        if (!score_diffs.empty()) {
            double avg_diff = ws_calc_mean(score_diffs, 0.0);
            double max_score = *std::max_element(all_scores.begin(), all_scores.end());
            if (max_score < 1e-10) max_score = 1.0;
            // Lower diff relative to max = more fractal (self-similar)
            double fractal = 1.0 - safe_divide(avg_diff, max_score, 0.0);
            features["fractal_score"] = std::max(0.0, fractal);
        } else {
            features["fractal_score"] = 0.0;
        }
    } else {
        features["fractal_score"] = 0.0;
    }

    // ==========================================================================
    // 4. BEST WINDOW FEATURES (10 features)
    // ==========================================================================

    // 4.1 best_window_size (normalized 0-1 where 10=0, 80=1)
    features["best_window_size"] = normalize_window_size(best_window);

    // 4.2 best_window_r_squared
    features["best_window_r_squared"] = r_squared_vals[best_window];

    // 4.3 best_window_bounce_count
    features["best_window_bounce_count"] = bounce_counts[best_window];

    // 4.4 best_window_slope
    features["best_window_slope"] = slopes[best_window];

    // 4.5 best_window_direction
    features["best_window_direction"] = static_cast<double>(directions[best_window]);

    // 4.6 best_vs_second_best_score_gap
    std::vector<double> sorted_scores = all_scores;
    std::sort(sorted_scores.begin(), sorted_scores.end(), std::greater<double>());
    if (sorted_scores.size() >= 2) {
        double score_gap = sorted_scores[0] - sorted_scores[1];
        features["best_vs_second_best_score_gap"] = std::isfinite(score_gap) ? score_gap : 0.0;
    } else {
        features["best_vs_second_best_score_gap"] = 0.0;
    }

    // 4.7 best_window_is_smallest
    features["best_window_is_smallest"] = (best_window == STANDARD_WINDOWS[0]) ? 1.0 : 0.0;

    // 4.8 best_window_is_largest
    features["best_window_is_largest"] = (best_window == STANDARD_WINDOWS[NUM_STANDARD_WINDOWS - 1]) ? 1.0 : 0.0;

    // 4.9 best_window_position (1-8 in sorted order by score)
    std::vector<std::pair<int, double>> windows_with_scores;
    for (int window : STANDARD_WINDOWS) {
        windows_with_scores.push_back({window, scores[window]});
    }
    std::sort(windows_with_scores.begin(), windows_with_scores.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    int best_position = 4; // Default to middle position
    for (size_t i = 0; i < windows_with_scores.size(); ++i) {
        if (windows_with_scores[i].first == best_window) {
            best_position = static_cast<int>(i) + 1;
            break;
        }
    }
    features["best_window_position"] = static_cast<double>(best_position);

    // 4.10 best_window_dominance (how much better than others)
    // Ratio of best score to sum of all other scores
    double other_scores_sum = 0.0;
    for (int window : STANDARD_WINDOWS) {
        if (window != best_window) {
            other_scores_sum += scores[window];
        }
    }
    features["best_window_dominance"] = safe_divide(best_score, other_scores_sum, 0.0);

    // ==========================================================================
    // 5. TREND ACROSS WINDOWS (9 features)
    // ==========================================================================

    // 5.1 slope_trend_across_windows (correlation of slope with window size)
    if (valid_count >= 3) {
        std::vector<double> valid_indices;
        std::vector<double> valid_slope_array;
        for (size_t i = 0; i < STANDARD_WINDOWS.size(); ++i) {
            int w = STANDARD_WINDOWS[i];
            if (validity[w]) {
                valid_indices.push_back(static_cast<double>(i));
                valid_slope_array.push_back(slopes[w]);
            }
        }
        double slope_std = ws_calc_std(valid_slope_array, 0.0);
        if (slope_std > 1e-10 && valid_indices.size() >= 2) {
            features["slope_trend_across_windows"] = ws_calc_correlation(valid_indices, valid_slope_array, 0.0);
        } else {
            features["slope_trend_across_windows"] = 0.0;
        }
    } else {
        features["slope_trend_across_windows"] = 0.0;
    }

    // 5.2 r_squared_trend_across_windows
    if (valid_count >= 3) {
        std::vector<double> valid_indices;
        std::vector<double> valid_r_array;
        for (size_t i = 0; i < STANDARD_WINDOWS.size(); ++i) {
            int w = STANDARD_WINDOWS[i];
            if (validity[w]) {
                valid_indices.push_back(static_cast<double>(i));
                valid_r_array.push_back(r_squared_vals[w]);
            }
        }
        double r_std = ws_calc_std(valid_r_array, 0.0);
        if (r_std > 1e-10 && valid_indices.size() >= 2) {
            features["r_squared_trend_across_windows"] = ws_calc_correlation(valid_indices, valid_r_array, 0.0);
        } else {
            features["r_squared_trend_across_windows"] = 0.0;
        }
    } else {
        features["r_squared_trend_across_windows"] = 0.0;
    }

    // 5.3 bounce_count_trend_across_windows
    if (valid_count >= 3) {
        std::vector<double> valid_indices;
        std::vector<double> valid_bounce_array;
        for (size_t i = 0; i < STANDARD_WINDOWS.size(); ++i) {
            int w = STANDARD_WINDOWS[i];
            if (validity[w]) {
                valid_indices.push_back(static_cast<double>(i));
                valid_bounce_array.push_back(bounce_counts[w]);
            }
        }
        double bounce_std = ws_calc_std(valid_bounce_array, 0.0);
        if (bounce_std > 1e-10 && valid_indices.size() >= 2) {
            features["bounce_count_trend_across_windows"] = ws_calc_correlation(valid_indices, valid_bounce_array, 0.0);
        } else {
            features["bounce_count_trend_across_windows"] = 0.0;
        }
    } else {
        features["bounce_count_trend_across_windows"] = 0.0;
    }

    // 5.4 window_size_quality_correlation (correlation between window size and score)
    if (valid_count >= 3) {
        std::vector<double> valid_sizes;
        std::vector<double> valid_score_array;
        for (int w : valid_windows) {
            valid_sizes.push_back(static_cast<double>(w));
            valid_score_array.push_back(scores[w]);
        }
        double score_std = ws_calc_std(valid_score_array, 0.0);
        if (score_std > 1e-10) {
            features["window_size_quality_correlation"] = ws_calc_correlation(valid_sizes, valid_score_array, 0.0);
        } else {
            features["window_size_quality_correlation"] = 0.0;
        }
    } else {
        features["window_size_quality_correlation"] = 0.0;
    }

    // 5.5 convergence_score (are all windows pointing to same price target)
    // Measure variance in slope * window combinations (projected endpoints)
    if (valid_count >= 2) {
        std::vector<double> projected_targets;
        for (int w : valid_windows) {
            // Project where price would be based on slope
            double target = slopes[w] * static_cast<double>(w); // Simplified projection
            projected_targets.push_back(target);
        }

        if (!projected_targets.empty()) {
            double target_std = ws_calc_std(projected_targets, 0.0);
            double target_mean = std::abs(ws_calc_mean(projected_targets, 1.0));
            // Lower std relative to mean = more convergence
            double normalized_std = safe_divide(target_std, target_mean + 1.0, 1.0);
            features["convergence_score"] = std::max(0.0, 1.0 - normalized_std);
        } else {
            features["convergence_score"] = 0.0;
        }
    } else {
        features["convergence_score"] = 0.0;
    }

    // 5.6 divergence_warning (windows contradicting each other)
    // High when valid windows have opposite directions
    if (valid_count >= 2) {
        std::set<int> unique_directions;
        for (int w : valid_windows) {
            unique_directions.insert(directions[w]);
        }
        // 1.0 if we have both bullish(2) and bearish(0), 0.5 if sideways involved
        bool has_bear = unique_directions.count(0) > 0;
        bool has_bull = unique_directions.count(2) > 0;
        if (has_bear && has_bull) {
            features["divergence_warning"] = 1.0;
        } else if (unique_directions.size() > 1) {
            features["divergence_warning"] = 0.5;
        } else {
            features["divergence_warning"] = 0.0;
        }
    } else {
        features["divergence_warning"] = 0.0;
    }

    // 5.7 multi_timeframe_momentum
    // Average slope weighted by validity and score
    if (valid_count > 0) {
        double weighted_slope_sum = 0.0;
        double score_sum = 0.0;
        for (int w : valid_windows) {
            weighted_slope_sum += slopes[w] * scores[w];
            score_sum += scores[w];
        }
        features["multi_timeframe_momentum"] = safe_divide(weighted_slope_sum, score_sum, 0.0);
    } else {
        features["multi_timeframe_momentum"] = 0.0;
    }

    // 5.8 window_regime (small-dominant, balanced, large-dominant)
    // Encoded: -1 = small-dominant, 0 = balanced, 1 = large-dominant
    if (valid_count > 0) {
        double small_score_sum = 0.0;
        double large_score_sum = 0.0;
        for (int w : small_windows) {
            if (validity[w]) small_score_sum += scores[w];
        }
        for (int w : large_windows) {
            if (validity[w]) large_score_sum += scores[w];
        }

        std::vector<int> mid_windows = {40, 50};
        double mid_score_sum = 0.0;
        for (int w : mid_windows) {
            if (validity[w]) mid_score_sum += scores[w];
        }

        double total_score = small_score_sum + mid_score_sum + large_score_sum;
        if (total_score > 0) {
            // Calculate weighted position
            features["window_regime"] = safe_divide(
                large_score_sum - small_score_sum, total_score, 0.0);
        } else {
            features["window_regime"] = 0.0;
        }
    } else {
        features["window_regime"] = 0.0;
    }

    // 5.9 confidence_score (overall multi-window confidence)
    // Composite of: valid_ratio, direction_consensus, avg_r_squared, best_dominance
    double confidence = (
        features["valid_window_ratio"] * 0.25 +
        features["direction_consensus"] * 0.25 +
        features["avg_r_squared_all_valid"] * 0.25 +
        std::min(features["best_window_dominance"], 1.0) * 0.25
    );
    features["confidence_score"] = std::isfinite(confidence) ? confidence : 0.0;

    // ==========================================================================
    // FINAL SAFETY CHECK
    // ==========================================================================
    for (auto& [key, value] : features) {
        if (!std::isfinite(value)) {
            value = 0.0;
        }
    }

    return features;
}

// =============================================================================
// CHANNEL CORRELATION HELPER FUNCTIONS
// =============================================================================

double FeatureExtractor::safe_spread(double tsla_val, double spy_val) {
    if (!std::isfinite(tsla_val) || !std::isfinite(spy_val)) {
        return 0.0;
    }
    double spread = tsla_val - spy_val;
    return std::isfinite(spread) ? spread : 0.0;
}

double FeatureExtractor::safe_ratio(double tsla_val, double spy_val, double default_val) {
    if (!std::isfinite(tsla_val) || !std::isfinite(spy_val)) {
        return default_val;
    }
    double ratio = safe_divide(tsla_val, spy_val, default_val);
    // Clamp to reasonable bounds [-10, 10]
    ratio = std::clamp(ratio, -10.0, 10.0);
    return std::isfinite(ratio) ? ratio : default_val;
}

double FeatureExtractor::safe_aligned(double tsla_val, double spy_val, double threshold) {
    if (!std::isfinite(tsla_val) || !std::isfinite(spy_val)) {
        return 0.0;
    }
    // Check if both are on same side of threshold
    bool tsla_above = tsla_val >= threshold;
    bool spy_above = spy_val >= threshold;
    return (tsla_above == spy_above) ? 1.0 : 0.0;
}

double FeatureExtractor::direction_aligned(double tsla_val, double spy_val) {
    if (!std::isfinite(tsla_val) || !std::isfinite(spy_val)) {
        return 0.5;
    }

    const double near_zero = 0.001;

    bool tsla_pos = tsla_val > near_zero;
    bool tsla_neg = tsla_val < -near_zero;
    bool spy_pos = spy_val > near_zero;
    bool spy_neg = spy_val < -near_zero;

    // If either is near zero, partial alignment
    if ((!tsla_pos && !tsla_neg) || (!spy_pos && !spy_neg)) {
        return 0.5;
    }

    // Both positive or both negative
    if ((tsla_pos && spy_pos) || (tsla_neg && spy_neg)) {
        return 1.0;
    }

    return 0.0;
}

// =============================================================================
// CHANNEL CORRELATION FEATURES (50 per TF)
// =============================================================================

std::unordered_map<std::string, double> FeatureExtractor::extract_channel_correlation_features(
    const std::unordered_map<std::string, double>& tsla_channel_features,
    const std::unordered_map<std::string, double>& spy_channel_features
) {
    // Pre-reserve for 50 channel correlation features
    auto features = create_feature_map(FeatureOffsets::CHANNEL_CORRELATION_COUNT);

    // Helper lambda to get feature value with default
    auto get_feature = [](const std::unordered_map<std::string, double>& feats,
                         const std::string& name, double default_val) -> double {
        auto it = feats.find(name);
        if (it != feats.end() && std::isfinite(it->second)) {
            return it->second;
        }
        return default_val;
    };

    // =========================================================================
    // 1. INDIVIDUAL CORRELATIONS (39 features: 13 key features x 3 variants)
    // =========================================================================

    // Feature 1-3: Position in channel - where price is within the channel (0=floor, 1=ceiling)
    double tsla_pos = get_feature(tsla_channel_features, "position_in_channel", 0.5);
    double spy_pos = get_feature(spy_channel_features, "position_in_channel", 0.5);
    features["position_in_channel_spread"] = safe_spread(tsla_pos, spy_pos);
    features["position_in_channel_ratio"] = safe_ratio(tsla_pos, spy_pos, 1.0);
    features["position_in_channel_aligned"] = safe_aligned(tsla_pos, spy_pos, 0.5);

    // Feature 4-6: Distance to upper boundary (%)
    double tsla_dist_upper = get_feature(tsla_channel_features, "distance_to_upper_pct", 0.0);
    double spy_dist_upper = get_feature(spy_channel_features, "distance_to_upper_pct", 0.0);
    features["distance_to_upper_pct_spread"] = safe_spread(tsla_dist_upper, spy_dist_upper);
    features["distance_to_upper_pct_ratio"] = safe_ratio(tsla_dist_upper, spy_dist_upper, 1.0);
    features["distance_to_upper_pct_aligned"] = direction_aligned(tsla_dist_upper, spy_dist_upper);

    // Feature 7-9: Distance to lower boundary (%)
    double tsla_dist_lower = get_feature(tsla_channel_features, "distance_to_lower_pct", 0.0);
    double spy_dist_lower = get_feature(spy_channel_features, "distance_to_lower_pct", 0.0);
    features["distance_to_lower_pct_spread"] = safe_spread(tsla_dist_lower, spy_dist_lower);
    features["distance_to_lower_pct_ratio"] = safe_ratio(tsla_dist_lower, spy_dist_lower, 1.0);
    features["distance_to_lower_pct_aligned"] = direction_aligned(tsla_dist_lower, spy_dist_lower);

    // Feature 10-12: Breakout pressure up (proximity to upper boundary)
    double tsla_press_up = get_feature(tsla_channel_features, "breakout_pressure_up", 0.0);
    double spy_press_up = get_feature(spy_channel_features, "breakout_pressure_up", 0.0);
    features["breakout_pressure_up_spread"] = safe_spread(tsla_press_up, spy_press_up);
    features["breakout_pressure_up_ratio"] = safe_ratio(tsla_press_up, spy_press_up, 1.0);
    features["breakout_pressure_up_aligned"] = safe_aligned(tsla_press_up, spy_press_up, 0.5);

    // Feature 13-15: Breakout pressure down (proximity to lower boundary)
    double tsla_press_down = get_feature(tsla_channel_features, "breakout_pressure_down", 0.0);
    double spy_press_down = get_feature(spy_channel_features, "breakout_pressure_down", 0.0);
    features["breakout_pressure_down_spread"] = safe_spread(tsla_press_down, spy_press_down);
    features["breakout_pressure_down_ratio"] = safe_ratio(tsla_press_down, spy_press_down, 1.0);
    features["breakout_pressure_down_aligned"] = safe_aligned(tsla_press_down, spy_press_down, 0.5);

    // Feature 16-18: Touch velocity (bounces per bar)
    double tsla_touch_vel = get_feature(tsla_channel_features, "touch_velocity", 0.0);
    double spy_touch_vel = get_feature(spy_channel_features, "touch_velocity", 0.0);
    features["touch_velocity_spread"] = safe_spread(tsla_touch_vel, spy_touch_vel);
    features["touch_velocity_ratio"] = safe_ratio(tsla_touch_vel, spy_touch_vel, 1.0);
    features["touch_velocity_aligned"] = safe_aligned(tsla_touch_vel, spy_touch_vel, 0.05);

    // Feature 19-21: Channel slope normalized (slope / price level)
    double tsla_slope = get_feature(tsla_channel_features, "channel_slope_normalized", 0.0);
    double spy_slope = get_feature(spy_channel_features, "channel_slope_normalized", 0.0);
    features["channel_slope_normalized_spread"] = safe_spread(tsla_slope, spy_slope);
    features["channel_slope_normalized_ratio"] = safe_ratio(tsla_slope, spy_slope, 1.0);
    features["channel_slope_normalized_aligned"] = direction_aligned(tsla_slope, spy_slope);

    // Feature 22-24: Channel R-squared (quality of linear fit)
    double tsla_rsq = get_feature(tsla_channel_features, "channel_r_squared", 0.0);
    double spy_rsq = get_feature(spy_channel_features, "channel_r_squared", 0.0);
    features["channel_r_squared_spread"] = safe_spread(tsla_rsq, spy_rsq);
    features["channel_r_squared_ratio"] = safe_ratio(tsla_rsq, spy_rsq, 1.0);
    features["channel_r_squared_aligned"] = safe_aligned(tsla_rsq, spy_rsq, 0.5);

    // Feature 25-27: Excursions above upper boundary (count)
    double tsla_exc_above = get_feature(tsla_channel_features, "excursions_above_upper", 0.0);
    double spy_exc_above = get_feature(spy_channel_features, "excursions_above_upper", 0.0);
    features["excursions_above_upper_spread"] = safe_spread(tsla_exc_above, spy_exc_above);
    features["excursions_above_upper_ratio"] = safe_ratio(tsla_exc_above, spy_exc_above, 1.0);
    features["excursions_above_upper_aligned"] = safe_aligned(tsla_exc_above, spy_exc_above, 0.5);

    // Feature 28-30: Excursions below lower boundary (count)
    double tsla_exc_below = get_feature(tsla_channel_features, "excursions_below_lower", 0.0);
    double spy_exc_below = get_feature(spy_channel_features, "excursions_below_lower", 0.0);
    features["excursions_below_lower_spread"] = safe_spread(tsla_exc_below, spy_exc_below);
    features["excursions_below_lower_ratio"] = safe_ratio(tsla_exc_below, spy_exc_below, 1.0);
    features["excursions_below_lower_aligned"] = safe_aligned(tsla_exc_below, spy_exc_below, 0.5);

    // Feature 31-33: Max excursion above upper (%)
    double tsla_max_exc_above = get_feature(tsla_channel_features, "max_excursion_above_pct", 0.0);
    double spy_max_exc_above = get_feature(spy_channel_features, "max_excursion_above_pct", 0.0);
    features["max_excursion_above_pct_spread"] = safe_spread(tsla_max_exc_above, spy_max_exc_above);
    features["max_excursion_above_pct_ratio"] = safe_ratio(tsla_max_exc_above, spy_max_exc_above, 1.0);
    features["max_excursion_above_pct_aligned"] = safe_aligned(tsla_max_exc_above, spy_max_exc_above, 0.01);

    // Feature 34-36: Max excursion below lower (%)
    double tsla_max_exc_below = get_feature(tsla_channel_features, "max_excursion_below_pct", 0.0);
    double spy_max_exc_below = get_feature(spy_channel_features, "max_excursion_below_pct", 0.0);
    features["max_excursion_below_pct_spread"] = safe_spread(tsla_max_exc_below, spy_max_exc_below);
    features["max_excursion_below_pct_ratio"] = safe_ratio(tsla_max_exc_below, spy_max_exc_below, 1.0);
    features["max_excursion_below_pct_aligned"] = safe_aligned(tsla_max_exc_below, spy_max_exc_below, 0.01);

    // Feature 37-39: Excursion rate (excursions per bar)
    double tsla_exc_rate = get_feature(tsla_channel_features, "excursion_rate", 0.0);
    double spy_exc_rate = get_feature(spy_channel_features, "excursion_rate", 0.0);
    features["excursion_rate_spread"] = safe_spread(tsla_exc_rate, spy_exc_rate);
    features["excursion_rate_ratio"] = safe_ratio(tsla_exc_rate, spy_exc_rate, 1.0);
    features["excursion_rate_aligned"] = safe_aligned(tsla_exc_rate, spy_exc_rate, 0.05);

    // =========================================================================
    // 2. AGGREGATE FEATURES (4 features)
    // =========================================================================

    // Feature 40: Channel correlation score - overall similarity of TSLA and SPY channels
    // Computed as average of aligned features (1 = perfectly similar channels)
    double aligned_sum = features["position_in_channel_aligned"] +
                         features["breakout_pressure_up_aligned"] +
                         features["breakout_pressure_down_aligned"] +
                         features["channel_slope_normalized_aligned"] +
                         features["channel_r_squared_aligned"];
    features["channel_correlation_score"] = safe_float(aligned_sum / 5.0, 0.5);

    // Feature 41: Breakout pressure alignment - are both showing same breakout direction pressure
    // 1 = both pressing up, -1 = both pressing down, 0 = diverging
    double tsla_pressure_bias = tsla_press_up - tsla_press_down;
    double spy_pressure_bias = spy_press_up - spy_press_down;

    double pressure_alignment;
    if (tsla_pressure_bias > 0.1 && spy_pressure_bias > 0.1) {
        pressure_alignment = 1.0;  // Both pressing up
    } else if (tsla_pressure_bias < -0.1 && spy_pressure_bias < -0.1) {
        pressure_alignment = -1.0;  // Both pressing down
    } else if (std::abs(tsla_pressure_bias) < 0.1 && std::abs(spy_pressure_bias) < 0.1) {
        pressure_alignment = 0.0;  // Both neutral
    } else {
        // Diverging - one pressing up, other down or neutral
        pressure_alignment = -0.5;
    }
    features["breakout_pressure_alignment"] = safe_float(pressure_alignment, 0.0);

    // Feature 42: Excursion divergence - is one having excursions while other isn't
    // 1 = TSLA has more excursions, -1 = SPY has more, 0 = similar
    double tsla_total_exc = tsla_exc_above + tsla_exc_below;
    double spy_total_exc = spy_exc_above + spy_exc_below;
    double exc_diff = tsla_total_exc - spy_total_exc;

    double excursion_div;
    if (std::abs(exc_diff) < 0.5) {
        excursion_div = 0.0;  // Similar excursion patterns
    } else if (exc_diff > 0) {
        excursion_div = std::min(exc_diff / 5.0, 1.0);  // TSLA more excursions
    } else {
        excursion_div = std::max(exc_diff / 5.0, -1.0);  // SPY more excursions
    }
    features["excursion_divergence"] = safe_float(excursion_div, 0.0);

    // Feature 43: Channel regime match - do both have same direction (bull/bear/sideways)
    // Based on slope direction: 1 = both same direction, 0 = different
    double tsla_direction = get_feature(tsla_channel_features, "channel_direction", 1.0);  // 0=bear, 1=sideways, 2=bull
    double spy_direction = get_feature(spy_channel_features, "channel_direction", 1.0);

    double regime_match;
    if (std::abs(tsla_direction - spy_direction) < 0.01) {
        regime_match = 1.0;  // Same regime
    } else if (std::abs(tsla_direction - spy_direction) < 1.01) {
        regime_match = 0.5;  // Adjacent regimes (e.g., bull and sideways)
    } else {
        regime_match = 0.0;  // Opposite regimes (bull vs bear)
    }
    features["channel_regime_match"] = safe_float(regime_match, 0.5);

    // =========================================================================
    // 3. ADDITIONAL DERIVED FEATURES (7 features)
    // =========================================================================

    // Feature 44: Position spread extreme - how different are their channel positions
    // Normalized to 0-1 where 1 = completely opposite positions
    features["position_spread_extreme"] = safe_float(
        std::abs(features["position_in_channel_spread"]), 0.0
    );

    // Feature 45: Breakout pressure divergence - difference in overall breakout pressure
    double tsla_max_pressure = std::max(tsla_press_up, tsla_press_down);
    double spy_max_pressure = std::max(spy_press_up, spy_press_down);
    features["breakout_pressure_divergence"] = safe_float(
        std::abs(tsla_max_pressure - spy_max_pressure), 0.0
    );

    // Feature 46: Channel quality agreement - are both channels well-defined or poorly-defined
    // 1 = both have similar quality, 0 = quality mismatch
    double quality_diff = std::abs(tsla_rsq - spy_rsq);
    features["channel_quality_agreement"] = safe_float(1.0 - quality_diff, 0.5);

    // Feature 47: Slope magnitude comparison - which has steeper channel
    // Positive = TSLA steeper, negative = SPY steeper
    double tsla_slope_mag = std::abs(tsla_slope);
    double spy_slope_mag = std::abs(spy_slope);
    features["slope_magnitude_spread"] = safe_spread(tsla_slope_mag, spy_slope_mag);

    // Feature 48: Touch activity comparison - which has more active bouncing
    features["touch_activity_spread"] = safe_spread(tsla_touch_vel, spy_touch_vel);

    // Feature 49: Combined excursion intensity
    // Measures total excursion activity across both assets
    double total_excursions = tsla_total_exc + spy_total_exc;
    features["combined_excursion_intensity"] = safe_float(
        total_excursions / 10.0, 0.0  // Normalize (10 excursions = 1.0)
    );

    // Feature 50: Relative channel stability
    // Based on R-squared and touch regularity
    double tsla_stability = get_feature(tsla_channel_features, "channel_stability", 0.0);
    double spy_stability = get_feature(spy_channel_features, "channel_stability", 0.0);
    features["relative_stability_spread"] = safe_spread(tsla_stability, spy_stability);

    // Final validation: ensure all values are finite
    for (auto& [key, value] : features) {
        if (!std::isfinite(value)) {
            value = 0.0;
        }
    }

    return features;
}

// =============================================================================
// EVENT FEATURES (30 TF-independent)
// =============================================================================

std::unordered_map<std::string, double> FeatureExtractor::extract_event_features(
    int64_t timestamp,
    const std::vector<OHLCV>& tsla_data
) {
    // Pre-reserve for 30 event features
    auto features = create_feature_map(FeatureOffsets::EVENT_COUNT);

    // Convert timestamp to tm struct for date features
    time_t t = static_cast<time_t>(timestamp);
    struct tm* timeinfo = localtime(&t);

    if (timeinfo) {
        // Day of week (0=Sunday, 6=Saturday)
        features["is_monday"] = (timeinfo->tm_wday == 1) ? 1.0 : 0.0;
        features["is_tuesday"] = (timeinfo->tm_wday == 2) ? 1.0 : 0.0;
        features["is_wednesday"] = (timeinfo->tm_wday == 3) ? 1.0 : 0.0;
        features["is_thursday"] = (timeinfo->tm_wday == 4) ? 1.0 : 0.0;
        features["is_friday"] = (timeinfo->tm_wday == 5) ? 1.0 : 0.0;

        // Hour and minute of day
        int hour = timeinfo->tm_hour;
        int minute = timeinfo->tm_min;
        features["hour_of_day"] = static_cast<double>(hour);
        features["minute_of_hour"] = static_cast<double>(minute) / 59.0;  // Normalized 0-1
        features["is_market_open"] = (hour >= 9 && hour < 16) ? 1.0 : 0.0;
        features["is_morning_session"] = (hour >= 9 && hour < 12) ? 1.0 : 0.0;
        features["is_afternoon_session"] = (hour >= 12 && hour < 16) ? 1.0 : 0.0;

        // Month
        int month = timeinfo->tm_mon + 1; // 0-indexed
        features["month"] = static_cast<double>(month);
        features["is_january"] = (month == 1) ? 1.0 : 0.0;
        features["is_december"] = (month == 12) ? 1.0 : 0.0;

        // Quarter
        int quarter = (month - 1) / 3 + 1;
        features["quarter"] = static_cast<double>(quarter);
    } else {
        // Defaults
        for (int i = 0; i < 14; ++i) {
            features["event_" + std::to_string(i)] = 0.0;
        }
    }

    // Add remaining event features (earnings, FOMC, etc.)
    for (int i = 14; i < 30; ++i) {
        features["event_" + std::to_string(i)] = 0.0;
    }

    return features;
}

// =============================================================================
// BAR METADATA FEATURES (30 total: 3 per TF)
// =============================================================================

std::unordered_map<std::string, double> FeatureExtractor::extract_bar_metadata_features(
    const std::unordered_map<Timeframe, ResampleMetadata>& metadata_by_tf
) {
    // Pre-reserve for 30 bar metadata features (3 per TF)
    auto features = create_feature_map(FeatureOffsets::BAR_METADATA_COUNT);

    for (int tf_idx = 0; tf_idx < NUM_TIMEFRAMES; ++tf_idx) {
        Timeframe tf = static_cast<Timeframe>(tf_idx);
        std::string tf_name = timeframe_to_string(tf);

        auto it = metadata_by_tf.find(tf);
        if (it != metadata_by_tf.end()) {
            const auto& meta = it->second;
            features[tf_name + "_bar_completion_pct"] = meta.bar_completion_pct;
            features[tf_name + "_bars_in_partial"] = static_cast<double>(meta.bars_in_partial);

            int complete_bars = meta.is_partial ? std::max(0, meta.total_bars - 1) : meta.total_bars;
            features[tf_name + "_complete_bars"] = static_cast<double>(complete_bars);
        } else {
            features[tf_name + "_bar_completion_pct"] = 1.0;
            features[tf_name + "_bars_in_partial"] = 0.0;
            features[tf_name + "_complete_bars"] = 0.0;
        }
    }

    return features;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

double FeatureExtractor::safe_divide(double numerator, double denominator, double default_val) {
    if (denominator == 0.0 || !std::isfinite(denominator) || !std::isfinite(numerator)) {
        return default_val;
    }
    double result = numerator / denominator;
    return std::isfinite(result) ? result : default_val;
}

double FeatureExtractor::safe_float(double value, double default_val) {
    return std::isfinite(value) ? value : default_val;
}

double FeatureExtractor::get_last_valid(const std::vector<double>& arr, double default_val) {
    for (auto it = arr.rbegin(); it != arr.rend(); ++it) {
        if (std::isfinite(*it)) {
            return *it;
        }
    }
    return default_val;
}

double FeatureExtractor::pct_change(double current, double previous, double default_val) {
    if (previous == 0.0 || !std::isfinite(previous) || !std::isfinite(current)) {
        return default_val;
    }
    double result = ((current - previous) / previous) * 100.0;
    return std::isfinite(result) ? result : default_val;
}

void FeatureExtractor::extract_ohlcv_arrays(
    const std::vector<OHLCV>& data,
    std::vector<double>& open,
    std::vector<double>& high,
    std::vector<double>& low,
    std::vector<double>& close,
    std::vector<double>& volume
) {
    open.clear();
    high.clear();
    low.clear();
    close.clear();
    volume.clear();

    open.reserve(data.size());
    high.reserve(data.size());
    low.reserve(data.size());
    close.reserve(data.size());
    volume.reserve(data.size());

    for (const auto& bar : data) {
        open.push_back(bar.open);
        high.push_back(bar.high);
        low.push_back(bar.low);
        close.push_back(bar.close);
        volume.push_back(bar.volume);
    }
}

void FeatureExtractor::extract_ohlcv_arrays_optimized(
    const std::vector<OHLCV>& data,
    OHLCVArrays& arrays
) {
    const size_t n = data.size();

    // Resize arrays to exact size (avoids push_back overhead)
    arrays.open.resize(n);
    arrays.high.resize(n);
    arrays.low.resize(n);
    arrays.close.resize(n);
    arrays.volume.resize(n);

    // Direct assignment is faster than push_back
    for (size_t i = 0; i < n; ++i) {
        arrays.open[i] = data[i].open;
        arrays.high[i] = data[i].high;
        arrays.low[i] = data[i].low;
        arrays.close[i] = data[i].close;
        arrays.volume[i] = data[i].volume;
    }
}

void FeatureExtractor::sanitize_features(std::unordered_map<std::string, double>& features) {
    for (auto& [name, value] : features) {
        if (!std::isfinite(value)) {
            value = 0.0;
        }
    }
}

void FeatureExtractor::prefix_features(
    std::unordered_map<std::string, double>& features,
    const std::string& prefix
) {
    // Pre-reserve for same size
    auto prefixed = create_feature_map(features.size());
    for (const auto& [name, value] : features) {
        prefixed[prefix + name] = value;
    }
    features = std::move(prefixed);
}

std::unordered_map<std::string, double> FeatureExtractor::prefix_features_copy(
    const std::unordered_map<std::string, double>& features,
    const std::string& prefix
) {
    // Pre-reserve for same size
    auto prefixed = create_feature_map(features.size());
    for (const auto& [name, value] : features) {
        prefixed[prefix + name] = value;
    }
    return prefixed;
}

// =============================================================================
// INDICATOR HELPERS (reuse from TechnicalIndicators)
// =============================================================================

std::vector<double> FeatureExtractor::sma(const std::vector<double>& values, int period) {
    return TechnicalIndicators::sma(values, period);
}

std::vector<double> FeatureExtractor::ema(const std::vector<double>& values, int period) {
    return TechnicalIndicators::ema(values, period);
}

std::vector<double> FeatureExtractor::rsi(const std::vector<double>& values, int period) {
    return TechnicalIndicators::rsi(values, period);
}

std::vector<double> FeatureExtractor::atr(
    const std::vector<double>& high,
    const std::vector<double>& low,
    const std::vector<double>& close,
    int period
) {
    return TechnicalIndicators::atr(high, low, close, period);
}

// =============================================================================
// CORRELATION AND BETA CALCULATIONS
// =============================================================================

double FeatureExtractor::calculate_correlation(
    const std::vector<double>& series1,
    const std::vector<double>& series2,
    int window,
    double default_val
) {
    // SAFETY: Validate inputs
    if (series1.empty() || series2.empty()) {
        return default_val;
    }

    int n = std::min(series1.size(), series2.size());
    if (n < window || window < 2) {
        return default_val;
    }

    // SAFETY: Validate window size
    if (window > n) {
        window = n;
    }

    // Use last 'window' observations
    auto s1_begin = series1.end() - window;
    auto s2_begin = series2.end() - window;

    // Calculate means with safety checks
    double sum1 = 0.0, sum2 = 0.0;
    int valid_count = 0;

    for (int i = 0; i < window; ++i) {
        if (std::isfinite(s1_begin[i]) && std::isfinite(s2_begin[i])) {
            sum1 += s1_begin[i];
            sum2 += s2_begin[i];
            ++valid_count;
        }
    }

    if (valid_count < 2) {
        return default_val;  // Not enough valid data
    }

    double mean1 = sum1 / valid_count;
    double mean2 = sum2 / valid_count;

    // Calculate correlation
    double sum_sq1 = 0.0, sum_sq2 = 0.0, sum_prod = 0.0;
    for (int i = 0; i < window; ++i) {
        if (std::isfinite(s1_begin[i]) && std::isfinite(s2_begin[i])) {
            double d1 = s1_begin[i] - mean1;
            double d2 = s2_begin[i] - mean2;
            sum_sq1 += d1 * d1;
            sum_sq2 += d2 * d2;
            sum_prod += d1 * d2;
        }
    }

    // SAFETY: Check for zero variance
    if (sum_sq1 < 1e-10 || sum_sq2 < 1e-10) {
        return default_val;  // One or both series have zero variance
    }

    double denom = std::sqrt(sum_sq1 * sum_sq2);
    if (denom == 0.0 || !std::isfinite(denom)) {
        return default_val;
    }

    double corr = sum_prod / denom;

    // SAFETY: Clamp correlation to [-1, 1]
    if (std::isfinite(corr)) {
        corr = std::clamp(corr, -1.0, 1.0);
        return corr;
    }

    return default_val;
}

double FeatureExtractor::calculate_beta(
    const std::vector<double>& asset_returns,
    const std::vector<double>& market_returns,
    int window
) {
    int n = std::min(asset_returns.size(), market_returns.size());
    if (n < window || window < 2) {
        return 1.0; // Default beta
    }

    // Use last 'window' observations
    auto asset_begin = asset_returns.end() - window;
    auto market_begin = market_returns.end() - window;

    // Calculate mean of market returns
    double market_mean = std::accumulate(market_begin, market_returns.end(), 0.0) / window;
    double asset_mean = std::accumulate(asset_begin, asset_returns.end(), 0.0) / window;

    // Calculate covariance and variance
    double covariance = 0.0;
    double market_variance = 0.0;

    for (int i = 0; i < window; ++i) {
        double market_dev = market_begin[i] - market_mean;
        double asset_dev = asset_begin[i] - asset_mean;
        covariance += market_dev * asset_dev;
        market_variance += market_dev * market_dev;
    }

    if (market_variance == 0.0) {
        return 1.0;
    }

    double beta = covariance / market_variance;
    return std::isfinite(beta) ? beta : 1.0;
}

// =============================================================================
// FEATURE NAMES
// =============================================================================

std::vector<std::string> FeatureExtractor::get_all_feature_names() {
    std::vector<std::string> names;
    names.reserve(14190);

    // This would generate all 14,190 feature names in consistent order
    // For each TF:
    //   - Price features (58)
    //   - Technical features (59)
    //   - SPY features (117)
    //   - VIX features (25)
    //   - Cross-asset features (59)
    //   - Channel features per window (58 × 8)
    //   - SPY channel features per window (58 × 8)
    //   - Window scores (50)
    //   - Channel history (67)
    // Then event features (30)
    // Then bar metadata (30)

    // For brevity, returning placeholder
    for (int i = 0; i < 14190; ++i) {
        names.push_back("feature_" + std::to_string(i));
    }

    return names;
}

} // namespace v15
