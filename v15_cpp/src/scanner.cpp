#include "scanner.hpp"
#include "label_generator.hpp"
#include "feature_extractor.hpp"
#include "serialization.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cmath>

namespace v15 {

// =============================================================================
// THREAD POOL IMPLEMENTATION
// =============================================================================

ThreadPool::ThreadPool(size_t workers)
    : stop_(false)
{
    if (workers == 0) {
        workers = std::thread::hardware_concurrency();
        if (workers == 0) workers = 4;  // Fallback
    }

    workers_.reserve(workers);
    for (size_t i = 0; i < workers; ++i) {
        workers_.emplace_back([this] { worker_thread(); });
    }
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::shutdown() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();

    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

void ThreadPool::worker_thread() {
    while (true) {
        std::function<void()> task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

            if (stop_ && tasks_.empty()) {
                return;
            }

            if (!tasks_.empty()) {
                task = std::move(tasks_.front());
                tasks_.pop();
            }
        }

        if (task) {
            task();
        }
    }
}

template<typename Func, typename... Args>
auto ThreadPool::submit(Func&& func, Args&&... args) -> std::future<typename std::invoke_result<Func, Args...>::type> {
    using return_type = typename std::invoke_result<Func, Args...>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<Func>(func), std::forward<Args>(args)...)
    );

    std::future<return_type> result = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("Cannot submit task to stopped ThreadPool");
        }
        tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();

    return result;
}

// =============================================================================
// SCANNER IMPLEMENTATION
// =============================================================================

Scanner::Scanner(const ScannerConfig& config)
    : config_(config)
    , shutdown_requested_(false)
{
}

Scanner::~Scanner() {
}

std::vector<ChannelSample> Scanner::scan(
    const std::vector<OHLCV>& tsla_df,
    const std::vector<OHLCV>& spy_df,
    const std::vector<OHLCV>& vix_df
) {
    shutdown_requested_ = false;
    stats_ = ScannerStats();  // Reset stats

    auto total_start = std::chrono::high_resolution_clock::now();

    size_t n_bars = tsla_df.size();

    // Validate alignment
    if (spy_df.size() != n_bars) {
        throw std::runtime_error("TSLA/SPY length mismatch! TSLA has " +
                                std::to_string(n_bars) + " bars, SPY has " +
                                std::to_string(spy_df.size()) + " bars.");
    }
    if (vix_df.size() != n_bars) {
        throw std::runtime_error("TSLA/VIX length mismatch! TSLA has " +
                                std::to_string(n_bars) + " bars, VIX has " +
                                std::to_string(vix_df.size()) + " bars.");
    }

    if (config_.verbose) {
        std::cout << "[VALIDATION] Input alignment OK: " << n_bars << " bars, timestamps aligned\n";
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "V15 Channel Scanner - CHANNEL-END SAMPLING Architecture\n";
        std::cout << std::string(60, '=') << "\n";
        std::cout << "  Workers: " << (config_.workers > 0 ? config_.workers : std::thread::hardware_concurrency()) << "\n";
        std::cout << "  Batch size: " << config_.batch_size << " channels\n";
        std::cout << "  Architecture: ONE sample per detected channel at channel END\n";
    }

    // =========================================================================
    // PASS 1: Pre-compute all channels
    // =========================================================================
    if (config_.verbose) {
        std::cout << "\n[PASS 1] Detecting all channels across dataset...\n";
        std::cout << "  Timeframes: 10 (" << TIMEFRAME_NAMES[0];
        for (int i = 1; i < NUM_TIMEFRAMES; ++i) {
            std::cout << ", " << TIMEFRAME_NAMES[i];
        }
        std::cout << ")\n";
        std::cout << "  Windows: " << NUM_STANDARD_WINDOWS << " (";
        for (int i = 0; i < NUM_STANDARD_WINDOWS; ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << STANDARD_WINDOWS[i];
        }
        std::cout << ")\n";
        std::cout << "  Channel detection step: " << config_.step << "\n";
    }

    auto pass1_start = std::chrono::high_resolution_clock::now();

    std::unordered_map<TFWindowKey, std::vector<Channel>, TFWindowKeyHash> tsla_channel_map;
    std::unordered_map<std::string, std::vector<OHLCV>> tsla_resampled_dfs;

    std::unordered_map<TFWindowKey, std::vector<Channel>, TFWindowKeyHash> spy_channel_map;
    std::unordered_map<std::string, std::vector<OHLCV>> spy_resampled_dfs;

    if (config_.verbose) {
        std::cout << "\n  [PASS 1] Detecting TSLA channels...\n";
    }
    auto tsla_detect_start = std::chrono::high_resolution_clock::now();
    detect_all_channels(tsla_df, "TSLA", tsla_channel_map, tsla_resampled_dfs);
    auto tsla_detect_end = std::chrono::high_resolution_clock::now();
    double tsla_detect_time = std::chrono::duration<double>(tsla_detect_end - tsla_detect_start).count();

    int tsla_channels = 0;
    for (const auto& pair : tsla_channel_map) {
        tsla_channels += pair.second.size();
    }
    stats_.tsla_channels_detected = tsla_channels;

    if (config_.verbose) {
        std::cout << "           Completed: " << tsla_channels << " channels detected in "
                  << std::fixed << std::setprecision(1) << tsla_detect_time << "s\n";
    }

    if (config_.verbose) {
        std::cout << "\n  [PASS 1] Detecting SPY channels...\n";
    }
    auto spy_detect_start = std::chrono::high_resolution_clock::now();
    detect_all_channels(spy_df, "SPY", spy_channel_map, spy_resampled_dfs);
    auto spy_detect_end = std::chrono::high_resolution_clock::now();
    double spy_detect_time = std::chrono::duration<double>(spy_detect_end - spy_detect_start).count();

    int spy_channels = 0;
    for (const auto& pair : spy_channel_map) {
        spy_channels += pair.second.size();
    }
    stats_.spy_channels_detected = spy_channels;

    if (config_.verbose) {
        std::cout << "           Completed: " << spy_channels << " channels detected in "
                  << std::fixed << std::setprecision(1) << spy_detect_time << "s\n";
    }

    auto pass1_end = std::chrono::high_resolution_clock::now();
    stats_.pass1_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pass1_end - pass1_start).count();

    if (config_.verbose) {
        std::cout << "\n  [PASS 1] Summary:\n";
        std::cout << "           TSLA: " << tsla_channels << " channels in "
                  << std::fixed << std::setprecision(1) << tsla_detect_time << "s\n";
        std::cout << "           SPY:  " << spy_channels << " channels in "
                  << std::fixed << std::setprecision(1) << spy_detect_time << "s\n";
        std::cout << "           Total: " << (tsla_channels + spy_channels) << " channels, Pass 1 time: "
                  << std::fixed << std::setprecision(1) << (stats_.pass1_duration_ms / 1000.0) << "s\n";
    }

    // =========================================================================
    // PASS 2: Generate labels at channel END positions
    // =========================================================================
    if (config_.verbose) {
        std::cout << "\n[PASS 2] Generating labels from channel maps...\n";
    }

    auto pass2_start = std::chrono::high_resolution_clock::now();

    SlimLabeledChannelMap tsla_slim_map;
    SlimLabeledChannelMap spy_slim_map;

    if (config_.verbose) {
        std::cout << "\n  Generating TSLA labels... (" << tsla_channels << " channels to process)\n";
    }
    auto tsla_label_start = std::chrono::high_resolution_clock::now();
    generate_all_labels(tsla_channel_map, tsla_resampled_dfs, tsla_slim_map);
    auto tsla_label_end = std::chrono::high_resolution_clock::now();
    double tsla_label_time = std::chrono::duration<double>(tsla_label_end - tsla_label_start).count();

    int tsla_labeled = 0;
    int tsla_valid = 0;
    for (const auto& pair : tsla_slim_map) {
        tsla_labeled += pair.second.size();
        for (const auto& ch : pair.second) {
            if (ch.labels.direction_valid) {
                ++tsla_valid;
            }
        }
    }
    stats_.tsla_labels_generated = tsla_labeled;
    stats_.tsla_labels_valid = tsla_valid;

    if (config_.verbose) {
        std::cout << "  TSLA complete: " << tsla_labeled << " labels generated in "
                  << std::fixed << std::setprecision(1) << tsla_label_time << "s (" << tsla_valid << " valid)\n";
    }

    if (config_.verbose) {
        std::cout << "\n  Generating SPY labels... (" << spy_channels << " channels to process)\n";
    }
    auto spy_label_start = std::chrono::high_resolution_clock::now();
    generate_all_labels(spy_channel_map, spy_resampled_dfs, spy_slim_map);
    auto spy_label_end = std::chrono::high_resolution_clock::now();
    double spy_label_time = std::chrono::duration<double>(spy_label_end - spy_label_start).count();

    int spy_labeled = 0;
    int spy_valid = 0;
    for (const auto& pair : spy_slim_map) {
        spy_labeled += pair.second.size();
        for (const auto& ch : pair.second) {
            if (ch.labels.direction_valid) {
                ++spy_valid;
            }
        }
    }
    stats_.spy_labels_generated = spy_labeled;
    stats_.spy_labels_valid = spy_valid;

    if (config_.verbose) {
        std::cout << "  SPY complete: " << spy_labeled << " labels generated in "
                  << std::fixed << std::setprecision(1) << spy_label_time << "s (" << spy_valid << " valid)\n";
    }

    auto pass2_end = std::chrono::high_resolution_clock::now();
    stats_.pass2_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pass2_end - pass2_start).count();

    if (config_.verbose) {
        std::cout << "\n  Pass 2 summary: " << (tsla_labeled + spy_labeled) << " total labels, "
                  << std::fixed << std::setprecision(1) << (stats_.pass2_duration_ms / 1000.0) << "s total time\n";
    }

    // Free Pass-1 artifacts
    tsla_channel_map.clear();
    spy_channel_map.clear();

    // =========================================================================
    // PASS 3: Create ONE sample per detected TSLA channel at channel END
    // =========================================================================
    if (config_.verbose) {
        std::cout << "\n[SCAN] Creating samples from labeled channels...\n";
    }

    // Build list of (tf, window, channel_idx) for all valid TSLA channels
    std::vector<ChannelWorkItem> channel_work_items;
    int total_valid_channels = 0;
    int warmup_filtered = 0;
    int debug_count = 0;

    for (const auto& pair : tsla_slim_map) {
        const std::string& tf = pair.first.tf;
        int window = pair.first.window;
        const std::vector<SlimLabeledChannel>& channels = pair.second;

        // Convert TF to bars_per_tf for warmup check
        Timeframe tf_enum = string_to_timeframe(tf);
        int bars_per_tf = get_bars_per_tf(tf_enum);

        for (size_t idx = 0; idx < channels.size(); ++idx) {
            const SlimLabeledChannel& ch = channels[idx];
            if (ch.channel_valid && ch.labels.direction_valid) {
                total_valid_channels++;

                // Convert channel end timestamp to 5min index for warmup check
                int64_t end_ts_ms = ch.end_timestamp;
                // NOTE: tsla_df timestamps are in SECONDS, channel timestamps are in MILLISECONDS
                std::time_t end_ts_sec = end_ts_ms / 1000;
                auto it = std::lower_bound(tsla_df.begin(), tsla_df.end(), end_ts_sec,
                                           [](const OHLCV& bar, std::time_t ts) { return bar.timestamp < ts; });
                int idx_5min = (it != tsla_df.end() && it->timestamp == end_ts_sec)
                               ? std::distance(tsla_df.begin(), it)
                               : std::distance(tsla_df.begin(), it) - 1;

                // Debug: print first few channels
                if (config_.verbose && debug_count < 5) {
                    std::cout << "  [DEBUG] Channel " << debug_count << ": tf=" << tf << ", window=" << window
                              << ", idx=" << idx << ", end_ts=" << end_ts_ms
                              << ", idx_5min=" << idx_5min
                              << ", tsla_df[idx_5min].ts=" << (idx_5min >= 0 && idx_5min < (int)tsla_df.size() ? tsla_df[idx_5min].timestamp : -1)
                              << ", warmup_bars=" << config_.warmup_bars
                              << ", pass=" << (idx_5min >= config_.warmup_bars ? "YES" : "NO") << "\n";
                    debug_count++;
                }

                // Apply warmup filter (matching Python logic)
                if (idx_5min >= config_.warmup_bars) {
                    channel_work_items.emplace_back(tf, window, static_cast<int>(idx), ch.end_timestamp);
                } else {
                    warmup_filtered++;
                }
            }
        }
    }

    // CRITICAL: Sort work items by end_timestamp to ensure consistent ordering
    // across Python and C++ (Python dict preserves insertion order, C++ unordered_map does not)
    std::sort(channel_work_items.begin(), channel_work_items.end(),
              [](const ChannelWorkItem& a, const ChannelWorkItem& b) {
                  return a.end_timestamp < b.end_timestamp;
              });

    if (config_.verbose) {
        std::cout << "  [PASS3 PREP] Total valid channels: " << total_valid_channels << "\n";
        std::cout << "  [PASS3 PREP] Filtered by warmup: " << warmup_filtered << "\n";
        std::cout << "  [PASS3 PREP] Work items created: " << channel_work_items.size() << "\n";
    }

    int total_channels_to_process = channel_work_items.size();

    if (config_.max_samples > 0 && total_channels_to_process > config_.max_samples) {
        channel_work_items.resize(config_.max_samples);
        if (config_.verbose) {
            std::cout << "\n[SCAN] Limited to " << config_.max_samples << " channels (max_samples specified)\n";
        }
    }

    if (config_.verbose) {
        std::cout << "\n[SCAN] Starting sample generation...\n";
        std::cout << "  Channels to process: " << channel_work_items.size() << "\n";
        std::cout << "  Each channel produces ONE sample at its end position\n";
        std::cout << "  Processing mode: " << (config_.workers != 1 ? "PARALLEL" : "SEQUENTIAL") << "\n";
        std::cout << "  Batch size: " << config_.batch_size << "\n";
    }

    auto pass3_start = std::chrono::high_resolution_clock::now();

    std::vector<ChannelSample> samples;  // Only used in non-streaming mode
    int valid_count = 0;
    int skipped_count = 0;
    int error_count = 0;
    std::vector<double> feature_times_ms;

    // Create batches
    std::vector<std::vector<ChannelWorkItem>> batches;
    for (size_t i = 0; i < channel_work_items.size(); i += config_.batch_size) {
        size_t end = std::min(i + config_.batch_size, channel_work_items.size());
        batches.emplace_back(channel_work_items.begin() + i, channel_work_items.begin() + end);
    }

    if (config_.verbose) {
        std::cout << "  Total batches: " << batches.size() << "\n";
        if (!batches.empty()) {
            std::cout << "  First batch size: " << batches[0].size() << "\n";
        }
    }

    // =========================================================================
    // STREAMING MODE: Write samples directly to disk to avoid OOM
    // =========================================================================
    bool use_streaming = config_.streaming && !config_.output_path.empty();
    std::unique_ptr<StreamingSampleWriter> streaming_writer;
    std::mutex writer_mutex;  // Protect streaming writer in parallel mode

    if (use_streaming) {
        if (config_.verbose) {
            std::cout << "\n  [STREAMING MODE] Writing samples directly to disk\n";
            std::cout << "    Output: " << config_.output_path << "\n";
            std::cout << "    Flush interval: " << config_.flush_interval << " samples\n";
        }
        streaming_writer = std::make_unique<StreamingSampleWriter>(
            config_.output_path, config_.flush_interval
        );
        streaming_writer->open();
    }

    // Process batches
    if (config_.workers == 1) {
        // Sequential processing
        if (config_.verbose) {
            std::cout << "\n  Running in sequential mode...\n";
        }

        for (size_t i = 0; i < batches.size(); ++i) {
            if (shutdown_requested_) {
                if (config_.verbose) {
                    std::cout << "\n[INTERRUPT] Stopping scan\n";
                }
                break;
            }

            auto batch_samples = process_channel_batch(
                batches[i], tsla_df, spy_df, vix_df, tsla_slim_map, spy_slim_map
            );

            for (auto& sample : batch_samples) {
                if (sample.is_valid()) {
                    if (use_streaming) {
                        // Write directly to disk
                        streaming_writer->write(sample);
                    } else {
                        // Accumulate in memory
                        samples.push_back(std::move(sample));
                    }
                    ++valid_count;
                } else {
                    ++skipped_count;
                }
            }

            if (config_.progress && (i % 10 == 0 || i == batches.size() - 1)) {
                double progress_pct = 100.0 * (i + 1) / batches.size();
                std::cout << "\r  Progress: " << std::fixed << std::setprecision(1)
                          << progress_pct << "% (" << (i + 1) << "/" << batches.size() << " batches)"
                          << " | Samples: " << valid_count;
                if (use_streaming) {
                    std::cout << " (streamed to disk)";
                }
                std::cout.flush();
            }
        }

        if (config_.progress) {
            std::cout << "\n";
        }
    } else {
        // Parallel processing
        int workers = config_.workers > 0 ? config_.workers : std::thread::hardware_concurrency();
        if (config_.verbose) {
            std::cout << "\n  Starting parallel processing with " << workers << " workers...\n";
        }

        ThreadPool pool(workers);
        std::vector<std::future<std::vector<ChannelSample>>> futures;

        // Submit all batches
        for (const auto& batch : batches) {
            futures.push_back(pool.submit(
                [this, &batch, &tsla_df, &spy_df, &vix_df, &tsla_slim_map, &spy_slim_map]() {
                    return process_channel_batch(batch, tsla_df, spy_df, vix_df, tsla_slim_map, spy_slim_map);
                }
            ));
        }

        // Collect results
        for (size_t i = 0; i < futures.size(); ++i) {
            if (shutdown_requested_) {
                if (config_.verbose) {
                    std::cout << "\n[INTERRUPT] Stopping at batch " << i << "/" << futures.size() << "\n";
                }
                break;
            }

            auto batch_samples = futures[i].get();
            for (auto& sample : batch_samples) {
                if (sample.is_valid()) {
                    if (use_streaming) {
                        // Write directly to disk (thread-safe)
                        std::lock_guard<std::mutex> lock(writer_mutex);
                        streaming_writer->write(sample);
                    } else {
                        // Accumulate in memory
                        samples.push_back(std::move(sample));
                    }
                    ++valid_count;
                } else {
                    ++skipped_count;
                }
            }

            if (config_.progress && (i % 10 == 0 || i == futures.size() - 1)) {
                double progress_pct = 100.0 * (i + 1) / futures.size();
                std::cout << "\r  Progress: " << std::fixed << std::setprecision(1)
                          << progress_pct << "% (" << (i + 1) << "/" << futures.size() << " batches)"
                          << " | Samples: " << valid_count;
                if (use_streaming) {
                    std::cout << " (streamed to disk)";
                }
                std::cout.flush();
            }
        }

        if (config_.progress) {
            std::cout << "\n";
        }
    }

    // Close streaming writer if used
    if (use_streaming && streaming_writer) {
        streaming_writer->close();
        if (config_.verbose) {
            std::cout << "\n  [STREAMING] Wrote " << streaming_writer->samples_written()
                      << " samples to " << config_.output_path << "\n";
        }
    }

    auto pass3_end = std::chrono::high_resolution_clock::now();
    stats_.pass3_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pass3_end - pass3_start).count();

    // Sort samples by channel_end_idx
    std::sort(samples.begin(), samples.end(),
              [](const ChannelSample& a, const ChannelSample& b) {
                  return a.channel_end_idx < b.channel_end_idx;
              });

    // Update stats
    auto total_end = std::chrono::high_resolution_clock::now();
    stats_.total_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
    stats_.channels_processed = channel_work_items.size();
    stats_.samples_created = valid_count;
    stats_.samples_skipped = skipped_count;
    stats_.errors_encountered = error_count;

    // Calculate throughput
    if (stats_.total_duration_ms > 0) {
        stats_.samples_per_second = (valid_count * 1000.0) / stats_.total_duration_ms;
    }
    if (stats_.pass1_duration_ms > 0) {
        stats_.channels_per_second_pass1 = ((tsla_channels + spy_channels) * 1000.0) / stats_.pass1_duration_ms;
    }
    if (stats_.pass2_duration_ms > 0) {
        stats_.labels_per_second_pass2 = ((tsla_labeled + spy_labeled) * 1000.0) / stats_.pass2_duration_ms;
    }

    // Print summary
    if (config_.verbose) {
        print_summary();
    }

    return samples;
}

// =============================================================================
// HELPER FUNCTIONS FOR RESAMPLING
// =============================================================================

namespace {

/**
 * Resample OHLCV data to a target timeframe using standard OHLC aggregation.
 *
 * Rules:
 * - Open: first value in period
 * - High: max value in period
 * - Low: min value in period
 * - Close: last value in period
 * - Volume: sum in period
 */
std::vector<OHLCV> resample_ohlcv(
    const std::vector<OHLCV>& source_data,
    const std::string& target_tf,
    int bars_per_period
) {
    if (source_data.empty()) {
        return {};
    }

    std::vector<OHLCV> resampled;
    resampled.reserve(source_data.size() / bars_per_period + 1);

    size_t i = 0;
    while (i < source_data.size()) {
        OHLCV bar;
        bar.timestamp = source_data[i].timestamp;  // Use first bar's timestamp
        bar.open = source_data[i].open;
        bar.high = source_data[i].high;
        bar.low = source_data[i].low;
        bar.close = source_data[i].close;
        bar.volume = source_data[i].volume;

        size_t bars_in_period = 1;
        size_t j = i + 1;

        // Aggregate bars_per_period bars (or until end of data)
        while (j < source_data.size() && bars_in_period < static_cast<size_t>(bars_per_period)) {
            bar.high = std::max(bar.high, source_data[j].high);
            bar.low = std::min(bar.low, source_data[j].low);
            bar.close = source_data[j].close;  // Last close
            bar.volume += source_data[j].volume;
            ++j;
            ++bars_in_period;
        }

        // Only add complete bars (skip partial bar at end)
        if (bars_in_period == static_cast<size_t>(bars_per_period)) {
            resampled.push_back(bar);
        }

        i = j;
    }

    return resampled;
}

/**
 * Get bars per timeframe (how many 5min bars per TF bar)
 */
int get_bars_per_tf(const std::string& tf) {
    if (tf == "5min") return 1;
    if (tf == "15min") return 3;
    if (tf == "30min") return 6;
    if (tf == "1h") return 12;
    if (tf == "2h") return 24;
    if (tf == "3h") return 36;
    if (tf == "4h") return 48;
    if (tf == "daily") return 78;
    if (tf == "weekly") return 390;
    if (tf == "monthly") return 1638;
    return 1;  // Default to 5min
}

} // anonymous namespace

// =============================================================================
// PASS 1: CHANNEL DETECTION
// =============================================================================

void Scanner::detect_all_channels(
    const std::vector<OHLCV>& df,
    const std::string& asset_name,
    std::unordered_map<TFWindowKey, std::vector<Channel>, TFWindowKeyHash>& channel_map,
    std::unordered_map<std::string, std::vector<OHLCV>>& resampled_dfs
) {
    // Calculate valid scan range in 5min bars
    // We need to leave enough forward data for label generation
    size_t n_bars_5min = df.size();
    size_t valid_end_5min = (n_bars_5min > static_cast<size_t>(SCANNER_FORWARD_5MIN))
                          ? n_bars_5min - SCANNER_FORWARD_5MIN
                          : 0;

    if (config_.verbose) {
        std::cout << "    [SCAN BOUNDS] Total 5min bars: " << n_bars_5min << "\n";
        std::cout << "    [SCAN BOUNDS] Scanner forward requirement: " << SCANNER_FORWARD_5MIN << " bars\n";
        std::cout << "    [SCAN BOUNDS] Valid scan end (5min): " << valid_end_5min << "\n";
    }

    // Step 1: Resample to all timeframes
    for (int tf_idx = 0; tf_idx < NUM_TIMEFRAMES; ++tf_idx) {
        const char* tf = TIMEFRAME_NAMES[tf_idx];

        if (std::string(tf) == "5min") {
            // No resampling needed for 5min
            resampled_dfs[tf] = df;
        } else {
            // Resample to target timeframe
            int bars_per_period = get_bars_per_tf(tf);
            resampled_dfs[tf] = resample_ohlcv(df, tf, bars_per_period);
        }

        if (config_.verbose) {
            std::cout << "    " << asset_name << " " << tf << ": "
                      << resampled_dfs[tf].size() << " bars\n";
        }
    }

    // Step 2: Detect channels for each (timeframe, window) combination
    // Use OpenMP to parallelize over timeframes
    int total_channels = 0;
    std::mutex channel_map_mutex;

    #pragma omp parallel for schedule(dynamic) if(config_.workers != 1)
    for (int tf_idx = 0; tf_idx < NUM_TIMEFRAMES; ++tf_idx) {
        const char* tf = TIMEFRAME_NAMES[tf_idx];
        const std::vector<OHLCV>& tf_data = resampled_dfs[tf];

        if (tf_data.empty()) {
            continue;
        }

        // Calculate valid scan end for this timeframe
        // Convert 5min forward requirement to TF bars
        int bars_per_tf = get_bars_per_tf(tf);
        size_t tf_forward_bars = (SCANNER_FORWARD_5MIN + bars_per_tf - 1) / bars_per_tf;  // Round up
        size_t valid_end_tf = (tf_data.size() > tf_forward_bars)
                            ? tf_data.size() - tf_forward_bars
                            : 0;

        if (config_.verbose) {
            #pragma omp critical
            {
                std::cout << "    [SCAN BOUNDS] " << asset_name << " " << tf
                          << ": total=" << tf_data.size() << " bars, forward_req=" << tf_forward_bars
                          << " bars, valid_end=" << valid_end_tf << "\n";
            }
        }

        // Extract price arrays for this timeframe
        std::vector<double> high, low, close;
        high.reserve(tf_data.size());
        low.reserve(tf_data.size());
        close.reserve(tf_data.size());

        for (const auto& bar : tf_data) {
            high.push_back(bar.high);
            low.push_back(bar.low);
            close.push_back(bar.close);
        }

        // Detect channels for all windows at this timeframe
        for (int window : STANDARD_WINDOWS) {
            if (static_cast<size_t>(window) > valid_end_tf) {
                continue;  // Not enough data for this window with forward requirement
            }

            std::vector<Channel> detected_channels;

            // Scan through the data with step size, respecting valid_end
            static int call_count = 0;
            for (size_t pos = window; pos < valid_end_tf; pos += config_.step) {
                // Extract window of data ending at pos (exclusive of current bar)
                size_t start_idx = pos - window;
                std::vector<double> win_high(high.begin() + start_idx, high.begin() + pos);
                std::vector<double> win_low(low.begin() + start_idx, low.begin() + pos);
                std::vector<double> win_close(close.begin() + start_idx, close.begin() + pos);

                // Debug: Log first call
                if (call_count == 0 && config_.verbose) {
                    std::cout << "        [SCANNER DEBUG] First detect_channel call:\n";
                    std::cout << "          window=" << window << " pos=" << pos << " start_idx=" << start_idx << "\n";
                    std::cout << "          win_high.size()=" << win_high.size() << "\n";
                    std::cout << "          win_low.size()=" << win_low.size() << "\n";
                    std::cout << "          win_close.size()=" << win_close.size() << "\n";
                    if (!win_close.empty()) {
                        std::cout << "          win_close[0]=" << win_close[0] << " win_close[last]=" << win_close[win_close.size()-1] << "\n";
                    }
                    call_count++;
                }

                // Detect channel
                Channel ch = ChannelDetector::detect_channel(
                    win_high,
                    win_low,
                    win_close,
                    window,
                    2.0,  // std_multiplier
                    0.10, // touch_threshold
                    config_.min_cycles
                );

                // Debug: Log first few rejected channels
                static int debug_count = 0;
                if (!ch.valid || ch.complete_cycles < config_.min_cycles) {
                    if (debug_count < 10 && config_.verbose) {
                        #pragma omp critical
                        {
                            std::cout << "      [DEBUG] Rejected channel at pos=" << pos
                                      << " valid=" << ch.valid
                                      << " cycles=" << ch.complete_cycles
                                      << " bounces=" << ch.bounce_count
                                      << " touches=" << ch.touches.size()
                                      << " upper_touches=" << ch.upper_touches
                                      << " lower_touches=" << ch.lower_touches << "\n";
                            debug_count++;
                        }
                    }
                }

                // Store valid channels
                // Use bounce_count instead of complete_cycles since bounces are more lenient
                static int store_debug_count = 0;
                if (ch.valid && ch.bounce_count >= config_.min_cycles) {
                    // Set position indices - critical for label generation!
                    ch.start_idx = static_cast<int>(start_idx);
                    ch.end_idx = static_cast<int>(pos) - 1;  // pos is exclusive, so end is pos-1
                    ch.window_size = window;

                    // Set timeframe (critical for is_valid() check in Pass 2)
                    ch.timeframe = static_cast<Timeframe>(tf_idx);

                    // Store timestamps if available (for Pass 2 label generation)
                    if (start_idx < tf_data.size()) {
                        ch.start_timestamp_ms = tf_data[start_idx].timestamp * 1000;  // Convert to ms
                    }
                    if (pos - 1 < tf_data.size()) {
                        ch.end_timestamp_ms = tf_data[pos - 1].timestamp * 1000;  // Convert to ms
                    }

                    detected_channels.push_back(ch);
                    if (store_debug_count < 5 && config_.verbose) {
                        std::cout << "      [STORED CHANNEL #" << store_debug_count << "] pos=" << pos
                                  << " start_idx=" << ch.start_idx << " end_idx=" << ch.end_idx
                                  << " timeframe=" << tf
                                  << " valid=" << ch.valid << " bounces=" << ch.bounce_count
                                  << " detected_channels.size()=" << detected_channels.size() << "\n";
                        store_debug_count++;
                    }
                } else {
                    if (store_debug_count < 5 && config_.verbose) {
                        std::cout << "      [SKIPPED CHANNEL #" << store_debug_count << "] pos=" << pos
                                  << " valid=" << ch.valid << " bounces=" << ch.bounce_count
                                  << " min_cycles=" << config_.min_cycles << "\n";
                        store_debug_count++;
                    }
                }
            }

            // Store results in channel_map (thread-safe)
            size_t num_channels = detected_channels.size();  // Get size BEFORE move
            {
                std::lock_guard<std::mutex> lock(channel_map_mutex);
                TFWindowKey key{tf, window};
                channel_map[key] = std::move(detected_channels);
                total_channels += num_channels;
            }

            if (config_.verbose) {
                #pragma omp critical
                {
                    std::cout << "    " << asset_name << " " << tf
                              << " window=" << window << ": "
                              << num_channels << " channels\n";
                }
            }
        }
    }

    if (config_.verbose) {
        std::cout << "    " << asset_name << " total: " << total_channels << " channels\n";
    }
}

// =============================================================================
// PASS 2: LABEL GENERATION
// =============================================================================

void Scanner::generate_all_labels(
    const std::unordered_map<TFWindowKey, std::vector<Channel>, TFWindowKeyHash>& channel_map,
    const std::unordered_map<std::string, std::vector<OHLCV>>& resampled_dfs,
    SlimLabeledChannelMap& labeled_map
) {
    // Create label generator
    LabelGenerator label_gen;

    // Prepare parallel processing across (tf, window) combinations
    std::vector<TFWindowKey> keys;
    keys.reserve(channel_map.size());
    for (const auto& pair : channel_map) {
        if (!pair.second.empty()) {
            keys.push_back(pair.first);
        }
    }

    if (keys.empty()) {
        if (config_.verbose) {
            std::cout << "  [PASS 2] No channels to label\n";
        }
        return;
    }

    // Synchronization for result collection
    std::mutex result_mutex;
    std::atomic<int> channels_processed{0};
    std::atomic<int> valid_labels_count{0};
    std::cout << "  [DEBUG] Starting label generation for " << keys.size() << " tf/window combinations\n";

    // Process each (tf, window) combination
    auto process_tf_window = [&](const TFWindowKey& key) {
        const std::string& tf_str = key.tf;
        int window = key.window;

        // Get channels for this (tf, window)
        auto it = channel_map.find(key);
        if (it == channel_map.end() || it->second.empty()) {
            return;
        }

        const std::vector<Channel>& channels = it->second;

        // Get resampled data for this timeframe
        auto df_it = resampled_dfs.find(tf_str);
        if (df_it == resampled_dfs.end()) {
            return;
        }

        const std::vector<OHLCV>& tf_df = df_it->second;
        int n_bars = tf_df.size();

        // Convert timeframe string to enum
        Timeframe tf = string_to_timeframe(tf_str);
        if (tf == Timeframe::INVALID) {
            return;
        }

        int max_scan = get_max_scan(tf);

        // Process each channel
        std::vector<SlimLabeledChannel> slim_channels;
        slim_channels.reserve(channels.size());

        for (size_t ch_idx = 0; ch_idx < channels.size(); ++ch_idx) {
            const Channel& channel = channels[ch_idx];

            // Validate channel
            if (!channel.is_valid()) {
                // Create invalid slim channel
                SlimLabeledChannel slim;
                slim.channel_valid = false;
                slim_channels.push_back(slim);
                channels_processed++;
                continue;
            }

            // Get forward data for label scanning
            int end_idx = channel.end_idx;
            int available_forward = n_bars - end_idx - 1;

            if (available_forward <= 0) {
                // Not enough forward data - create channel without valid labels
                static int no_forward_count = 0;
                if (no_forward_count < 5 && config_.verbose) {
                    std::cout << "      [NO_FORWARD #" << no_forward_count << "] end_idx=" << end_idx
                              << " n_bars=" << n_bars << " available_forward=" << available_forward << "\n";
                    no_forward_count++;
                }
                SlimLabeledChannel slim;
                slim.start_timestamp = channel.start_timestamp_ms;
                slim.end_timestamp = channel.end_timestamp_ms;
                slim.start_idx = channel.start_idx;
                slim.end_idx = channel.end_idx;
                slim.channel_slope = channel.slope;
                slim.channel_intercept = channel.intercept;
                slim.channel_std_dev = channel.std_dev;
                slim.channel_r_squared = channel.r_squared;
                slim.channel_direction = static_cast<int>(channel.direction);
                slim.channel_valid = true;
                slim.channel_window = window;
                slim.channel_bounce_count = channel.bounce_count;
                slim.tf = tf_str;
                slim.labels.direction_valid = false;
                slim_channels.push_back(slim);
                channels_processed++;
                continue;
            }

            // Limit forward scan to TF_MAX_SCAN
            int scan_bars = std::min(available_forward, max_scan);

            // Extract forward price arrays
            std::vector<double> forward_high(scan_bars);
            std::vector<double> forward_low(scan_bars);
            std::vector<double> forward_close(scan_bars);

            for (int i = 0; i < scan_bars; ++i) {
                int df_idx = end_idx + 1 + i;
                forward_high[i] = tf_df[df_idx].high;
                forward_low[i] = tf_df[df_idx].low;
                forward_close[i] = tf_df[df_idx].close;
            }

            // Compute next channel direction (look ahead at next 2 channels)
            int next_channel_direction = -1;
            if (ch_idx + 1 < channels.size()) {
                const Channel& next_ch = channels[ch_idx + 1];
                if (next_ch.is_valid()) {
                    next_channel_direction = static_cast<int>(next_ch.direction);
                }
            }

            // Build full close prices array for RSI computation
            // Need lookback for RSI period + channel window + forward scan
            int rsi_lookback = 14;  // RSI period
            int required_lookback = rsi_lookback + channel.window_size;
            int start_price_idx = std::max(0, end_idx - required_lookback);
            int end_price_idx = std::min(n_bars - 1, end_idx + scan_bars);
            int full_close_size = end_price_idx - start_price_idx + 1;

            std::vector<double> full_close_prices(full_close_size);
            for (int i = 0; i < full_close_size; ++i) {
                int df_idx = start_price_idx + i;
                if (df_idx >= 0 && df_idx < n_bars) {
                    full_close_prices[i] = tf_df[df_idx].close;
                } else {
                    full_close_prices[i] = 0.0;  // Should not happen with bounds checks
                }
            }

            // Adjust channel_end_idx to be relative to full_close_prices array
            int adjusted_end_idx = end_idx - start_price_idx;

            // Debug: Log first few label generation calls
            static int label_gen_count = 0;
            if (label_gen_count < 5 && config_.verbose) {
                std::cout << "      [LABEL_GEN #" << label_gen_count << "] tf=" << tf_str
                          << " window=" << window << " end_idx=" << end_idx
                          << " scan_bars=" << scan_bars << " max_scan=" << max_scan
                          << " channel.slope=" << channel.slope
                          << " channel.std_dev=" << channel.std_dev << "\n";
                label_gen_count++;
            }

            // Generate labels using forward scan
            ChannelLabels labels = label_gen.generate_labels_forward_scan(
                channel,
                adjusted_end_idx,
                forward_high.data(),
                forward_low.data(),
                forward_close.data(),
                scan_bars,
                max_scan,
                next_channel_direction,
                full_close_prices.data(),
                full_close_size
            );

            // Debug: Log result of first few label generations (if verbose)
            if (label_gen_count <= 5 && config_.verbose) {
                std::cout << "      [LABEL_RESULT #" << (label_gen_count-1) << "] "
                          << " duration_valid=" << labels.duration_valid
                          << " direction_valid=" << labels.direction_valid
                          << " break_scan_valid=" << labels.break_scan_valid
                          << " break_direction=" << labels.break_direction
                          << " duration_bars=" << labels.duration_bars << "\n";
            }

            // Create SlimLabeledChannel (strip heavy arrays)
            SlimLabeledChannel slim;
            slim.start_timestamp = channel.start_timestamp_ms;
            slim.end_timestamp = channel.end_timestamp_ms;
            slim.start_idx = channel.start_idx;
            slim.end_idx = channel.end_idx;
            slim.channel_slope = channel.slope;
            slim.channel_intercept = channel.intercept;
            slim.channel_std_dev = channel.std_dev;
            slim.channel_r_squared = channel.r_squared;
            slim.channel_direction = static_cast<int>(channel.direction);
            slim.channel_valid = true;
            slim.channel_window = window;
            slim.channel_bounce_count = channel.bounce_count;
            slim.tf = tf_str;
            slim.labels = labels;

            slim_channels.push_back(slim);

            channels_processed++;
            if (labels.direction_valid) {
                valid_labels_count++;
            } else {
                if (channels_processed <= 5) {
                    std::cout << "      [INVALID_LABEL ch=" << channels_processed << "] tf=" << tf_str
                              << " window=" << window << " end_idx=" << end_idx
                              << " scan_bars=" << scan_bars
                              << " duration_bars=" << labels.duration_bars
                              << " duration_valid=" << labels.duration_valid
                              << " direction_valid=" << labels.direction_valid
                              << " break_scan_valid=" << labels.break_scan_valid << "\n";
                }
            }
        }

        // Sort by end_idx for binary search preparation
        std::sort(slim_channels.begin(), slim_channels.end(),
                  [](const SlimLabeledChannel& a, const SlimLabeledChannel& b) {
                      return a.end_timestamp < b.end_timestamp;
                  });

        // Store in labeled_map (thread-safe)
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            labeled_map[key] = std::move(slim_channels);
        }
    };

    // Execute in parallel or sequentially
    if (config_.workers == 1) {
        // Sequential processing
        for (const auto& key : keys) {
            if (shutdown_requested_) {
                break;
            }
            process_tf_window(key);
        }
    } else {
        // Parallel processing using thread pool
        int workers = config_.workers > 0 ? config_.workers : std::thread::hardware_concurrency();
        if (workers == 0) workers = 4;

        ThreadPool pool(workers);
        std::vector<std::future<void>> futures;

        for (const auto& key : keys) {
            futures.push_back(pool.submit([&, key]() {
                if (!shutdown_requested_) {
                    process_tf_window(key);
                }
            }));
        }

        // Wait for all tasks to complete
        for (auto& future : futures) {
            future.get();
        }
    }

    if (config_.verbose) {
        std::cout << "  Processed " << channels_processed.load() << " channels, "
                  << valid_labels_count.load() << " with valid labels\n";
    }
}

SlimLabeledChannelMap Scanner::create_slim_labeled_map(
    const std::unordered_map<TFWindowKey, std::vector<Channel>, TFWindowKeyHash>& channel_map,
    const std::unordered_map<std::string, std::vector<OHLCV>>& resampled_dfs
) {
    // Placeholder implementation
    SlimLabeledChannelMap result;
    return result;
}

std::vector<ChannelSample> Scanner::process_channel_batch(
    const std::vector<ChannelWorkItem>& batch,
    const std::vector<OHLCV>& tsla_df,
    const std::vector<OHLCV>& spy_df,
    const std::vector<OHLCV>& vix_df,
    const SlimLabeledChannelMap& tsla_slim_map,
    const SlimLabeledChannelMap& spy_slim_map
) {
    static std::atomic<int> batch_call_count{0};
    int this_batch_id = batch_call_count.fetch_add(1);

    if (config_.verbose && this_batch_id == 0) {
        std::cout << "\n[PASS3 BATCH #" << this_batch_id << "] Starting, batch.size()=" << batch.size() << "\n";
    }

    std::vector<ChannelSample> samples;
    samples.reserve(batch.size());

    // Lookback bars for feature extraction (from Python config)
    constexpr int SCANNER_LOOKBACK_5MIN = 32760;

    int items_processed = 0;
    int items_skipped_map_not_found = 0;
    int items_skipped_invalid_idx = 0;
    int items_skipped_channel_invalid = 0;
    int items_skipped_labels_invalid = 0;
    int items_skipped_index_bounds = 0;
    int items_skipped_warmup = 0;
    int items_samples_created = 0;

    for (const auto& work_item : batch) {
        items_processed++;
        if (shutdown_requested_) {
            break;
        }

        const std::string& primary_tf = work_item.primary_tf;
        int primary_window = work_item.primary_window;
        int channel_idx = work_item.channel_idx;

        static int batch_debug_count = 0;
        if (batch_debug_count < 5 && config_.verbose) {
            std::cout << "      [BATCH_PROCESS #" << batch_debug_count << "] "
                      << "tf=" << primary_tf << " window=" << primary_window
                      << " channel_idx=" << channel_idx << "\n";
            batch_debug_count++;
        }

        try {
            // Get the PRIMARY channel from slim_map
            TFWindowKey key{primary_tf, primary_window};
            auto it = tsla_slim_map.find(key);
            if (it == tsla_slim_map.end()) {
                items_skipped_map_not_found++;
                if (batch_debug_count <= 5 && config_.verbose) {
                    std::cout << "      [SKIP] Channel map not found for (" << primary_tf << "," << primary_window << ")\n";
                }
                continue;  // Skip - channel map not found
            }

            const std::vector<SlimLabeledChannel>& slim_channels = it->second;
            if (channel_idx < 0 || channel_idx >= static_cast<int>(slim_channels.size())) {
                items_skipped_invalid_idx++;
                if (batch_debug_count <= 5 && config_.verbose) {
                    std::cout << "      [SKIP] Invalid channel index " << channel_idx
                              << " (size=" << slim_channels.size() << ")\n";
                }
                continue;  // Skip - invalid channel index
            }

            const SlimLabeledChannel& primary_channel = slim_channels[channel_idx];

            // Skip invalid channels
            if (!primary_channel.channel_valid) {
                items_skipped_channel_invalid++;
                if (batch_debug_count <= 5 && config_.verbose) {
                    std::cout << "      [SKIP] Channel not valid\n";
                }
                continue;
            }

            // Skip channels without valid labels
            if (!primary_channel.labels.direction_valid) {
                items_skipped_labels_invalid++;
                if (batch_debug_count <= 5 && config_.verbose) {
                    std::cout << "      [SKIP] Labels not valid\n";
                }
                continue;
            }

            if (batch_debug_count <= 5 && config_.verbose) {
                std::cout << "      [PROCESSING] Channel passed validation checks\n";
            }

            // SAMPLE POSITION = CHANNEL END
            int64_t sample_timestamp = primary_channel.end_timestamp;

            // Convert end_idx from TF space to 5min space
            Timeframe tf_enum = string_to_timeframe(primary_tf);
            int bars_per_tf = get_bars_per_tf(tf_enum);
            int idx_5min = primary_channel.end_idx * bars_per_tf;

            if (batch_debug_count <= 5 && config_.verbose) {
                std::cout << "      [INDEX] end_idx=" << primary_channel.end_idx
                          << " bars_per_tf=" << bars_per_tf
                          << " idx_5min=" << idx_5min
                          << " tsla_df.size()=" << tsla_df.size() << "\n";
            }

            // SAFETY: Validate index
            if (idx_5min < 0 || idx_5min >= static_cast<int>(tsla_df.size())) {
                items_skipped_index_bounds++;
                if (config_.verbose) {
                    std::cerr << "[WARNING] Invalid 5min index " << idx_5min
                              << " (tf=" << primary_tf << " end_idx=" << primary_channel.end_idx
                              << " bars_per_tf=" << bars_per_tf << ")"
                              << " for channel (" << primary_tf << "," << primary_window << "," << channel_idx << ")\n";
                }
                continue;
            }

            // NOTE: Warmup check removed - Pass 1 already filtered by scan bounds
            // All channels here are guaranteed to be within valid scan range

            // SAFETY: Calculate safe slice indices
            int start_idx = std::max(0, idx_5min - SCANNER_LOOKBACK_5MIN);
            int slice_size = idx_5min - start_idx;

            if (slice_size <= 0) {
                if (config_.verbose) {
                    std::cerr << "[WARNING] Invalid slice size " << slice_size
                              << " for channel at idx " << idx_5min << "\n";
                }
                continue;
            }

            // SAFETY: Validate data alignment
            if (tsla_df.size() != spy_df.size() || tsla_df.size() != vix_df.size()) {
                std::cerr << "[ERROR] Data size mismatch in batch processing\n";
                continue;
            }

            // Get data slices for feature extraction
            std::vector<OHLCV> tsla_slice(tsla_df.begin() + start_idx, tsla_df.begin() + idx_5min);
            std::vector<OHLCV> spy_slice(spy_df.begin() + start_idx, spy_df.begin() + idx_5min);
            std::vector<OHLCV> vix_slice(vix_df.begin() + start_idx, vix_df.begin() + idx_5min);

            // SAFETY: Validate slices
            if (tsla_slice.empty() || spy_slice.empty() || vix_slice.empty()) {
                if (config_.verbose) {
                    std::cerr << "[WARNING] Empty data slices for channel at idx " << idx_5min << "\n";
                }
                continue;
            }

            // Extract features at channel end position (with timing)
            auto feature_start = std::chrono::high_resolution_clock::now();

            auto tf_features = FeatureExtractor::extract_all_features(
                tsla_slice,
                spy_slice,
                vix_slice,
                sample_timestamp,
                static_cast<int>(tsla_slice.size()),  // source_bar_count - use slice size!
                true       // include_bar_metadata
            );

            auto feature_end = std::chrono::high_resolution_clock::now();
            double feature_extraction_time_ms =
                std::chrono::duration<double, std::milli>(feature_end - feature_start).count();

            if (batch_debug_count <= 5 && config_.verbose) {
                std::cout << "      [FEATURES_EXTRACTED] Got " << tf_features.size() << " features\n";
            }

            // SAFETY: Validate feature extraction
            if (tf_features.empty()) {
                if (config_.verbose) {
                    std::cerr << "[WARNING] Empty features returned for channel ("
                              << primary_tf << "," << primary_window << "," << channel_idx << ")\n";
                }
                if (config_.strict) {
                    continue;
                }
            }

            // Validate feature count
            size_t expected_features = static_cast<size_t>(FeatureExtractor::get_total_feature_count());
            if (tf_features.size() != expected_features) {
                if (config_.verbose) {
                    std::cerr << "WARNING: Feature count mismatch for channel (" << primary_tf << ","
                              << primary_window << "," << channel_idx << "): got "
                              << tf_features.size() << ", expected " << expected_features << "\n";
                }
                if (config_.strict) {
                    if (batch_debug_count <= 5 && config_.verbose) {
                        std::cout << "      [STRICT] Skipping due to feature count mismatch\n";
                    }
                    continue;
                } else {
                    if (batch_debug_count <= 5 && config_.verbose) {
                        std::cout << "      [NOT_STRICT] Continuing despite mismatch (strict=" << config_.strict << ")\n";
                    }
                }
            }

            if (batch_debug_count <= 5 && config_.verbose) {
                std::cout << "      [CHECKPOINT] About to build labels_per_window...\n";
            }

            // BUILD labels_per_window
            // For PRIMARY channel: use labels directly (no lookup)
            // For OTHER channels: binary search lookup at same timestamp

            if (batch_debug_count <= 5 && config_.verbose) {
                std::cout << "      [LABELS] Building labels_per_window map...\n";
            }

            // Create labels_per_window map: window -> timeframe -> labels
            std::unordered_map<int, std::unordered_map<std::string, ChannelLabels>> labels_per_window;

            int label_hits = 0;
            int label_misses = 0;

            for (int w : STANDARD_WINDOWS) {
                labels_per_window[w] = {};

                for (int tf_idx = 0; tf_idx < NUM_TIMEFRAMES; ++tf_idx) {
                    const char* tf_name = TIMEFRAME_NAMES[tf_idx];
                    std::string tf_str(tf_name);

                    // TSLA labels
                    ChannelLabels tsla_labels;
                    bool tsla_found = false;

                    if (tf_str == primary_tf && w == primary_window) {
                        // PRIMARY channel - use precomputed labels DIRECTLY
                        tsla_labels = primary_channel.labels;
                        tsla_found = true;
                    } else {
                        // Other (tf, window) - lookup channel at this timestamp
                        const SlimLabeledChannel* other_channel =
                            find_channel_at_timestamp(tsla_slim_map, tf_str, w, sample_timestamp);

                        if (other_channel != nullptr && other_channel->labels.direction_valid) {
                            tsla_labels = other_channel->labels;
                            tsla_found = true;
                        }
                    }

                    if (tsla_found && tsla_labels.direction_valid) {
                        ++label_hits;
                    } else {
                        ++label_misses;
                    }

                    // SPY labels - always lookup (we iterate TSLA channels)
                    ChannelLabels spy_labels;
                    bool spy_found = false;

                    const SlimLabeledChannel* spy_channel =
                        find_channel_at_timestamp(spy_slim_map, tf_str, w, sample_timestamp);

                    if (spy_channel != nullptr && spy_channel->labels.direction_valid) {
                        spy_labels = spy_channel->labels;
                        spy_found = true;
                    }

                    if (spy_found && spy_labels.direction_valid) {
                        ++label_hits;
                    } else {
                        ++label_misses;
                    }

                    // Copy SPY's values into TSLA's spy_* cross-reference fields
                    if (spy_found && tsla_found) {
                        // Source channel parameters
                        tsla_labels.spy_source_channel_slope = spy_labels.source_channel_slope;
                        tsla_labels.spy_source_channel_intercept = spy_labels.source_channel_intercept;
                        tsla_labels.spy_source_channel_std_dev = spy_labels.source_channel_std_dev;
                        tsla_labels.spy_source_channel_r_squared = spy_labels.source_channel_r_squared;
                        tsla_labels.spy_source_channel_direction = spy_labels.source_channel_direction;
                        tsla_labels.spy_source_channel_bounce_count = spy_labels.source_channel_bounce_count;
                        tsla_labels.spy_source_channel_start_ts = spy_labels.source_channel_start_ts;
                        tsla_labels.spy_source_channel_end_ts = spy_labels.source_channel_end_ts;

                        // Next channel labels
                        tsla_labels.spy_best_next_channel_direction = spy_labels.best_next_channel_direction;
                        tsla_labels.spy_best_next_channel_bars_away = spy_labels.best_next_channel_bars_away;
                        tsla_labels.spy_best_next_channel_duration = spy_labels.best_next_channel_duration;
                        tsla_labels.spy_best_next_channel_r_squared = spy_labels.best_next_channel_r_squared;
                        tsla_labels.spy_best_next_channel_bounce_count = spy_labels.best_next_channel_bounce_count;
                        tsla_labels.spy_shortest_next_channel_direction = spy_labels.shortest_next_channel_direction;
                        tsla_labels.spy_shortest_next_channel_bars_away = spy_labels.shortest_next_channel_bars_away;
                        tsla_labels.spy_shortest_next_channel_duration = spy_labels.shortest_next_channel_duration;
                        tsla_labels.spy_small_channels_before_best = spy_labels.small_channels_before_best;

                        // RSI labels
                        tsla_labels.spy_rsi_at_first_break = spy_labels.rsi_at_first_break;
                        tsla_labels.spy_rsi_at_permanent_break = spy_labels.rsi_at_permanent_break;
                        tsla_labels.spy_rsi_at_channel_end = spy_labels.rsi_at_channel_end;
                        tsla_labels.spy_rsi_overbought_at_break = spy_labels.rsi_overbought_at_break;
                        tsla_labels.spy_rsi_oversold_at_break = spy_labels.rsi_oversold_at_break;
                        tsla_labels.spy_rsi_divergence_at_break = spy_labels.rsi_divergence_at_break;
                        tsla_labels.spy_rsi_trend_in_channel = spy_labels.rsi_trend_in_channel;
                        tsla_labels.spy_rsi_range_in_channel = spy_labels.rsi_range_in_channel;

                        // Break scan features
                        tsla_labels.spy_break_direction = spy_labels.break_direction;
                        tsla_labels.spy_break_magnitude = spy_labels.break_magnitude;
                        tsla_labels.spy_bars_to_first_break = spy_labels.bars_to_first_break;
                        tsla_labels.spy_returned_to_channel = spy_labels.returned_to_channel;
                        tsla_labels.spy_bounces_after_return = spy_labels.bounces_after_return;
                        tsla_labels.spy_round_trip_bounces = spy_labels.round_trip_bounces;
                        tsla_labels.spy_channel_continued = spy_labels.channel_continued;
                        tsla_labels.spy_permanent_break_direction = spy_labels.permanent_break_direction;
                        tsla_labels.spy_permanent_break_magnitude = spy_labels.permanent_break_magnitude;
                        tsla_labels.spy_bars_to_permanent_break = spy_labels.bars_to_permanent_break;
                        tsla_labels.spy_duration_to_permanent = spy_labels.duration_to_permanent;
                        tsla_labels.spy_avg_bars_outside = spy_labels.avg_bars_outside;
                        tsla_labels.spy_total_bars_outside = spy_labels.total_bars_outside;
                        tsla_labels.spy_durability_score = spy_labels.durability_score;
                        tsla_labels.spy_first_break_returned = spy_labels.first_break_returned;
                        tsla_labels.spy_exit_return_rate = spy_labels.exit_return_rate;
                        tsla_labels.spy_exits_returned_count = spy_labels.exits_returned_count;
                        tsla_labels.spy_exits_stayed_out_count = spy_labels.exits_stayed_out_count;
                        tsla_labels.spy_scan_timed_out = spy_labels.scan_timed_out;
                        tsla_labels.spy_bars_verified_permanent = spy_labels.bars_verified_permanent;

                        // Copy exit event arrays
                        tsla_labels.spy_exit_bars = spy_labels.exit_bars;
                        tsla_labels.spy_exit_magnitudes = spy_labels.exit_magnitudes;
                        tsla_labels.spy_exit_durations = spy_labels.exit_durations;
                        tsla_labels.spy_exit_types = spy_labels.exit_types;
                        tsla_labels.spy_exit_returned = spy_labels.exit_returned;
                    }

                    // Store TSLA labels (with SPY cross-references)
                    labels_per_window[w][tf_str] = tsla_labels;
                }
            }

            // Build bar metadata
            std::unordered_map<std::string, std::unordered_map<std::string, double>> bar_metadata;
            bar_metadata["5min"]["bar_completion_pct"] = 1.0;
            bar_metadata["5min"]["bars_in_partial"] = 0;
            bar_metadata["5min"]["total_bars"] = static_cast<double>(tsla_slice.size());
            bar_metadata["5min"]["is_partial"] = 0.0;  // false

            if (batch_debug_count <= 5 && config_.verbose) {
                std::cout << "      [CREATE] About to create ChannelSample...\n";
                std::cout << "        tf_features.size()=" << tf_features.size() << "\n";
                std::cout << "        labels_per_window.size()=" << labels_per_window.size() << "\n";
            }

            // Create sample
            ChannelSample sample;
            sample.timestamp = sample_timestamp;
            sample.channel_end_idx = idx_5min;
            sample.tf_features = std::move(tf_features);
            sample.labels_per_window = std::move(labels_per_window);
            sample.bar_metadata = std::move(bar_metadata);
            sample.best_window = primary_window;

            // Debug: Log sample validation
            static int sample_count = 0;
            if (sample_count < 5 && config_.verbose) {
                std::cout << "      [SAMPLE #" << sample_count << "] "
                          << "timestamp=" << sample.timestamp
                          << " channel_end_idx=" << sample.channel_end_idx
                          << " best_window=" << sample.best_window
                          << " tf_features.size()=" << sample.tf_features.size()
                          << " labels_per_window.size()=" << sample.labels_per_window.size()
                          << " is_valid()=" << sample.is_valid() << "\n";
                sample_count++;
            }

            samples.push_back(std::move(sample));
            items_samples_created++;

        } catch (const std::exception& e) {
            // ALWAYS log exceptions during debugging
            std::cerr << "[EXCEPTION] ERROR processing channel (" << primary_tf << ","
                      << primary_window << "," << channel_idx << "): "
                      << e.what() << "\n";
            if (config_.strict) {
                throw;
            }
            // Continue processing other channels on error
        }
    }

    if (config_.verbose && this_batch_id == 0) {
        std::cout << "[PASS3 BATCH #" << this_batch_id << "] Complete:\n";
        std::cout << "  Items processed: " << items_processed << "\n";
        std::cout << "  Skipped (map not found): " << items_skipped_map_not_found << "\n";
        std::cout << "  Skipped (invalid idx): " << items_skipped_invalid_idx << "\n";
        std::cout << "  Skipped (channel invalid): " << items_skipped_channel_invalid << "\n";
        std::cout << "  Skipped (labels invalid): " << items_skipped_labels_invalid << "\n";
        std::cout << "  Skipped (index bounds): " << items_skipped_index_bounds << "\n";
        std::cout << "  Skipped (warmup): " << items_skipped_warmup << "\n";
        std::cout << "  Samples created: " << items_samples_created << "\n";
        std::cout << "  Samples returned: " << samples.size() << "\n";
    }

    return samples;
}


const SlimLabeledChannel* Scanner::find_channel_at_timestamp(
    const SlimLabeledChannelMap& slim_map,
    const std::string& tf,
    int window,
    int64_t timestamp
) const {
    TFWindowKey key{tf, window};
    auto it = slim_map.find(key);
    if (it == slim_map.end()) {
        return nullptr;
    }

    const std::vector<SlimLabeledChannel>& channels = it->second;
    if (channels.empty()) {
        return nullptr;
    }

    // Binary search for channel ending at/before timestamp
    int left = 0;
    int right = channels.size() - 1;
    const SlimLabeledChannel* found = nullptr;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        const SlimLabeledChannel& ch = channels[mid];

        if (ch.end_timestamp <= timestamp) {
            found = &ch;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return found;
}

// =============================================================================
// PROGRESS AND STATISTICS
// =============================================================================

void Scanner::log_progress(
    int valid_count,
    int total_expected,
    int64_t scan_start_ms,
    const std::vector<double>& feature_times_ms,
    int skipped_count,
    int error_count
) const {
    auto now = std::chrono::high_resolution_clock::now();
    int64_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - std::chrono::time_point<std::chrono::high_resolution_clock>(std::chrono::milliseconds(scan_start_ms))
    ).count();

    int processed = valid_count + skipped_count + error_count;
    double samples_per_sec = 0.0;
    int64_t eta_ms = 0;

    if (elapsed_ms > 0 && processed > 0) {
        samples_per_sec = (processed * 1000.0) / elapsed_ms;
        int remaining = total_expected - processed;
        eta_ms = (remaining * elapsed_ms) / processed;
    }

    double avg_feature_time = 0.0;
    if (!feature_times_ms.empty()) {
        for (double t : feature_times_ms) {
            avg_feature_time += t;
        }
        avg_feature_time /= feature_times_ms.size();
    }

    double progress_pct = (total_expected > 0) ? (100.0 * processed / total_expected) : 0.0;

    std::cout << "  [PROGRESS] " << valid_count << "/" << total_expected << " samples ("
              << std::fixed << std::setprecision(1) << progress_pct << "%) | "
              << "Rate: " << std::setprecision(1) << samples_per_sec << "/s | "
              << "ETA: " << format_time(eta_ms) << " | "
              << "Avg feature extraction: " << std::setprecision(1) << avg_feature_time << "ms | "
              << "Skipped: " << skipped_count << ", Errors: " << error_count << "\n";
}

void Scanner::print_summary() const {
    std::cout << "\n\n" << std::string(70, '=') << "\n";
    if (shutdown_requested_) {
        std::cout << "                         SCAN INTERRUPTED\n";
    } else {
        std::cout << "                         SCAN COMPLETE\n";
    }
    std::cout << std::string(70, '=') << "\n";

    // Results Summary
    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << "RESULTS SUMMARY\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << "  Total channels processed:     " << stats_.channels_processed << "\n";
    std::cout << "  Valid samples created:        " << stats_.samples_created << "\n";
    std::cout << "  Skipped (invalid/no labels):  " << stats_.samples_skipped << "\n";
    std::cout << "  Errors:                       " << stats_.errors_encountered << "\n";

    // Timing Breakdown
    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << "TIMING BREAKDOWN\n";
    std::cout << std::string(70, '-') << "\n";

    double pass1_sec = stats_.pass1_duration_ms / 1000.0;
    double pass2_sec = stats_.pass2_duration_ms / 1000.0;
    double pass3_sec = stats_.pass3_duration_ms / 1000.0;
    double total_sec = stats_.total_duration_ms / 1000.0;

    double pass1_pct = (total_sec > 0) ? (pass1_sec * 100.0 / total_sec) : 0.0;
    double pass2_pct = (total_sec > 0) ? (pass2_sec * 100.0 / total_sec) : 0.0;
    double pass3_pct = (total_sec > 0) ? (pass3_sec * 100.0 / total_sec) : 0.0;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  Pass 1 (channel detection):   " << std::setw(8) << pass1_sec << "s  ("
              << std::setw(5) << pass1_pct << "%)\n";
    std::cout << "  Pass 2 (label generation):    " << std::setw(8) << pass2_sec << "s  ("
              << std::setw(5) << pass2_pct << "%)\n";
    std::cout << "  Pass 3 (sample generation):   " << std::setw(8) << pass3_sec << "s  ("
              << std::setw(5) << pass3_pct << "%)\n";
    std::cout << "  " << std::string(40, '-') << "\n";
    std::cout << "  TOTAL WALL CLOCK TIME:        " << std::setw(8) << total_sec << "s  (100.0%)\n";

    // Performance Metrics
    std::cout << "\n" << std::string(70, '-') << "\n";
    std::cout << "PERFORMANCE METRICS\n";
    std::cout << std::string(70, '-') << "\n";

    if (stats_.samples_per_second > 0) {
        std::cout << "  Overall throughput:           " << std::setprecision(2)
                  << stats_.samples_per_second << " samples/sec\n";
    }
    if (stats_.channels_per_second_pass1 > 0) {
        std::cout << "  Pass 1 channel detection:     " << std::setprecision(2)
                  << stats_.channels_per_second_pass1 << " channels/sec\n";
    }
    if (stats_.labels_per_second_pass2 > 0) {
        std::cout << "  Pass 2 label generation:      " << std::setprecision(2)
                  << stats_.labels_per_second_pass2 << " labels/sec\n";
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "  COMPLETE: " << stats_.samples_created << " samples generated in "
              << std::setprecision(1) << total_sec << "s\n";
    std::cout << std::string(70, '=') << "\n";
}

std::string Scanner::format_time(int64_t ms) {
    if (ms < 0) {
        return "unknown";
    }

    int64_t seconds = ms / 1000;
    int64_t hours = seconds / 3600;
    int64_t minutes = (seconds % 3600) / 60;
    int64_t secs = seconds % 60;

    std::ostringstream oss;
    if (hours > 0) {
        oss << hours << "h " << std::setfill('0') << std::setw(2) << minutes << "m "
            << std::setw(2) << secs << "s";
    } else if (minutes > 0) {
        oss << minutes << "m " << std::setfill('0') << std::setw(2) << secs << "s";
    } else {
        oss << secs << "s";
    }

    return oss.str();
}

} // namespace v15
