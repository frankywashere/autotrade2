#pragma once

#include "types.hpp"
#include "sample.hpp"
#include "labels.hpp"
#include "data_loader.hpp"
#include "channel_detector.hpp"  // Provides Channel struct for Pass 1
#include "channel_history.hpp"   // For ChannelHistoryEntry
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include <thread>
#include <future>
#include <atomic>
#include <functional>
#include <cstdint>

namespace v15 {

// =============================================================================
// SLIM CHANNEL STRUCTURES (memory-efficient for workers)
// =============================================================================

/**
 * Memory-efficient channel with precomputed labels for worker processes.
 *
 * ARCHITECTURAL NOTE:
 * - end_idx IS the sample position (in TF-space)
 * - end_timestamp IS the sample timestamp
 * - labels are PRECOMPUTED in Pass 2 - use them directly, no lookup needed
 *
 * Memory reduction: Strips heavy arrays (upper_line, lower_line, center_line)
 * from channels. ~100x reduction per map (from GBs to MBs).
 */
struct SlimLabeledChannel {
    int64_t start_timestamp;        // Unix timestamp in milliseconds
    int64_t end_timestamp;          // Unix timestamp in milliseconds (sample position)
    int start_idx;                  // Start index in TF-space
    int end_idx;                    // End index in TF-space (THIS IS THE SAMPLE POSITION)

    // Channel regression parameters
    double channel_slope;
    double channel_intercept;
    double channel_std_dev;
    double channel_r_squared;
    int channel_direction;          // 0=BEAR, 1=SIDEWAYS, 2=BULL
    bool channel_valid;
    int channel_window;
    int channel_bounce_count;

    std::string tf;                 // Timeframe string (e.g., "1h")

    // PRECOMPUTED labels from Pass 2 - USE DIRECTLY
    ChannelLabels labels;

    // Touch data for feature extraction (from Channel.touches)
    std::vector<Touch> touches;

    // Cached line values for position_in_channel calculation
    double first_upper_val = 0.0;
    double last_upper_val = 0.0;
    double first_lower_val = 0.0;
    double last_lower_val = 0.0;
    double first_center_val = 0.0;
    double last_center_val = 0.0;
    std::array<double, 5> upper_line_tail{};
    std::array<double, 5> lower_line_tail{};
    int tail_count = 0;

    SlimLabeledChannel()
        : start_timestamp(0)
        , end_timestamp(0)
        , start_idx(0)
        , end_idx(0)
        , channel_slope(0.0)
        , channel_intercept(0.0)
        , channel_std_dev(0.0)
        , channel_r_squared(0.0)
        , channel_direction(-1)
        , channel_valid(false)
        , channel_window(0)
        , channel_bounce_count(0)
        , tail_count(0)
    {}
};

/**
 * Key for slim channel map: (timeframe, window)
 */
struct TFWindowKey {
    std::string tf;
    int window;

    bool operator==(const TFWindowKey& other) const {
        return tf == other.tf && window == other.window;
    }
};

/**
 * Hash function for TFWindowKey
 */
struct TFWindowKeyHash {
    std::size_t operator()(const TFWindowKey& k) const {
        return std::hash<std::string>()(k.tf) ^ (std::hash<int>()(k.window) << 1);
    }
};

/**
 * Slim labeled channel map: (tf, window) -> list of labeled channels
 */
using SlimLabeledChannelMap = std::unordered_map<TFWindowKey, std::vector<SlimLabeledChannel>, TFWindowKeyHash>;

// =============================================================================
// CHANNEL WORK ITEM
// =============================================================================

/**
 * Work item for parallel channel processing.
 * Each item represents ONE channel that will produce ONE sample.
 */
struct ChannelWorkItem {
    std::string primary_tf;         // Primary channel timeframe
    int primary_window;             // Primary channel window
    int channel_idx;                // Index in the slim_map for this (tf, window)
    int64_t end_timestamp;          // Channel end timestamp for sorting

    // Pre-computed history snapshots for each timeframe
    // Maps timeframe string (e.g., "5min", "1h") to vector of last 5 channels
    std::unordered_map<std::string, std::vector<ChannelHistoryEntry>> tsla_history_by_tf;
    std::unordered_map<std::string, std::vector<ChannelHistoryEntry>> spy_history_by_tf;

    ChannelWorkItem()
        : primary_tf(""), primary_window(0), channel_idx(0), end_timestamp(0) {}

    ChannelWorkItem(const std::string& tf, int win, int idx, int64_t ts = 0)
        : primary_tf(tf), primary_window(win), channel_idx(idx), end_timestamp(ts) {}
};

// =============================================================================
// SCANNER STATISTICS
// =============================================================================

/**
 * Scanner performance and progress statistics
 */
struct ScannerStats {
    // Pass 1: Channel detection
    int64_t pass1_duration_ms;
    int tsla_channels_detected;
    int spy_channels_detected;

    // Pass 2: Label generation
    int64_t pass2_duration_ms;
    int tsla_labels_generated;
    int spy_labels_generated;
    int tsla_labels_valid;
    int spy_labels_valid;

    // Pass 3: Sample generation
    int64_t pass3_duration_ms;
    int channels_processed;
    int samples_created;
    int samples_skipped;
    int errors_encountered;

    // Label lookup stats
    int label_hits;
    int label_misses;

    // Feature extraction timing
    double avg_feature_time_ms;
    double min_feature_time_ms;
    double max_feature_time_ms;

    // Overall timing
    int64_t total_duration_ms;

    // Throughput metrics
    double samples_per_second;
    double channels_per_second_pass1;
    double labels_per_second_pass2;

    // Memory usage (if available)
    double memory_rss_mb;
    double memory_vms_mb;
    double memory_percent;

    ScannerStats()
        : pass1_duration_ms(0)
        , tsla_channels_detected(0)
        , spy_channels_detected(0)
        , pass2_duration_ms(0)
        , tsla_labels_generated(0)
        , spy_labels_generated(0)
        , tsla_labels_valid(0)
        , spy_labels_valid(0)
        , pass3_duration_ms(0)
        , channels_processed(0)
        , samples_created(0)
        , samples_skipped(0)
        , errors_encountered(0)
        , label_hits(0)
        , label_misses(0)
        , avg_feature_time_ms(0.0)
        , min_feature_time_ms(0.0)
        , max_feature_time_ms(0.0)
        , total_duration_ms(0)
        , samples_per_second(0.0)
        , channels_per_second_pass1(0.0)
        , labels_per_second_pass2(0.0)
        , memory_rss_mb(0.0)
        , memory_vms_mb(0.0)
        , memory_percent(0.0)
    {}
};

// =============================================================================
// SCANNER CONFIGURATION
// =============================================================================

/**
 * Scanner configuration parameters
 */
struct ScannerConfig {
    // Pass 1: Channel detection parameters
    int step;                       // Step size for channel detection (default: 10)
    int min_cycles;                 // Min cycles for valid channel (default: 1)
    int min_gap_bars;               // Min gap between channels (default: 5)

    // Pass 2: Label generation parameters
    std::string labeling_method;    // "hybrid", "first_break", etc. (default: "hybrid")

    // Pass 3: Sample generation parameters
    int warmup_bars;                // Min 5min bars before first sample (default: 32760)
    int max_samples;                // Max samples to generate (0 = unlimited)

    // Parallelization
    int workers;                    // Worker threads (0 = auto-detect)
    int batch_size;                 // Channels per batch (default: 8)

    // Progress and logging
    bool progress;                  // Show progress bar (default: true)
    bool verbose;                   // Verbose logging (default: true)
    bool strict;                    // Raise on errors (default: true)

    // Output
    std::string output_path;        // Output file path (empty = don't save)

    // Streaming mode (for large datasets)
    bool streaming;                 // Enable streaming output (default: true for large scans)
    size_t flush_interval;          // Samples between disk flushes (default: 1000)

    ScannerConfig()
        : step(10)
        , min_cycles(1)
        , min_gap_bars(5)
        , labeling_method("hybrid")
        , warmup_bars(32760)
        , max_samples(0)
        , workers(0)
        , batch_size(8)
        , progress(true)
        , verbose(true)
        , strict(false)  // Allow partial features for early channels
        , streaming(true)           // Enable streaming by default
        , flush_interval(1000)      // Flush every 1000 samples
    {}
};

// =============================================================================
// THREAD POOL
// =============================================================================

/**
 * Simple thread pool for parallel batch processing.
 *
 * Workers are initialized once with shared data (DataFrames, slim maps).
 * Tasks are submitted as batches of channel work items.
 */
class ThreadPool {
public:
    /**
     * Create thread pool with specified number of workers.
     * If workers=0, uses std::thread::hardware_concurrency().
     */
    explicit ThreadPool(size_t workers = 0);

    /**
     * Destructor - waits for all tasks to complete
     */
    ~ThreadPool();

    /**
     * Submit a task to the pool.
     * Returns a future that will contain the result.
     */
    template<typename Func, typename... Args>
    auto submit(Func&& func, Args&&... args) -> std::future<typename std::invoke_result<Func, Args...>::type>;

    /**
     * Get number of workers in pool
     */
    size_t size() const { return workers_.size(); }

    /**
     * Shutdown the pool (wait for all tasks to complete)
     */
    void shutdown();

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_;

    void worker_thread();
};

// =============================================================================
// SCANNER
// =============================================================================

/**
 * V15 Channel Scanner - 3-Pass Architecture
 *
 * SIMPLIFIED ARCHITECTURE:
 * 1. PASS 1: Detect all channels across the dataset (detect_all_channels)
 * 2. PASS 2: Compute labels at channel END positions (generate_all_labels)
 * 3. PASS 3: Iterate over detected channels. Each channel = ONE sample at its end_idx.
 *
 * KEY PRINCIPLE - ONE SAMPLE PER CHANNEL:
 * - We iterate over DETECTED CHANNELS from the slim_map
 * - Each channel's end_idx IS the sample position
 * - Labels are PRECOMPUTED in Pass 2 and stored in SlimLabeledChannel
 * - For the PRIMARY channel, use its labels directly (no lookup needed)
 * - For OTHER TF/window combinations at same timestamp, do binary search lookup
 *
 * The --step parameter controls CHANNEL DETECTION spacing in Pass 1.
 * Number of samples = number of valid detected channels.
 *
 * PARALLEL ARCHITECTURE:
 * - Pass 1: Parallel by timeframe
 * - Pass 2: Parallel by timeframe/window combination
 * - Pass 3: Parallel batch processing of channels
 */
class Scanner {
public:
    /**
     * Constructor
     *
     * @param config Scanner configuration
     */
    explicit Scanner(const ScannerConfig& config = ScannerConfig());

    /**
     * Destructor
     */
    ~Scanner();

    /**
     * Run the 3-pass channel scanner
     *
     * @param tsla_df TSLA OHLCV data (5min resolution)
     * @param spy_df SPY OHLCV data (aligned to TSLA)
     * @param vix_df VIX OHLCV data (aligned to TSLA)
     * @return Vector of ChannelSample objects
     */
    std::vector<ChannelSample> scan(
        const std::vector<OHLCV>& tsla_df,
        const std::vector<OHLCV>& spy_df,
        const std::vector<OHLCV>& vix_df
    );

    /**
     * Get scanner statistics from last run
     */
    const ScannerStats& get_stats() const { return stats_; }

    /**
     * Set configuration
     */
    void set_config(const ScannerConfig& config) { config_ = config; }

    /**
     * Get configuration
     */
    const ScannerConfig& get_config() const { return config_; }

private:
    ScannerConfig config_;
    ScannerStats stats_;
    std::atomic<bool> shutdown_requested_;

    // ==========================================================================
    // PASS 1: Channel Detection
    // ==========================================================================

    /**
     * Detect all channels for a single asset across all timeframes and windows.
     * Runs in parallel by timeframe.
     */
    void detect_all_channels(
        const std::vector<OHLCV>& df,
        const std::string& asset_name,
        std::unordered_map<TFWindowKey, std::vector<Channel>, TFWindowKeyHash>& channel_map,
        std::unordered_map<std::string, std::vector<OHLCV>>& resampled_dfs
    );

    // ==========================================================================
    // PASS 2: Label Generation
    // ==========================================================================

    /**
     * Generate labels for all channels from a channel map.
     * Runs in parallel by (timeframe, window) combination.
     */
    void generate_all_labels(
        const std::unordered_map<TFWindowKey, std::vector<Channel>, TFWindowKeyHash>& channel_map,
        const std::unordered_map<std::string, std::vector<OHLCV>>& resampled_dfs,
        SlimLabeledChannelMap& labeled_map
    );

    /**
     * Create slim labeled channel map from full labeled map.
     * Strips heavy arrays to reduce memory by ~100x.
     */
    SlimLabeledChannelMap create_slim_labeled_map(
        const std::unordered_map<TFWindowKey, std::vector<Channel>, TFWindowKeyHash>& channel_map,
        const std::unordered_map<std::string, std::vector<OHLCV>>& resampled_dfs
    );

    // ==========================================================================
    // PASS 3: Sample Generation
    // ==========================================================================

    /**
     * Process a batch of channels to generate samples.
     * This is the worker function called by thread pool.
     */
    std::vector<ChannelSample> process_channel_batch(
        const std::vector<ChannelWorkItem>& batch,
        const std::vector<OHLCV>& tsla_df,
        const std::vector<OHLCV>& spy_df,
        const std::vector<OHLCV>& vix_df,
        const SlimLabeledChannelMap& tsla_slim_map,
        const SlimLabeledChannelMap& spy_slim_map
    );

    /**
     * Find channel at or before timestamp using binary search.
     * O(log N) complexity.
     */
    const SlimLabeledChannel* find_channel_at_timestamp(
        const SlimLabeledChannelMap& slim_map,
        const std::string& tf,
        int window,
        int64_t timestamp
    ) const;

    // ==========================================================================
    // Progress and Statistics
    // ==========================================================================

    /**
     * Log progress information
     */
    void log_progress(
        int valid_count,
        int total_expected,
        int64_t scan_start_ms,
        const std::vector<double>& feature_times_ms,
        int skipped_count,
        int error_count
    ) const;

    /**
     * Print summary statistics
     */
    void print_summary() const;

    /**
     * Format time duration to human-readable string
     */
    static std::string format_time(int64_t ms);
};

} // namespace v15
