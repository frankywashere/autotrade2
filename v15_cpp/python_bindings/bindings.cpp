/**
 * v15scanner Python Bindings using pybind11
 *
 * Exposes the C++ v15 scanner to Python with:
 * - Scanner class with scan() method
 * - ChannelSample structure with proper conversions
 * - Configuration options (step, max_samples, workers, etc.)
 * - Statistics and performance metrics
 * - Seamless conversion between C++ and Python types
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include <vector>
#include <string>
#include <unordered_map>

// v15 includes
#include "scanner.hpp"
#include "sample.hpp"
#include "labels.hpp"
#include "types.hpp"
#include "feature_extractor.hpp"

namespace py = pybind11;

// =============================================================================
// Helper Functions for Type Conversion
// =============================================================================

/**
 * Convert Unix timestamp (ms) to Python datetime
 */
py::object ms_to_datetime(int64_t timestamp_ms) {
    py::module_ datetime_mod = py::module_::import("datetime");
    py::object datetime_class = datetime_mod.attr("datetime");
    py::object timezone = datetime_mod.attr("timezone");
    py::object utc = timezone.attr("utc");

    // Convert ms to seconds and microseconds
    int64_t seconds = timestamp_ms / 1000;
    int64_t microseconds = (timestamp_ms % 1000) * 1000;

    return datetime_class.attr("fromtimestamp")(
        py::float_(seconds + microseconds / 1000000.0),
        utc
    );
}

/**
 * Convert Python datetime to Unix timestamp (seconds) for OHLCV
 */
std::time_t datetime_to_time_t(py::object dt) {
    py::object timestamp = dt.attr("timestamp")();
    double ts_seconds = timestamp.cast<double>();
    return static_cast<std::time_t>(ts_seconds);
}

/**
 * Convert Python datetime to Unix timestamp (ms) for ChannelSample
 */
int64_t datetime_to_ms(py::object dt) {
    py::object timestamp = dt.attr("timestamp")();
    double ts_seconds = timestamp.cast<double>();
    return static_cast<int64_t>(ts_seconds * 1000);
}

/**
 * Convert Unix timestamp (seconds) to Python datetime for OHLCV
 */
py::object time_t_to_datetime(std::time_t timestamp_sec) {
    py::module_ datetime_mod = py::module_::import("datetime");
    py::object datetime_class = datetime_mod.attr("datetime");
    py::object timezone = datetime_mod.attr("timezone");
    py::object utc = timezone.attr("utc");

    return datetime_class.attr("fromtimestamp")(
        static_cast<double>(timestamp_sec),
        utc
    );
}

/**
 * Convert C++ OHLCV vector to Python list of dicts
 */
py::list ohlcv_to_python(const std::vector<v15::OHLCV>& data) {
    py::list result;
    for (const auto& bar : data) {
        py::dict d;
        d["timestamp"] = time_t_to_datetime(bar.timestamp);
        d["open"] = bar.open;
        d["high"] = bar.high;
        d["low"] = bar.low;
        d["close"] = bar.close;
        d["volume"] = bar.volume;
        result.append(d);
    }
    return result;
}

/**
 * Convert Python DataFrame/list to C++ OHLCV vector
 */
std::vector<v15::OHLCV> python_to_ohlcv(py::object data) {
    std::vector<v15::OHLCV> result;

    // Check if it's a pandas DataFrame
    if (py::hasattr(data, "iterrows")) {
        // It's a DataFrame
        py::object iterrows = data.attr("iterrows")();

        for (auto item : iterrows) {
            py::tuple row_tuple = item.cast<py::tuple>();
            py::object timestamp = row_tuple[0];
            py::object row = row_tuple[1];

            v15::OHLCV bar;
            bar.timestamp = datetime_to_time_t(timestamp);
            bar.open = row["open"].cast<double>();
            bar.high = row["high"].cast<double>();
            bar.low = row["low"].cast<double>();
            bar.close = row["close"].cast<double>();
            bar.volume = row["volume"].cast<double>();

            result.push_back(bar);
        }
    } else if (py::isinstance<py::list>(data)) {
        // It's a list of dicts
        py::list data_list = data.cast<py::list>();
        for (auto item : data_list) {
            py::dict d = item.cast<py::dict>();

            v15::OHLCV bar;
            bar.timestamp = datetime_to_time_t(d["timestamp"]);
            bar.open = d["open"].cast<double>();
            bar.high = d["high"].cast<double>();
            bar.low = d["low"].cast<double>();
            bar.close = d["close"].cast<double>();
            bar.volume = d["volume"].cast<double>();

            result.push_back(bar);
        }
    } else {
        throw std::runtime_error("Data must be a pandas DataFrame or list of dicts");
    }

    return result;
}

/**
 * Convert C++ ChannelLabels to Python dict
 */
py::dict labels_to_dict(const v15::ChannelLabels& labels) {
    py::dict d;

    // Prediction targets
    d["duration_bars"] = labels.duration_bars;
    d["next_channel_direction"] = labels.next_channel_direction;
    d["permanent_break"] = labels.permanent_break;
    d["timeframe"] = v15::timeframe_to_string(labels.timeframe);

    // Break scan features
    d["break_direction"] = labels.break_direction;
    d["break_magnitude"] = labels.break_magnitude;
    d["bars_to_first_break"] = labels.bars_to_first_break;
    d["returned_to_channel"] = labels.returned_to_channel;
    d["bounces_after_return"] = labels.bounces_after_return;
    d["round_trip_bounces"] = labels.round_trip_bounces;
    d["channel_continued"] = labels.channel_continued;

    // Permanent break
    d["permanent_break_direction"] = labels.permanent_break_direction;
    d["permanent_break_magnitude"] = labels.permanent_break_magnitude;
    d["bars_to_permanent_break"] = labels.bars_to_permanent_break;

    // Exit dynamics
    d["duration_to_permanent"] = labels.duration_to_permanent;
    d["avg_bars_outside"] = labels.avg_bars_outside;
    d["total_bars_outside"] = labels.total_bars_outside;
    d["durability_score"] = labels.durability_score;

    // Source channel parameters
    d["source_channel_slope"] = labels.source_channel_slope;
    d["source_channel_intercept"] = labels.source_channel_intercept;
    d["source_channel_std_dev"] = labels.source_channel_std_dev;
    d["source_channel_r_squared"] = labels.source_channel_r_squared;
    d["source_channel_direction"] = labels.source_channel_direction;
    d["source_channel_start_ts"] = labels.source_channel_start_ts;
    d["source_channel_end_ts"] = labels.source_channel_end_ts;

    // Next channel detection
    d["best_next_channel_direction"] = labels.best_next_channel_direction;
    d["best_next_channel_bars_away"] = labels.best_next_channel_bars_away;
    d["best_next_channel_duration"] = labels.best_next_channel_duration;
    d["best_next_channel_r_squared"] = labels.best_next_channel_r_squared;
    d["best_next_channel_bounce_count"] = labels.best_next_channel_bounce_count;

    d["shortest_next_channel_direction"] = labels.shortest_next_channel_direction;
    d["shortest_next_channel_bars_away"] = labels.shortest_next_channel_bars_away;
    d["shortest_next_channel_duration"] = labels.shortest_next_channel_duration;
    d["small_channels_before_best"] = labels.small_channels_before_best;

    // RSI labels
    d["rsi_at_first_break"] = labels.rsi_at_first_break;
    d["rsi_at_permanent_break"] = labels.rsi_at_permanent_break;
    d["rsi_at_channel_end"] = labels.rsi_at_channel_end;
    d["rsi_overbought_at_break"] = labels.rsi_overbought_at_break;
    d["rsi_oversold_at_break"] = labels.rsi_oversold_at_break;
    d["rsi_divergence_at_break"] = labels.rsi_divergence_at_break;
    d["rsi_trend_in_channel"] = labels.rsi_trend_in_channel;
    d["rsi_range_in_channel"] = labels.rsi_range_in_channel;

    // Validity flags
    d["duration_valid"] = labels.duration_valid;
    d["direction_valid"] = labels.direction_valid;
    d["next_channel_valid"] = labels.next_channel_valid;
    d["break_scan_valid"] = labels.break_scan_valid;

    return d;
}

/**
 * Convert C++ ChannelSample to Python dict (for pickle compatibility)
 */
py::dict sample_to_dict(const v15::ChannelSample& sample) {
    py::dict d;

    // Core data
    d["timestamp"] = ms_to_datetime(sample.timestamp);
    d["channel_end_idx"] = sample.channel_end_idx;
    d["best_window"] = sample.best_window;

    // Features (convert unordered_map to dict)
    py::dict features;
    for (const auto& kv : sample.tf_features) {
        features[py::str(kv.first)] = kv.second;
    }
    d["tf_features"] = features;

    // Labels per window
    py::dict labels_per_window;
    for (const auto& win_pair : sample.labels_per_window) {
        int window = win_pair.first;
        py::dict tf_labels;

        for (const auto& tf_pair : win_pair.second) {
            const std::string& tf = tf_pair.first;
            const v15::ChannelLabels& labels = tf_pair.second;
            tf_labels[py::str(tf)] = labels_to_dict(labels);
        }

        labels_per_window[py::int_(window)] = tf_labels;
    }
    d["labels_per_window"] = labels_per_window;

    // Bar metadata
    py::dict bar_metadata;
    for (const auto& tf_pair : sample.bar_metadata) {
        const std::string& tf = tf_pair.first;
        py::dict metadata;

        for (const auto& meta_pair : tf_pair.second) {
            metadata[py::str(meta_pair.first)] = meta_pair.second;
        }

        bar_metadata[py::str(tf)] = metadata;
    }
    d["bar_metadata"] = bar_metadata;

    return d;
}

// =============================================================================
// Native TF Data Conversion Helper
// =============================================================================

/**
 * Convert Python dict to C++ NativeTFData.
 *
 * Expected Python format:
 *   {'tsla': {'daily': DataFrame, ...}, 'spy': {...}, 'vix': {...}}
 *
 * Keys are lowercase asset names and timeframe strings matching
 * timeframe_to_string() output (e.g., "daily", "weekly", "monthly").
 */
v15::NativeTFData convert_native_tf_dict(py::dict dict) {
    v15::NativeTFData result;
    for (auto& [asset_key, tf_dict_obj] : dict) {
        std::string asset = asset_key.cast<std::string>();
        auto& target = (asset == "tsla") ? result.tsla :
                       (asset == "spy")  ? result.spy : result.vix;
        py::dict tf_dict = tf_dict_obj.cast<py::dict>();
        for (auto& [tf_key, df_obj] : tf_dict) {
            std::string tf_name = tf_key.cast<std::string>();
            py::object df = df_obj.cast<py::object>();
            // Skip None or empty DataFrames
            if (df.is_none()) continue;
            if (py::hasattr(df, "__len__") && py::len(df) == 0) continue;
            target[tf_name] = python_to_ohlcv(df);
        }
    }
    return result;
}

// =============================================================================
// Python Module Definition
// =============================================================================

PYBIND11_MODULE(v15scanner_cpp, m) {
    m.doc() = R"pbdoc(
        v15scanner C++ Backend
        ----------------------

        High-performance C++ implementation of the v15 channel scanner.

        This module provides a drop-in replacement for the Python scanner with
        significant performance improvements through parallel processing and
        optimized algorithms.

        Example usage:
            import v15scanner_cpp

            # Create scanner with configuration
            config = v15scanner_cpp.ScannerConfig()
            config.step = 10
            config.workers = 8
            config.max_samples = 10000

            scanner = v15scanner_cpp.Scanner(config)

            # Run scan (requires pandas DataFrames)
            samples = scanner.scan(tsla_df, spy_df, vix_df)

            # Get statistics
            stats = scanner.get_stats()
            print(f"Generated {stats.samples_created} samples")
            print(f"Total time: {stats.total_duration_ms / 1000:.1f}s")
    )pbdoc";

    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("backend") = "cpp";

    // =========================================================================
    // ScannerConfig
    // =========================================================================

    py::class_<v15::ScannerConfig>(m, "ScannerConfig", R"pbdoc(
        Scanner configuration parameters.

        Attributes:
            step (int): Step size for channel detection in Pass 1 (default: 10)
            min_cycles (int): Minimum cycles for valid channel (default: 1)
            min_gap_bars (int): Minimum gap between channels (default: 5)
            labeling_method (str): Labeling method - "hybrid", "first_break", etc. (default: "hybrid")
            warmup_bars (int): Minimum 5min bars before first sample (default: 32760)
            max_samples (int): Maximum samples to generate (0 = unlimited, default: 0)
            workers (int): Number of worker threads (0 = auto-detect, default: 0)
            batch_size (int): Channels per batch for parallel processing (default: 8)
            progress (bool): Show progress indicators (default: True)
            verbose (bool): Verbose logging (default: True)
            strict (bool): Raise exceptions on errors (default: True)
            output_path (str): Output file path for saving results (default: "")
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("step", &v15::ScannerConfig::step)
        .def_readwrite("min_cycles", &v15::ScannerConfig::min_cycles)
        .def_readwrite("min_gap_bars", &v15::ScannerConfig::min_gap_bars)
        .def_readwrite("labeling_method", &v15::ScannerConfig::labeling_method)
        .def_readwrite("warmup_bars", &v15::ScannerConfig::warmup_bars)
        .def_readwrite("max_samples", &v15::ScannerConfig::max_samples)
        .def_readwrite("workers", &v15::ScannerConfig::workers)
        .def_readwrite("batch_size", &v15::ScannerConfig::batch_size)
        .def_readwrite("progress", &v15::ScannerConfig::progress)
        .def_readwrite("verbose", &v15::ScannerConfig::verbose)
        .def_readwrite("strict", &v15::ScannerConfig::strict)
        .def_readwrite("output_path", &v15::ScannerConfig::output_path)
        .def("__repr__", [](const v15::ScannerConfig& c) {
            return "<ScannerConfig step=" + std::to_string(c.step) +
                   " workers=" + std::to_string(c.workers) +
                   " max_samples=" + std::to_string(c.max_samples) + ">";
        });

    // =========================================================================
    // ScannerStats
    // =========================================================================

    py::class_<v15::ScannerStats>(m, "ScannerStats", R"pbdoc(
        Scanner performance and progress statistics.

        Contains detailed metrics from the 3-pass scanning process including
        timing breakdowns, throughput rates, and memory usage.
    )pbdoc")
        .def(py::init<>())
        // Pass 1 stats
        .def_readonly("pass1_duration_ms", &v15::ScannerStats::pass1_duration_ms)
        .def_readonly("tsla_channels_detected", &v15::ScannerStats::tsla_channels_detected)
        .def_readonly("spy_channels_detected", &v15::ScannerStats::spy_channels_detected)
        // Pass 2 stats
        .def_readonly("pass2_duration_ms", &v15::ScannerStats::pass2_duration_ms)
        .def_readonly("tsla_labels_generated", &v15::ScannerStats::tsla_labels_generated)
        .def_readonly("spy_labels_generated", &v15::ScannerStats::spy_labels_generated)
        .def_readonly("tsla_labels_valid", &v15::ScannerStats::tsla_labels_valid)
        .def_readonly("spy_labels_valid", &v15::ScannerStats::spy_labels_valid)
        // Pass 3 stats
        .def_readonly("pass3_duration_ms", &v15::ScannerStats::pass3_duration_ms)
        .def_readonly("channels_processed", &v15::ScannerStats::channels_processed)
        .def_readonly("samples_created", &v15::ScannerStats::samples_created)
        .def_readonly("samples_skipped", &v15::ScannerStats::samples_skipped)
        .def_readonly("errors_encountered", &v15::ScannerStats::errors_encountered)
        // Label lookup stats
        .def_readonly("label_hits", &v15::ScannerStats::label_hits)
        .def_readonly("label_misses", &v15::ScannerStats::label_misses)
        // Timing
        .def_readonly("avg_feature_time_ms", &v15::ScannerStats::avg_feature_time_ms)
        .def_readonly("min_feature_time_ms", &v15::ScannerStats::min_feature_time_ms)
        .def_readonly("max_feature_time_ms", &v15::ScannerStats::max_feature_time_ms)
        .def_readonly("total_duration_ms", &v15::ScannerStats::total_duration_ms)
        // Throughput
        .def_readonly("samples_per_second", &v15::ScannerStats::samples_per_second)
        .def_readonly("channels_per_second_pass1", &v15::ScannerStats::channels_per_second_pass1)
        .def_readonly("labels_per_second_pass2", &v15::ScannerStats::labels_per_second_pass2)
        // Memory
        .def_readonly("memory_rss_mb", &v15::ScannerStats::memory_rss_mb)
        .def_readonly("memory_vms_mb", &v15::ScannerStats::memory_vms_mb)
        .def_readonly("memory_percent", &v15::ScannerStats::memory_percent)
        .def("__repr__", [](const v15::ScannerStats& s) {
            return "<ScannerStats samples=" + std::to_string(s.samples_created) +
                   " time=" + std::to_string(s.total_duration_ms / 1000.0) + "s" +
                   " throughput=" + std::to_string(s.samples_per_second) + "/s>";
        });

    // =========================================================================
    // ChannelLabels
    // =========================================================================

    py::class_<v15::ChannelLabels>(m, "ChannelLabels", R"pbdoc(
        Channel labels structure.

        Contains all prediction targets and break scan features for a single
        channel at a specific window size and timeframe.
    )pbdoc")
        .def(py::init<>())
        .def("to_dict", &labels_to_dict, "Convert labels to Python dictionary")
        // Key fields (expose most commonly used ones)
        .def_readwrite("duration_bars", &v15::ChannelLabels::duration_bars)
        .def_readwrite("next_channel_direction", &v15::ChannelLabels::next_channel_direction)
        .def_readwrite("permanent_break", &v15::ChannelLabels::permanent_break)
        .def_readwrite("timeframe", &v15::ChannelLabels::timeframe)
        .def_readwrite("duration_valid", &v15::ChannelLabels::duration_valid)
        .def_readwrite("direction_valid", &v15::ChannelLabels::direction_valid)
        .def("__repr__", [](const v15::ChannelLabels& l) {
            return "<ChannelLabels tf=" + std::string(v15::timeframe_to_string(l.timeframe)) +
                   " duration=" + std::to_string(l.duration_bars) +
                   " direction=" + std::to_string(l.next_channel_direction) + ">";
        });

    // =========================================================================
    // ChannelSample
    // =========================================================================

    py::class_<v15::ChannelSample>(m, "ChannelSample", R"pbdoc(
        A complete channel sample for prediction.

        Contains features, labels, and metadata for a single sample point
        (channel end position).

        Attributes:
            timestamp: Sample timestamp (channel end time) as datetime
            channel_end_idx: Index in 5min data where channel ends
            best_window: Optimal window size for this sample
            tf_features: Dictionary of all features (TF-prefixed)
            labels_per_window: Nested dict of labels by window and timeframe
            bar_metadata: Bar completion metadata by timeframe
    )pbdoc")
        .def(py::init<>())
        .def_property("timestamp",
            [](const v15::ChannelSample& s) { return ms_to_datetime(s.timestamp); },
            [](v15::ChannelSample& s, py::object dt) { s.timestamp = datetime_to_ms(dt); })
        .def_readwrite("channel_end_idx", &v15::ChannelSample::channel_end_idx)
        .def_readwrite("best_window", &v15::ChannelSample::best_window)
        .def_property("tf_features",
            [](const v15::ChannelSample& s) {
                py::dict d;
                for (const auto& kv : s.tf_features) {
                    d[py::str(kv.first)] = kv.second;
                }
                return d;
            },
            [](v15::ChannelSample& s, py::dict features) {
                s.tf_features.clear();
                for (auto item : features) {
                    s.tf_features[item.first.cast<std::string>()] = item.second.cast<double>();
                }
            })
        .def("to_dict", &sample_to_dict, "Convert sample to Python dictionary for pickle")
        .def("get_feature", &v15::ChannelSample::get_feature,
             py::arg("key"), py::arg("default_val") = 0.0,
             "Get feature value by key")
        .def("set_feature", &v15::ChannelSample::set_feature,
             py::arg("key"), py::arg("value"),
             "Set feature value")
        .def("has_feature", &v15::ChannelSample::has_feature,
             py::arg("key"),
             "Check if feature exists")
        .def("feature_count", &v15::ChannelSample::feature_count,
             "Get total feature count")
        .def("label_count", &v15::ChannelSample::label_count,
             "Get total label count")
        .def("is_valid", &v15::ChannelSample::is_valid,
             "Check if sample is valid")
        .def("__repr__", [](const v15::ChannelSample& s) {
            return "<ChannelSample idx=" + std::to_string(s.channel_end_idx) +
                   " features=" + std::to_string(s.tf_features.size()) +
                   " window=" + std::to_string(s.best_window) + ">";
        });

    // =========================================================================
    // Scanner
    // =========================================================================

    py::class_<v15::Scanner>(m, "Scanner", R"pbdoc(
        V15 Channel Scanner - 3-Pass Architecture.

        High-performance C++ implementation of the channel scanner with:
        - Pass 1: Parallel channel detection across all timeframes
        - Pass 2: Parallel label generation for all detected channels
        - Pass 3: Parallel sample generation with feature extraction

        Example:
            config = v15scanner_cpp.ScannerConfig()
            config.step = 10
            config.workers = 8

            scanner = v15scanner_cpp.Scanner(config)
            samples = scanner.scan(tsla_df, spy_df, vix_df)

            stats = scanner.get_stats()
            print(f"Generated {stats.samples_created} samples in {stats.total_duration_ms/1000:.1f}s")
    )pbdoc")
        .def(py::init<>(), "Create scanner with default configuration")
        .def(py::init<const v15::ScannerConfig&>(),
             py::arg("config"),
             "Create scanner with custom configuration")
        .def("scan",
             [](v15::Scanner& scanner, py::object tsla_df, py::object spy_df, py::object vix_df) {
                 // Convert Python DataFrames to C++ vectors
                 std::vector<v15::OHLCV> tsla_data = python_to_ohlcv(tsla_df);
                 std::vector<v15::OHLCV> spy_data = python_to_ohlcv(spy_df);
                 std::vector<v15::OHLCV> vix_data = python_to_ohlcv(vix_df);

                 // Run scanner
                 std::vector<v15::ChannelSample> samples = scanner.scan(tsla_data, spy_data, vix_data);

                 // Convert samples to Python dicts for pickle compatibility
                 py::list result;
                 for (const auto& sample : samples) {
                     result.append(sample_to_dict(sample));
                 }

                 return result;
             },
             py::arg("tsla_df"),
             py::arg("spy_df"),
             py::arg("vix_df"),
             R"pbdoc(
                 Run the 3-pass channel scanner.

                 Args:
                     tsla_df: TSLA OHLCV DataFrame with DatetimeIndex
                     spy_df: SPY OHLCV DataFrame aligned to TSLA
                     vix_df: VIX OHLCV DataFrame aligned to TSLA

                 Returns:
                     List of ChannelSample dictionaries (pickle-compatible)
             )pbdoc")
        .def("get_stats", &v15::Scanner::get_stats,
             py::return_value_policy::reference_internal,
             "Get scanner statistics from last run")
        .def("get_config", &v15::Scanner::get_config,
             py::return_value_policy::reference_internal,
             "Get current configuration")
        .def("set_config", &v15::Scanner::set_config,
             py::arg("config"),
             "Set scanner configuration")
        .def("__repr__", [](const v15::Scanner& s) {
            const auto& cfg = s.get_config();
            return "<Scanner step=" + std::to_string(cfg.step) +
                   " workers=" + std::to_string(cfg.workers) + ">";
        });

    // =========================================================================
    // Feature Extraction (for live inference)
    // =========================================================================

    m.def("extract_features",
        [](py::object tsla_df, py::object spy_df, py::object vix_df,
           int64_t timestamp_ms, int source_bar_count,
           py::object native_tf_dict) -> py::dict {
            // Convert DataFrames to C++ OHLCV vectors
            std::vector<v15::OHLCV> tsla = python_to_ohlcv(tsla_df);
            std::vector<v15::OHLCV> spy = python_to_ohlcv(spy_df);
            std::vector<v15::OHLCV> vix = python_to_ohlcv(vix_df);

            // Convert native TF dict if provided
            v15::NativeTFData native_data;
            v15::NativeTFData* native_ptr = nullptr;
            if (!native_tf_dict.is_none()) {
                native_data = convert_native_tf_dict(native_tf_dict.cast<py::dict>());
                native_ptr = &native_data;
            }

            // Run full feature extraction pipeline
            auto features = v15::FeatureExtractor::extract_features_for_inference(
                tsla, spy, vix, timestamp_ms, source_bar_count, true, native_ptr
            );

            // Convert to Python dict
            py::dict result;
            for (const auto& [name, value] : features) {
                result[py::str(name)] = value;
            }
            return result;
        },
        py::arg("tsla_df"),
        py::arg("spy_df"),
        py::arg("vix_df"),
        py::arg("timestamp_ms"),
        py::arg("source_bar_count") = -1,
        py::arg("native_tf_data") = py::none(),
        R"pbdoc(
            Extract all features from raw OHLCV data for model inference.

            This runs the full C++ feature extraction pipeline:
            1. Uses native TF bars (from yfinance) when available, else resamples
            2. Detects channels at all 8 windows for TSLA and SPY
            3. Extracts ~15,350 features matching training data exactly

            Args:
                tsla_df: TSLA 5-min OHLCV DataFrame with DatetimeIndex
                spy_df: SPY 5-min OHLCV DataFrame aligned to TSLA
                vix_df: VIX 5-min OHLCV DataFrame aligned to TSLA
                timestamp_ms: Current timestamp in milliseconds
                source_bar_count: Number of 5min bars (-1 = use data size)
                native_tf_data: Optional dict of native TF bars from yfinance.
                    Format: {'tsla': {'daily': df, ...}, 'spy': {...}, 'vix': {...}}

            Returns:
                Dict of feature name -> value (matching training features)
        )pbdoc"
    );

    m.def("get_feature_names",
        []() -> py::list {
            auto names = v15::FeatureExtractor::get_all_feature_names();
            py::list result;
            for (const auto& name : names) {
                result.append(py::str(name));
            }
            return result;
        },
        "Get all feature names in consistent order"
    );

    m.def("get_feature_count",
        []() -> int {
            return v15::FeatureExtractor::get_total_feature_count();
        },
        "Get expected total feature count (15,350)"
    );
}
