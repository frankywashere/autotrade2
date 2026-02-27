#pragma once

#include "sample.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <algorithm>

namespace v15 {

// =============================================================================
// BINARY SERIALIZATION FORMAT SPECIFICATION
// =============================================================================

/**
 * Binary format for efficient ChannelSample storage.
 *
 * DESIGN GOALS:
 *   - Fast write/read performance
 *   - Minimal overhead (no JSON/text parsing)
 *   - Forward compatible with Python loading
 *   - Versioned format for future evolution
 *
 * FILE STRUCTURE (v3 with feature name table optimization):
 *
 * [HEADER]
 *   magic_bytes:    8 bytes  "V15SAMP\0"
 *   version:        4 bytes  uint32_t (format version, currently 3)
 *   num_samples:    8 bytes  uint64_t (total samples in file)
 *   num_features:   4 bytes  uint32_t (features per sample, for validation)
 *
 * [FEATURE NAME TABLE] (v3 only - names stored once instead of per-sample)
 *   feature_table_count: 4 bytes  uint32_t (number of feature names)
 *   For each name:
 *     name_length:  2 bytes  uint16_t
 *     name_data:    N bytes  UTF-8 string (no null terminator)
 *
 * [SAMPLE RECORDS] (repeated num_samples times)
 *   For each sample:
 *     timestamp:           8 bytes   int64_t
 *     channel_end_idx:     4 bytes   int32_t
 *     best_window:         4 bytes   int32_t
 *
 *     [FEATURES] (v3: index-based)
 *       feature_count:     4 bytes   uint32_t
 *       For each feature:
 *         feature_index:   2 bytes   uint16_t (index into feature name table)
 *         value:           8 bytes   double
 *
 *     [LABELS_PER_WINDOW]
 *       window_count:      4 bytes   uint32_t (number of windows)
 *       For each window:
 *         window_size:     4 bytes   int32_t
 *         tf_count:        4 bytes   uint32_t (number of timeframes)
 *         For each timeframe:
 *           tf_key_length: 2 bytes   uint16_t
 *           tf_key_data:   N bytes   UTF-8 string
 *           [ChannelLabels data - see serialize_channel_labels()]
 *
 *     [BAR_METADATA]
 *       metadata_tf_count: 4 bytes   uint32_t (number of timeframes with metadata)
 *       For each timeframe:
 *         tf_key_length:   2 bytes   uint16_t
 *         tf_key_data:     N bytes   UTF-8 string
 *         meta_count:      4 bytes   uint32_t (number of metadata entries)
 *         For each metadata entry:
 *           meta_key_length: 2 bytes uint16_t
 *           meta_key_data:   N bytes UTF-8 string
 *           meta_value:      8 bytes double
 *
 * VERSIONING:
 *   Version 1: Initial format with full ChannelLabels support
 *   Version 2: Added SPY labels and source channel parameters
 *   Version 3: Feature name table optimization (~70% file size reduction)
 *
 * ENDIANNESS:
 *   Little-endian for all multi-byte integers and doubles
 *   Compatible with x86/x64 and most modern ARM systems
 *
 * PYTHON COMPATIBILITY:
 *   Python can read using struct.unpack() with '<' prefix
 *   Example: struct.unpack('<Q', bytes) for uint64_t
 */

// Format constants
constexpr uint8_t MAGIC_BYTES[8] = {'V', '1', '5', 'S', 'A', 'M', 'P', '\0'};
constexpr uint32_t FORMAT_VERSION = 3;
constexpr uint32_t FORMAT_VERSION_V2 = 2;  // For backward compatibility

// =============================================================================
// FEATURE NAME TABLE
// =============================================================================

/**
 * Feature name table for v3 format optimization.
 *
 * Stores feature names once in the file header, then references them by
 * uint16_t index in each sample. This reduces file size by ~70% for
 * datasets with many samples.
 *
 * DESIGN:
 *   - Names stored alphabetically for determinism
 *   - uint16_t indices support up to 65,535 features (current: ~14,840)
 *   - Bidirectional lookup: name -> index and index -> name
 */
class FeatureNameTable {
public:
    /**
     * Build the table from a sample's features.
     * Names are sorted alphabetically for deterministic ordering.
     */
    void build_from_sample(const ChannelSample& sample) {
        names_.clear();
        name_to_index_.clear();

        // Collect all feature names
        for (const auto& pair : sample.tf_features) {
            names_.push_back(pair.first);
        }

        // Sort alphabetically for determinism
        std::sort(names_.begin(), names_.end());

        // Build reverse lookup
        for (size_t i = 0; i < names_.size(); ++i) {
            name_to_index_[names_[i]] = static_cast<uint16_t>(i);
        }
    }

    /**
     * Get index for a feature name.
     * @throws std::runtime_error if name not found
     */
    uint16_t get_index(const std::string& name) const {
        auto it = name_to_index_.find(name);
        if (it == name_to_index_.end()) {
            throw std::runtime_error("Feature name not in table: " + name);
        }
        return it->second;
    }

    /**
     * Get name for a feature index.
     * @throws std::out_of_range if index invalid
     */
    const std::string& get_name(uint16_t index) const {
        if (index >= names_.size()) {
            throw std::out_of_range("Feature index out of range: " + std::to_string(index));
        }
        return names_[index];
    }

    /**
     * Check if table contains a feature name.
     */
    bool contains(const std::string& name) const {
        return name_to_index_.find(name) != name_to_index_.end();
    }

    /**
     * Get total number of feature names.
     */
    size_t size() const { return names_.size(); }

    /**
     * Check if table is empty.
     */
    bool empty() const { return names_.empty(); }

    /**
     * Get all names (in sorted order).
     */
    const std::vector<std::string>& names() const { return names_; }

    /**
     * Add a name to the table (used during deserialization).
     */
    void add_name(const std::string& name) {
        uint16_t index = static_cast<uint16_t>(names_.size());
        names_.push_back(name);
        name_to_index_[name] = index;
    }

    /**
     * Clear the table.
     */
    void clear() {
        names_.clear();
        name_to_index_.clear();
    }

private:
    std::vector<std::string> names_;                        // Ordered list of names
    std::unordered_map<std::string, uint16_t> name_to_index_;  // Reverse lookup
};

// =============================================================================
// PUBLIC API
// =============================================================================

/**
 * Save ChannelSample vector to binary file.
 *
 * @param samples Vector of samples to save
 * @param filename Output file path
 * @throws std::runtime_error on I/O error or invalid data
 */
void save_samples(const std::vector<ChannelSample>& samples, const std::string& filename);

/**
 * Load ChannelSample vector from binary file.
 *
 * @param filename Input file path
 * @return Vector of loaded samples
 * @throws std::runtime_error on I/O error, format error, or version mismatch
 */
std::vector<ChannelSample> load_samples(const std::string& filename);

/**
 * Validate file format without full deserialization.
 *
 * @param filename File to validate
 * @return true if file format is valid and version is supported
 */
bool validate_sample_file(const std::string& filename);

/**
 * Get file metadata without loading samples.
 *
 * @param filename File to inspect
 * @param version Output: format version
 * @param num_samples Output: number of samples in file
 * @param num_features Output: features per sample
 * @return true if metadata was successfully read
 */
bool get_file_metadata(const std::string& filename,
                       uint32_t& version,
                       uint64_t& num_samples,
                       uint32_t& num_features);

// =============================================================================
// STREAMING SAMPLE WRITER
// =============================================================================

/**
 * Streaming writer for incremental sample output.
 *
 * DESIGN GOALS:
 *   - Write samples to disk immediately as they're generated
 *   - Never hold all samples in memory at once
 *   - Enable processing of datasets that would otherwise cause OOM
 *   - Support flush intervals for periodic disk writes
 *
 * USAGE:
 *   StreamingSampleWriter writer("output.bin");
 *   writer.open();
 *   for (each sample) {
 *       writer.write(sample);
 *   }
 *   writer.close();  // Updates header with final count
 *
 * FILE FORMAT:
 *   Same as save_samples(), but header is written with placeholder count
 *   at open() time, then updated with actual count at close() time.
 */
class StreamingSampleWriter {
public:
    /**
     * Constructor
     * @param filename Output file path
     * @param flush_interval Samples between disk flushes (0 = flush every write)
     */
    explicit StreamingSampleWriter(const std::string& filename, size_t flush_interval = 100);

    /**
     * Destructor - closes file if still open
     */
    ~StreamingSampleWriter();

    // Disable copy
    StreamingSampleWriter(const StreamingSampleWriter&) = delete;
    StreamingSampleWriter& operator=(const StreamingSampleWriter&) = delete;

    // Enable move
    StreamingSampleWriter(StreamingSampleWriter&& other) noexcept;
    StreamingSampleWriter& operator=(StreamingSampleWriter&& other) noexcept;

    /**
     * Open file and write header with placeholder count.
     * @throws std::runtime_error on I/O error
     */
    void open();

    /**
     * Write a single sample to the file.
     * @param sample Sample to write
     * @throws std::runtime_error on I/O error
     */
    void write(const ChannelSample& sample);

    /**
     * Write multiple samples to the file.
     * @param samples Vector of samples to write
     * @throws std::runtime_error on I/O error
     */
    void write_batch(const std::vector<ChannelSample>& samples);

    /**
     * Flush buffered data to disk.
     */
    void flush();

    /**
     * Close file and update header with final sample count.
     * @throws std::runtime_error on I/O error
     */
    void close();

    /**
     * Check if file is open
     */
    bool is_open() const { return is_open_; }

    /**
     * Get number of samples written so far
     */
    uint64_t samples_written() const { return samples_written_; }

    /**
     * Get total features written (for average calculation)
     */
    uint64_t total_features() const { return total_features_; }

private:
    std::string filename_;
    size_t flush_interval_;
    std::ofstream ofs_;
    bool is_open_;
    uint64_t samples_written_;
    uint64_t total_features_;
    size_t samples_since_flush_;
    std::streampos count_position_;  // Position of sample count in header
    std::streampos features_position_;  // Position of avg features in header

    // v3 feature name table support
    FeatureNameTable feature_table_;
    bool feature_table_written_;

    void write_sample_internal(const ChannelSample& sample);
};

} // namespace v15
