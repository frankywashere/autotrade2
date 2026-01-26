#pragma once

#include "sample.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <fstream>

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
 * FILE STRUCTURE:
 *
 * [HEADER]
 *   magic_bytes:    8 bytes  "V15SAMP\0"
 *   version:        4 bytes  uint32_t (format version, currently 1)
 *   num_samples:    8 bytes  uint64_t (total samples in file)
 *   num_features:   4 bytes  uint32_t (features per sample, for validation)
 *
 * [SAMPLE RECORDS] (repeated num_samples times)
 *   For each sample:
 *     timestamp:           8 bytes   int64_t
 *     channel_end_idx:     4 bytes   int32_t
 *     best_window:         4 bytes   int32_t
 *
 *     [FEATURES]
 *       feature_count:     4 bytes   uint32_t
 *       For each feature:
 *         key_length:      2 bytes   uint16_t
 *         key_data:        N bytes   UTF-8 string (no null terminator)
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
 *   Future versions can extend ChannelLabels or add new sections
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
constexpr uint32_t FORMAT_VERSION = 2;

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

    void write_sample_internal(const ChannelSample& sample);
};

} // namespace v15
