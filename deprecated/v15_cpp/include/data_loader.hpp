#pragma once

#include "types.hpp"
#include <string>
#include <vector>
#include <ctime>
#include <stdexcept>

namespace v15 {

// Market data for all assets
struct MarketData {
    std::vector<OHLCV> tsla;
    std::vector<OHLCV> spy;
    std::vector<OHLCV> vix;

    // Date range info
    std::time_t start_time;
    std::time_t end_time;
    size_t num_bars;

    MarketData() : start_time(0), end_time(0), num_bars(0) {}
};

// Data loader exception
class DataLoadError : public std::runtime_error {
public:
    explicit DataLoadError(const std::string& message)
        : std::runtime_error(message) {}
};

// Fast CSV data loader
class DataLoader {
public:
    // Constructor
    explicit DataLoader(const std::string& data_dir, bool validate = true);

    // Load all market data (TSLA, SPY, VIX)
    // Returns aligned data at 5-minute resolution
    MarketData load();

    // Validation options
    void set_validation(bool enabled) { validate_ = enabled; }
    bool get_validation() const { return validate_; }

private:
    std::string data_dir_;
    bool validate_;

    // Load and parse CSV files
    std::vector<OHLCV> load_1min_csv(const std::string& filename, const std::string& asset_name);
    std::vector<OHLCV> load_vix_csv(const std::string& filename);

    // Fast CSV parsing
    void parse_csv_line_1min(const char* line, OHLCV& out);
    void parse_csv_line_vix(const char* line, OHLCV& out);

    // Timestamp parsing
    std::time_t parse_timestamp_iso(const char* str);  // "YYYY-MM-DD HH:MM:SS"
    std::time_t parse_timestamp_mdy(const char* str);  // "MM/DD/YYYY"

    // Fast number parsing
    double fast_parse_double(const char*& ptr);
    long fast_parse_long(const char*& ptr);

    // Data processing
    std::vector<OHLCV> resample_to_5min(const std::vector<OHLCV>& data, const std::string& asset_name);
    void align_to_tsla(const std::vector<OHLCV>& tsla,
                       const std::vector<OHLCV>& spy,
                       const std::vector<OHLCV>& vix,
                       MarketData& out);

    // Validation
    void validate_ohlcv(const std::vector<OHLCV>& data,
                       const std::string& asset_name,
                       bool require_volume = true,
                       bool strict_ohlc = true);
    void validate_alignment(const MarketData& data);

    // Utility
    void skip_whitespace(const char*& ptr);
    void skip_to_char(const char*& ptr, char c);
    std::time_t normalize_to_date(std::time_t timestamp);
};

} // namespace v15
