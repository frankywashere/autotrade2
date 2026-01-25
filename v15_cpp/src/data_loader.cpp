#include "data_loader.hpp"
#include <fstream>
#include <sstream>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <set>
#include <cmath>
#include <sys/stat.h>

namespace v15 {

// Constants
constexpr size_t ESTIMATED_ROWS_1MIN = 2000000;  // Pre-allocate for ~2M rows
constexpr size_t ESTIMATED_ROWS_5MIN = 400000;   // Pre-allocate for ~400K 5min bars

DataLoader::DataLoader(const std::string& data_dir, bool validate)
    : data_dir_(data_dir), validate_(validate) {

    // Check directory exists
    struct stat info;
    if (stat(data_dir_.c_str(), &info) != 0) {
        throw DataLoadError("Data directory does not exist: " + data_dir_);
    }
    if (!(info.st_mode & S_IFDIR)) {
        throw DataLoadError("Path is not a directory: " + data_dir_);
    }
}

MarketData DataLoader::load() {
    MarketData result;

    // Load TSLA (1min -> resample to 5min)
    auto tsla_1min = load_1min_csv("TSLA_1min.csv", "TSLA");
    auto tsla_5min = resample_to_5min(tsla_1min, "TSLA");
    if (validate_) {
        validate_ohlcv(tsla_5min, "TSLA", true, true);
    }

    // Load SPY (1min -> resample to 5min)
    auto spy_1min = load_1min_csv("SPY_1min.csv", "SPY");
    auto spy_5min = resample_to_5min(spy_1min, "SPY");
    if (validate_) {
        validate_ohlcv(spy_5min, "SPY", true, true);
    }

    // Load VIX (daily data)
    auto vix = load_vix_csv("VIX_History.csv");
    if (validate_) {
        validate_ohlcv(vix, "VIX", false, false);  // VIX has no volume, relaxed OHLC checks
    }

    // Align all data to TSLA's index
    align_to_tsla(tsla_5min, spy_5min, vix, result);

    // Final validation
    if (validate_) {
        validate_alignment(result);
    }

    return result;
}

std::vector<OHLCV> DataLoader::load_1min_csv(const std::string& filename, const std::string& asset_name) {
    std::string filepath = data_dir_ + "/" + filename;

    // Open file
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw DataLoadError(asset_name + ": File not found: " + filepath);
    }

    // Reset to beginning of file
    file.seekg(0, std::ios::beg);

    // Pre-allocate vector
    std::vector<OHLCV> data;
    data.reserve(ESTIMATED_ROWS_1MIN);

    // Read header line
    std::string header;
    std::getline(file, header);
    if (header.empty()) {
        throw DataLoadError(asset_name + ": Empty CSV file");
    }

    // Parse data lines
    std::string line;
    line.reserve(128);  // Reserve space for line buffer
    size_t line_num = 1;

    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || line[0] == '\r' || line[0] == '\n') {
            continue;
        }

        try {
            OHLCV bar;
            parse_csv_line_1min(line.c_str(), bar);
            data.push_back(bar);
        } catch (const std::exception& e) {
            throw DataLoadError(asset_name + ": Parse error at line " +
                              std::to_string(line_num) + ": " + e.what());
        }
    }

    if (data.empty()) {
        throw DataLoadError(asset_name + ": No data loaded from " + filepath);
    }

    // Sort by timestamp
    std::sort(data.begin(), data.end(),
              [](const OHLCV& a, const OHLCV& b) { return a.timestamp < b.timestamp; });

    // Check for duplicates
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i].timestamp == data[i-1].timestamp) {
            char buf[32];
            strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&data[i].timestamp));
            throw DataLoadError(asset_name + ": Duplicate timestamp found: " + std::string(buf));
        }
    }

    return data;
}

std::vector<OHLCV> DataLoader::load_vix_csv(const std::string& filename) {
    std::string filepath = data_dir_ + "/" + filename;

    // Open file
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        throw DataLoadError("VIX: File not found: " + filepath);
    }

    // Pre-allocate
    std::vector<OHLCV> data;
    data.reserve(10000);  // VIX has ~9K rows

    // Read header
    std::string header;
    std::getline(file, header);
    if (header.empty()) {
        throw DataLoadError("VIX: Empty CSV file");
    }

    // Parse data
    std::string line;
    line.reserve(128);
    size_t line_num = 1;

    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || line[0] == '\r' || line[0] == '\n') {
            continue;
        }

        try {
            OHLCV bar;
            parse_csv_line_vix(line.c_str(), bar);
            bar.volume = 0;  // VIX has no volume
            data.push_back(bar);
        } catch (const std::exception& e) {
            throw DataLoadError("VIX: Parse error at line " +
                              std::to_string(line_num) + ": " + e.what());
        }
    }

    if (data.empty()) {
        throw DataLoadError("VIX: No data loaded from " + filepath);
    }

    // Sort by timestamp
    std::sort(data.begin(), data.end(),
              [](const OHLCV& a, const OHLCV& b) { return a.timestamp < b.timestamp; });

    return data;
}

void DataLoader::parse_csv_line_1min(const char* line, OHLCV& out) {
    // Format: timestamp,open,high,low,close,volume
    // Example: 2015-01-02 11:40:00,223.29,223.29,223.29,223.29,175

    const char* ptr = line;

    // Parse timestamp
    out.timestamp = parse_timestamp_iso(ptr);
    skip_to_char(ptr, ',');
    ptr++;

    // Parse OHLCV values
    out.open = fast_parse_double(ptr);
    skip_to_char(ptr, ',');
    ptr++;

    out.high = fast_parse_double(ptr);
    skip_to_char(ptr, ',');
    ptr++;

    out.low = fast_parse_double(ptr);
    skip_to_char(ptr, ',');
    ptr++;

    out.close = fast_parse_double(ptr);
    skip_to_char(ptr, ',');
    ptr++;

    out.volume = fast_parse_double(ptr);
}

void DataLoader::parse_csv_line_vix(const char* line, OHLCV& out) {
    // Format: DATE,OPEN,HIGH,LOW,CLOSE
    // Example: 01/02/1990,17.240000,17.240000,17.240000,17.240000

    const char* ptr = line;

    // Parse date
    out.timestamp = parse_timestamp_mdy(ptr);
    skip_to_char(ptr, ',');
    ptr++;

    // Parse OHLC values
    out.open = fast_parse_double(ptr);
    skip_to_char(ptr, ',');
    ptr++;

    out.high = fast_parse_double(ptr);
    skip_to_char(ptr, ',');
    ptr++;

    out.low = fast_parse_double(ptr);
    skip_to_char(ptr, ',');
    ptr++;

    out.close = fast_parse_double(ptr);
    out.volume = 0;  // VIX has no volume
}

std::time_t DataLoader::parse_timestamp_iso(const char* str) {
    // Parse: YYYY-MM-DD HH:MM:SS
    // Fast manual parsing avoiding strptime

    struct tm tm_time = {};
    const char* ptr = str;

    // Year
    tm_time.tm_year = fast_parse_long(ptr) - 1900;
    ptr++;  // skip '-'

    // Month
    tm_time.tm_mon = fast_parse_long(ptr) - 1;
    ptr++;  // skip '-'

    // Day
    tm_time.tm_mday = fast_parse_long(ptr);
    ptr++;  // skip space

    // Hour
    tm_time.tm_hour = fast_parse_long(ptr);
    ptr++;  // skip ':'

    // Minute
    tm_time.tm_min = fast_parse_long(ptr);
    ptr++;  // skip ':'

    // Second
    tm_time.tm_sec = fast_parse_long(ptr);

    tm_time.tm_isdst = -1;  // Let mktime determine DST

    std::time_t result = mktime(&tm_time);
    if (result == -1) {
        throw DataLoadError("Failed to parse timestamp: " + std::string(str));
    }

    return result;
}

std::time_t DataLoader::parse_timestamp_mdy(const char* str) {
    // Parse: MM/DD/YYYY

    struct tm tm_time = {};
    const char* ptr = str;

    // Month
    tm_time.tm_mon = fast_parse_long(ptr) - 1;
    ptr++;  // skip '/'

    // Day
    tm_time.tm_mday = fast_parse_long(ptr);
    ptr++;  // skip '/'

    // Year
    tm_time.tm_year = fast_parse_long(ptr) - 1900;

    tm_time.tm_hour = 0;
    tm_time.tm_min = 0;
    tm_time.tm_sec = 0;
    tm_time.tm_isdst = -1;

    std::time_t result = mktime(&tm_time);
    if (result == -1) {
        throw DataLoadError("Failed to parse date: " + std::string(str));
    }

    return result;
}

double DataLoader::fast_parse_double(const char*& ptr) {
    // Fast double parsing using strtod (faster than stringstream)
    char* end;
    double value = strtod(ptr, &end);

    if (ptr == end) {
        throw DataLoadError("Failed to parse number at: " + std::string(ptr, 10));
    }

    return value;
}

long DataLoader::fast_parse_long(const char*& ptr) {
    // Fast integer parsing
    long value = 0;
    bool negative = false;

    if (*ptr == '-') {
        negative = true;
        ptr++;
    }

    while (isdigit(*ptr)) {
        value = value * 10 + (*ptr - '0');
        ptr++;
    }

    return negative ? -value : value;
}

void DataLoader::skip_whitespace(const char*& ptr) {
    while (*ptr && isspace(*ptr)) {
        ptr++;
    }
}

void DataLoader::skip_to_char(const char*& ptr, char c) {
    while (*ptr && *ptr != c) {
        ptr++;
    }
}

std::vector<OHLCV> DataLoader::resample_to_5min(const std::vector<OHLCV>& data,
                                                  const std::string& asset_name) {
    if (data.empty()) {
        throw DataLoadError(asset_name + ": Cannot resample empty data");
    }

    std::vector<OHLCV> result;
    result.reserve(ESTIMATED_ROWS_5MIN);

    // Group by 5-minute intervals
    std::time_t current_5min = (data[0].timestamp / 300) * 300;  // Round down to 5min

    OHLCV bar_5min;
    bar_5min.timestamp = current_5min;
    bar_5min.open = data[0].open;
    bar_5min.high = data[0].high;
    bar_5min.low = data[0].low;
    bar_5min.close = data[0].close;
    bar_5min.volume = data[0].volume;

    for (size_t i = 1; i < data.size(); ++i) {
        std::time_t ts_5min = (data[i].timestamp / 300) * 300;

        if (ts_5min == current_5min) {
            // Same 5-minute bar - aggregate
            bar_5min.high = std::max(bar_5min.high, data[i].high);
            bar_5min.low = std::min(bar_5min.low, data[i].low);
            bar_5min.close = data[i].close;  // Last close
            bar_5min.volume += data[i].volume;
        } else {
            // New 5-minute bar
            result.push_back(bar_5min);

            // Start new bar
            current_5min = ts_5min;
            bar_5min.timestamp = current_5min;
            bar_5min.open = data[i].open;
            bar_5min.high = data[i].high;
            bar_5min.low = data[i].low;
            bar_5min.close = data[i].close;
            bar_5min.volume = data[i].volume;
        }
    }

    // Add last bar
    result.push_back(bar_5min);

    if (result.empty()) {
        throw DataLoadError(asset_name + ": Resampling resulted in empty data");
    }

    return result;
}

std::time_t DataLoader::normalize_to_date(std::time_t timestamp) {
    // Get date part only (midnight UTC)
    struct tm* tm_time = localtime(&timestamp);
    tm_time->tm_hour = 0;
    tm_time->tm_min = 0;
    tm_time->tm_sec = 0;
    return mktime(tm_time);
}

void DataLoader::align_to_tsla(const std::vector<OHLCV>& tsla,
                                const std::vector<OHLCV>& spy,
                                const std::vector<OHLCV>& vix,
                                MarketData& out) {
    if (tsla.empty() || spy.empty() || vix.empty()) {
        throw DataLoadError("Cannot align empty datasets");
    }

    // Find common date range
    std::set<std::time_t> tsla_dates, spy_dates, vix_dates;

    for (const auto& bar : tsla) {
        tsla_dates.insert(normalize_to_date(bar.timestamp));
    }
    for (const auto& bar : spy) {
        spy_dates.insert(normalize_to_date(bar.timestamp));
    }
    for (const auto& bar : vix) {
        vix_dates.insert(normalize_to_date(bar.timestamp));
    }

    // Find intersection
    std::vector<std::time_t> common_dates;
    std::set_intersection(tsla_dates.begin(), tsla_dates.end(),
                         spy_dates.begin(), spy_dates.end(),
                         std::back_inserter(common_dates));

    std::vector<std::time_t> final_common;
    std::set_intersection(common_dates.begin(), common_dates.end(),
                         vix_dates.begin(), vix_dates.end(),
                         std::back_inserter(final_common));

    if (final_common.empty()) {
        char buf1[32], buf2[32];
        strftime(buf1, sizeof(buf1), "%Y-%m-%d", localtime(&(*tsla_dates.begin())));
        strftime(buf2, sizeof(buf2), "%Y-%m-%d", localtime(&(*tsla_dates.rbegin())));
        throw DataLoadError("No overlapping dates found between TSLA, SPY, and VIX. "
                          "TSLA range: " + std::string(buf1) + " to " + std::string(buf2));
    }

    std::time_t start_date = *final_common.begin();
    std::time_t end_date = *final_common.rbegin();

    // Build VIX lookup map (date -> OHLCV) for forward-fill
    std::map<std::time_t, OHLCV> vix_map;
    for (const auto& bar : vix) {
        std::time_t date = normalize_to_date(bar.timestamp);
        if (date >= start_date && date <= end_date) {
            vix_map[date] = bar;
        }
    }

    // Build SPY lookup for forward-fill (sorted vector for binary search)
    std::vector<OHLCV> spy_filtered;
    for (const auto& bar : spy) {
        std::time_t date = normalize_to_date(bar.timestamp);
        if (date >= start_date && date <= end_date) {
            spy_filtered.push_back(bar);
        }
    }

    // Align to TSLA's timestamps using forward-fill for both SPY and VIX
    out.tsla.reserve(tsla.size());
    out.spy.reserve(tsla.size());
    out.vix.reserve(tsla.size());

    size_t spy_idx = 0;
    OHLCV last_vix = vix_map.begin()->second;  // For forward-fill
    bool spy_initialized = false;
    bool vix_initialized = false;

    for (const auto& tsla_bar : tsla) {
        std::time_t date = normalize_to_date(tsla_bar.timestamp);

        // Skip if outside common date range
        if (date < start_date || date > end_date) {
            continue;
        }

        // Forward-fill SPY: find latest SPY bar <= TSLA timestamp
        while (spy_idx < spy_filtered.size() && spy_filtered[spy_idx].timestamp <= tsla_bar.timestamp) {
            spy_idx++;
            spy_initialized = true;
        }
        if (!spy_initialized) {
            // Skip until we have SPY data
            continue;
        }
        OHLCV spy_aligned = spy_filtered[spy_idx - 1];
        spy_aligned.timestamp = tsla_bar.timestamp;  // Set to TSLA timestamp

        // Forward-fill VIX from daily data
        auto vix_it = vix_map.find(date);
        if (vix_it != vix_map.end()) {
            last_vix = vix_it->second;
            vix_initialized = true;
        }
        if (!vix_initialized) {
            // Skip until we have VIX data
            continue;
        }

        // Set VIX timestamp to match TSLA
        OHLCV vix_aligned = last_vix;
        vix_aligned.timestamp = tsla_bar.timestamp;

        // Add aligned bars (all with TSLA's timestamp)
        out.tsla.push_back(tsla_bar);
        out.spy.push_back(spy_aligned);
        out.vix.push_back(vix_aligned);
    }

    if (out.tsla.empty()) {
        throw DataLoadError("No valid rows after alignment");
    }

    out.num_bars = out.tsla.size();
    out.start_time = out.tsla.front().timestamp;
    out.end_time = out.tsla.back().timestamp;
}

void DataLoader::validate_ohlcv(const std::vector<OHLCV>& data,
                                 const std::string& asset_name,
                                 bool require_volume,
                                 bool strict_ohlc) {
    if (data.empty()) {
        throw DataLoadError(asset_name + ": Empty data");
    }

    for (size_t i = 0; i < data.size(); ++i) {
        const auto& bar = data[i];

        // Check high >= low
        if (bar.high < bar.low) {
            char buf[32];
            strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&bar.timestamp));
            throw DataLoadError(asset_name + ": high < low at " + std::string(buf) +
                              " (high=" + std::to_string(bar.high) +
                              ", low=" + std::to_string(bar.low) + ")");
        }

        // Strict OHLC checks
        if (strict_ohlc) {
            if (bar.high < bar.open || bar.high < bar.close) {
                char buf[32];
                strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&bar.timestamp));
                throw DataLoadError(asset_name + ": high < open/close at " + std::string(buf));
            }
            if (bar.low > bar.open || bar.low > bar.close) {
                char buf[32];
                strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&bar.timestamp));
                throw DataLoadError(asset_name + ": low > open/close at " + std::string(buf));
            }
        }

        // Check for non-positive prices
        if (bar.open <= 0 || bar.high <= 0 || bar.low <= 0 || bar.close <= 0) {
            char buf[32];
            strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&bar.timestamp));
            throw DataLoadError(asset_name + ": Non-positive price at " + std::string(buf));
        }

        // Check for infinite values
        if (std::isinf(bar.open) || std::isinf(bar.high) ||
            std::isinf(bar.low) || std::isinf(bar.close)) {
            char buf[32];
            strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&bar.timestamp));
            throw DataLoadError(asset_name + ": Infinite value at " + std::string(buf));
        }

        // Check volume if required
        if (require_volume && bar.volume < 0) {
            char buf[32];
            strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&bar.timestamp));
            throw DataLoadError(asset_name + ": Negative volume at " + std::string(buf));
        }
    }
}

void DataLoader::validate_alignment(const MarketData& data) {
    // Check all vectors have same length
    if (data.tsla.size() != data.spy.size()) {
        throw DataLoadError("Length mismatch: TSLA=" + std::to_string(data.tsla.size()) +
                          ", SPY=" + std::to_string(data.spy.size()));
    }
    if (data.tsla.size() != data.vix.size()) {
        throw DataLoadError("Length mismatch: TSLA=" + std::to_string(data.tsla.size()) +
                          ", VIX=" + std::to_string(data.vix.size()));
    }

    // Check all timestamps match exactly (all aligned to TSLA's timestamps)
    for (size_t i = 0; i < data.tsla.size(); ++i) {
        if (data.tsla[i].timestamp != data.spy[i].timestamp) {
            char buf1[32], buf2[32];
            strftime(buf1, sizeof(buf1), "%Y-%m-%d %H:%M:%S", localtime(&data.tsla[i].timestamp));
            strftime(buf2, sizeof(buf2), "%Y-%m-%d %H:%M:%S", localtime(&data.spy[i].timestamp));
            throw DataLoadError("Timestamp mismatch between TSLA and SPY at index " + std::to_string(i) +
                              ": TSLA=" + std::string(buf1) + ", SPY=" + std::string(buf2));
        }

        if (data.tsla[i].timestamp != data.vix[i].timestamp) {
            char buf1[32], buf2[32];
            strftime(buf1, sizeof(buf1), "%Y-%m-%d %H:%M:%S", localtime(&data.tsla[i].timestamp));
            strftime(buf2, sizeof(buf2), "%Y-%m-%d %H:%M:%S", localtime(&data.vix[i].timestamp));
            throw DataLoadError("Timestamp mismatch between TSLA and VIX at index " + std::to_string(i) +
                              ": TSLA=" + std::string(buf1) + ", VIX=" + std::string(buf2));
        }
    }

    // Check num_bars matches
    if (data.num_bars != data.tsla.size()) {
        throw DataLoadError("num_bars mismatch: " + std::to_string(data.num_bars) +
                          " vs " + std::to_string(data.tsla.size()));
    }
}

} // namespace v15
