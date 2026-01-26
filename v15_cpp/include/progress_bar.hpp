#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <chrono>
#include <sstream>
#include <mutex>

namespace v15 {

/**
 * Lightweight terminal progress bar with ETA estimation.
 *
 * Features:
 * - In-place updates using carriage return (\r)
 * - Percentage complete and items processed
 * - ETA calculation using elapsed time
 * - Thread-safe updates
 * - No external dependencies
 */
class ProgressBar {
public:
    explicit ProgressBar(
        size_t total,
        const std::string& prefix = "",
        int width = 30,
        std::ostream& out = std::cerr
    )
        : total_(total)
        , current_(0)
        , prefix_(prefix)
        , width_(width)
        , out_(out)
        , started_(false)
        , finished_(false)
    {}

    void update(size_t current) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (finished_) return;

        if (!started_) {
            start_time_ = std::chrono::steady_clock::now();
            started_ = true;
        }

        current_ = current;
        render();
    }

    void tick() {
        update(current_ + 1);
    }

    void finish() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (finished_) return;

        current_ = total_;
        render();
        out_ << std::endl;
        finished_ = true;
    }

    size_t current() const { return current_; }
    size_t total() const { return total_; }

private:
    size_t total_;
    size_t current_;
    std::string prefix_;
    int width_;
    std::ostream& out_;
    std::chrono::steady_clock::time_point start_time_;
    bool started_;
    bool finished_;
    mutable std::mutex mutex_;

    void render() {
        double progress = (total_ > 0) ? static_cast<double>(current_) / total_ : 0.0;
        int filled = static_cast<int>(progress * width_);
        int percent = static_cast<int>(progress * 100);

        out_ << '\r';

        if (!prefix_.empty()) {
            out_ << prefix_ << " ";
        }

        // Progress bar [=====>    ]
        out_ << '[';
        for (int i = 0; i < width_; ++i) {
            if (i < filled) {
                out_ << '=';
            } else if (i == filled && current_ < total_) {
                out_ << '>';
            } else {
                out_ << ' ';
            }
        }
        out_ << "] ";

        // Percentage and count
        out_ << std::setw(3) << percent << "% " << current_ << "/" << total_;

        // ETA
        if (current_ > 0 && current_ < total_ && started_) {
            auto elapsed = std::chrono::steady_clock::now() - start_time_;
            double elapsed_sec = std::chrono::duration<double>(elapsed).count();
            double rate = current_ / elapsed_sec;
            double remaining_sec = (total_ - current_) / rate;

            out_ << " ETA: " << format_duration(remaining_sec);
            out_ << " (" << std::fixed << std::setprecision(1) << rate << "/s)";
        } else if (current_ >= total_ && started_) {
            auto elapsed = std::chrono::steady_clock::now() - start_time_;
            double elapsed_sec = std::chrono::duration<double>(elapsed).count();
            out_ << " Done in " << format_duration(elapsed_sec);
        }

        // Clear rest of line
        out_ << "        " << std::flush;
    }

    static std::string format_duration(double seconds) {
        std::ostringstream oss;

        if (seconds < 60) {
            oss << std::fixed << std::setprecision(1) << seconds << "s";
        } else if (seconds < 3600) {
            int mins = static_cast<int>(seconds / 60);
            int secs = static_cast<int>(seconds) % 60;
            oss << mins << "m" << secs << "s";
        } else {
            int hours = static_cast<int>(seconds / 3600);
            int mins = (static_cast<int>(seconds) % 3600) / 60;
            oss << hours << "h" << mins << "m";
        }

        return oss.str();
    }
};

} // namespace v15
