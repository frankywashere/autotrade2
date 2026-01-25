#include "data_loader.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>

using namespace x14;

int main(int argc, char** argv) {
    std::string data_dir = (argc > 1) ? argv[1] : "../data";
    int num_runs = (argc > 2) ? std::atoi(argv[2]) : 5;

    std::cout << "=== Data Loader Benchmark ===" << std::endl;
    std::cout << "Data directory: " << data_dir << std::endl;
    std::cout << "Number of runs: " << num_runs << std::endl;
    std::cout << std::endl;

    std::vector<double> load_times;
    size_t total_bars = 0;

    for (int i = 0; i < num_runs; ++i) {
        std::cout << "Run " << (i+1) << "/" << num_runs << "... " << std::flush;

        auto start = std::chrono::high_resolution_clock::now();

        try {
            DataLoader loader(data_dir, i == 0);  // Only validate on first run
            MarketData data = loader.load();
            total_bars = data.num_bars;

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            load_times.push_back(duration.count());
            std::cout << duration.count() << " ms" << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }

    // Calculate statistics
    double sum = 0;
    double min_time = load_times[0];
    double max_time = load_times[0];

    for (double t : load_times) {
        sum += t;
        min_time = std::min(min_time, t);
        max_time = std::max(max_time, t);
    }

    double avg_time = sum / num_runs;

    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total bars loaded: " << total_bars << std::endl;
    std::cout << "Average time: " << avg_time << " ms" << std::endl;
    std::cout << "Min time: " << min_time << " ms" << std::endl;
    std::cout << "Max time: " << max_time << " ms" << std::endl;
    std::cout << "Throughput: " << (total_bars / (avg_time / 1000.0)) << " bars/sec" << std::endl;

    return 0;
}
