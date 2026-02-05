#include "data_loader.hpp"
#include <iostream>
#include <iomanip>

using namespace x14;

int main() {
    try {
        // 1. Create data loader
        std::cout << "Loading market data..." << std::endl;
        DataLoader loader("../data", true);  // path, enable validation

        // 2. Load all data (TSLA, SPY, VIX aligned)
        MarketData data = loader.load();

        // 3. Print summary
        std::cout << "Loaded " << data.num_bars << " aligned 5-minute bars" << std::endl;
        std::cout << std::endl;

        // 4. Access data
        std::cout << "First bar:" << std::endl;
        const auto& tsla = data.tsla[0];
        const auto& spy = data.spy[0];
        const auto& vix = data.vix[0];

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  TSLA: $" << tsla.close << std::endl;
        std::cout << "  SPY:  $" << spy.close << std::endl;
        std::cout << "  VIX:  " << vix.close << std::endl;
        std::cout << std::endl;

        // 5. Process data
        std::cout << "Processing bars..." << std::endl;
        size_t count = 0;
        double tsla_total = 0;

        for (size_t i = 0; i < data.num_bars; ++i) {
            // All timestamps are aligned
            if (data.tsla[i].timestamp != data.spy[i].timestamp) {
                std::cerr << "ERROR: Timestamps not aligned!" << std::endl;
                return 1;
            }

            // Calculate average TSLA price
            tsla_total += data.tsla[i].close;
            count++;

            // Example: Find high volume bars
            if (data.tsla[i].volume > 100000) {
                // High volume bar detected
                // ... your logic here ...
            }
        }

        double tsla_avg = tsla_total / count;
        std::cout << "Average TSLA close: $" << tsla_avg << std::endl;

        return 0;

    } catch (const DataLoadError& e) {
        std::cerr << "Data load error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
