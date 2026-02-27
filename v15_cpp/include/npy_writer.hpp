#pragma once

/**
 * NumPy .npy File Writer
 *
 * Writes data to NumPy binary format (.npy) compatible with numpy.load()
 */

#include <string>
#include <cstdint>
#include <vector>

namespace v15 {

class NpyWriter {
public:
    // Write a 2D float32 array
    static void write_float32_2d(const std::string& path, const float* data, size_t rows, size_t cols);

    // Write a 1D float32 array
    static void write_float32_1d(const std::string& path, const float* data, size_t size);
    static void write_float32_1d(const std::string& path, const std::vector<float>& data);

    // Write a 2D float32 array (convenience overload for vector)
    static void write_float32_2d(const std::string& path, const std::vector<float>& data, size_t rows, size_t cols);

    // Write a 1D int64 array
    static void write_int64_1d(const std::string& path, const int64_t* data, size_t size);
    static void write_int64_1d(const std::string& path, const std::vector<int64_t>& data);

    // Write a 2D int64 array
    static void write_int64_2d(const std::string& path, const int64_t* data, size_t rows, size_t cols);
    static void write_int64_2d(const std::string& path, const std::vector<int64_t>& data, size_t rows, size_t cols);

    // Write a 1D uint8 array (for bool - numpy uses uint8 for bool)
    static void write_bool_1d(const std::string& path, const uint8_t* data, size_t size);
    static void write_bool_1d(const std::string& path, const std::vector<uint8_t>& data);

private:
    static void write_array(
        const std::string& path,
        const void* data,
        size_t data_size_bytes,
        const std::string& dtype,
        const std::vector<size_t>& shape
    );

    static std::string format_shape(const std::vector<size_t>& shape);
    static std::string create_header(const std::string& dtype, const std::string& shape);
};

} // namespace v15
