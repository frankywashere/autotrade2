/**
 * NumPy .npy File Writer Implementation
 */

#include "npy_writer.hpp"
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <sstream>

namespace v15 {

std::string NpyWriter::format_shape(const std::vector<size_t>& shape) {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    if (shape.size() == 1) oss << ",";
    oss << ")";
    return oss.str();
}

std::string NpyWriter::create_header(const std::string& dtype, const std::string& shape) {
    std::ostringstream oss;
    oss << "{'descr': '" << dtype << "', 'fortran_order': False, 'shape': " << shape << ", }";
    return oss.str();
}

void NpyWriter::write_array(
    const std::string& path,
    const void* data,
    size_t data_size_bytes,
    const std::string& dtype,
    const std::vector<size_t>& shape)
{
    std::string shape_str = format_shape(shape);
    std::string header_content = create_header(dtype, shape_str);

    // Pad to 64-byte alignment (magic=6 + version=2 + header_len=2 = 10 bytes prefix)
    size_t total_header = 10 + header_content.size();
    size_t padding = (64 - (total_header % 64)) % 64;
    if (padding == 0) padding = 64; // Always have at least newline
    header_content.append(padding - 1, ' ');
    header_content += '\n';

    uint16_t header_len = static_cast<uint16_t>(header_content.size());

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Failed to open: " + path);

    // Magic + version
    const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    ofs.write(magic, 6);
    uint8_t ver[2] = {1, 0};
    ofs.write(reinterpret_cast<char*>(ver), 2);

    // Header length + header
    ofs.write(reinterpret_cast<char*>(&header_len), 2);
    ofs.write(header_content.data(), header_content.size());

    // Data
    if (data_size_bytes > 0) {
        ofs.write(reinterpret_cast<const char*>(data), data_size_bytes);
    }

    ofs.close();
}

void NpyWriter::write_float32_2d(const std::string& path, const float* data, size_t rows, size_t cols) {
    write_array(path, data, rows * cols * sizeof(float), "<f4", {rows, cols});
}

void NpyWriter::write_float32_1d(const std::string& path, const float* data, size_t size) {
    write_array(path, data, size * sizeof(float), "<f4", {size});
}

void NpyWriter::write_float32_2d(const std::string& path, const std::vector<float>& data, size_t rows, size_t cols) {
    write_float32_2d(path, data.data(), rows, cols);
}

void NpyWriter::write_float32_1d(const std::string& path, const std::vector<float>& data) {
    write_float32_1d(path, data.data(), data.size());
}

void NpyWriter::write_int64_1d(const std::string& path, const int64_t* data, size_t size) {
    write_array(path, data, size * sizeof(int64_t), "<i8", {size});
}

void NpyWriter::write_int64_1d(const std::string& path, const std::vector<int64_t>& data) {
    write_int64_1d(path, data.data(), data.size());
}

void NpyWriter::write_int64_2d(const std::string& path, const int64_t* data, size_t rows, size_t cols) {
    write_array(path, data, rows * cols * sizeof(int64_t), "<i8", {rows, cols});
}

void NpyWriter::write_int64_2d(const std::string& path, const std::vector<int64_t>& data, size_t rows, size_t cols) {
    write_int64_2d(path, data.data(), rows, cols);
}

void NpyWriter::write_bool_1d(const std::string& path, const uint8_t* data, size_t size) {
    write_array(path, data, size * sizeof(uint8_t), "|b1", {size});
}

void NpyWriter::write_bool_1d(const std::string& path, const std::vector<uint8_t>& data) {
    write_bool_1d(path, data.data(), data.size());
}

} // namespace v15
