# v15scanner - High-Performance C++ Scanner

A high-performance C++ implementation of the v15 trading system scanner with Python bindings.

## Features

- **High Performance**: Optimized with -O3, -march=native, and SIMD instructions
- **Parallel Processing**: OpenMP support for multi-threaded execution
- **Linear Algebra**: Eigen3 library for efficient matrix operations
- **Python Integration**: pybind11 bindings for seamless Python usage
- **Cross-Platform**: Supports both macOS and Linux

## Requirements

- CMake 3.15 or later
- C++17 compatible compiler (GCC 7+, Clang 5+, or MSVC 2017+)
- Python 3.7+ (for Python bindings)
- Git (for fetching dependencies)

### Optional Dependencies

The build system will automatically fetch these if not found:
- Eigen3 3.3+ (linear algebra library)
- pybind11 2.11+ (Python bindings)
- OpenMP (parallel processing - optional but recommended)

## Building the Project

### Quick Start

```bash
cd /Users/frank/Desktop/CodingProjects/x14/v15_cpp
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Build Types

**Release Build** (recommended for production):
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

**Debug Build** (for development):
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

### Platform-Specific Instructions

#### macOS

```bash
# Install dependencies via Homebrew (optional)
brew install cmake eigen pybind11 libomp

# Build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(sysctl -n hw.ncpu)
```

#### Linux

```bash
# Install dependencies (optional - CMake will fetch if missing)
sudo apt-get update
sudo apt-get install cmake libeigen3-dev pybind11-dev

# Build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Using Pre-installed Dependencies

If you have Eigen3 or pybind11 installed system-wide:

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DEigen3_DIR=/path/to/eigen3 \
      -Dpybind11_DIR=/path/to/pybind11 \
      ..
```

## Build Targets

- **v15scanner**: Core C++ library with scanner functionality
- **v15scanner_py**: Python module for Python integration

## Using the Python Module

After building, the Python module can be imported:

```python
import sys
sys.path.insert(0, '/Users/frank/Desktop/CodingProjects/x14/v15_cpp/build')

import v15scanner

# Use the scanner functions
# result = v15scanner.scan(...)
```

Or install the module:

```bash
cd build
cmake --install . --prefix ~/.local
```

Then add to your Python path:

```python
import sys
sys.path.insert(0, '~/.local/python')
import v15scanner
```

## Testing

To build and run tests:

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON ..
make -j$(nproc)
ctest --output-on-failure
```

## Project Structure

```
v15_cpp/
├── CMakeLists.txt          # Build configuration
├── README.md               # This file
├── include/                # Public header files
├── src/                    # C++ implementation files
├── python_bindings/        # pybind11 Python bindings
├── tests/                  # Test files
└── build/                  # Build directory (generated)
```

## Optimization Flags

The Release build uses the following optimization flags:

- `-O3`: Maximum optimization level
- `-march=native`: Optimize for the current CPU architecture
- `-DNDEBUG`: Disable debug assertions

Additional flags:
- `-Wall -Wextra -Wpedantic`: Enable comprehensive warnings
- OpenMP flags (if available): Enable parallel processing

## Performance Tips

1. **Always use Release builds** for production/benchmarking
2. **Enable OpenMP**: Ensure OpenMP is available for parallel processing
3. **CPU-specific optimization**: The `-march=native` flag optimizes for your specific CPU
4. **Link-time optimization**: Add `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON` for LTO

Example with LTO:

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
      ..
```

## Troubleshooting

### OpenMP not found on macOS

```bash
brew install libomp
cmake -DCMAKE_BUILD_TYPE=Release \
      -DOpenMP_ROOT=$(brew --prefix libomp) \
      ..
```

### Eigen3 not found

The build system will automatically fetch Eigen3 from GitLab. If you encounter issues:

```bash
# Install via package manager
# macOS:
brew install eigen

# Linux:
sudo apt-get install libeigen3-dev
```

### pybind11 not found

```bash
# Install via pip
pip install pybind11

# Or via package manager
# macOS:
brew install pybind11

# Linux:
sudo apt-get install pybind11-dev
```

### Python module not loading

Make sure the Python version used to build matches the Python version you're using:

```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE=$(which python3) \
      ..
```

## Clean Build

To start fresh:

```bash
rm -rf build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

## Contributing

When adding new source files:

1. Place `.cpp` files in `src/`
2. Place `.h`/`.hpp` files in `include/`
3. Place Python bindings in `python_bindings/`
4. CMake will automatically detect and compile them

## License

[Add your license information here]

## Contact

[Add contact information here]
