# Manual Build Instructions for v15_cpp Scanner

This document describes how to build the v15_cpp scanner without CMake or package managers.

## Quick Start

```bash
# Build everything
./build_manual.sh

# Clean build directory
./build_manual.sh --clean

# Build without tests
./build_manual.sh --skip-tests

# Build without examples
./build_manual.sh --skip-examples
```

## Requirements

- **Compiler**: clang++ or g++ with C++17 support
- **Platform**: macOS or Linux
- **Tools**: curl or wget (for downloading Eigen3)

No package managers required!

## What the Script Does

The `build_manual.sh` script is completely self-contained and:

1. **Detects Compiler**: Automatically finds clang++ or g++ on your system
2. **Downloads Eigen3**: Fetches Eigen 3.4.0 headers if not already installed
3. **Compiles Library**: Builds all .cpp files in src/ to object files
4. **Creates Static Library**: Links objects into `libv15scanner.a`
5. **Builds Executables**: Compiles main scanner and test programs
6. **Provides Clear Output**: Shows build progress and results

## Build Artifacts

After a successful build, you'll find:

```
build_manual/
├── deps/
│   └── eigen3/          # Downloaded Eigen headers (if not system-installed)
├── obj/
│   ├── data_loader.o
│   ├── channel_detector.o
│   ├── indicators.o
│   ├── scanner.o
│   ├── feature_extractor.o
│   └── label_generator.o
├── lib/
│   └── libv15scanner.a  # Static library
└── bin/
    ├── v15_scanner      # Main scanner executable
    ├── test_*           # Test programs
    └── quick_start      # Example programs
```

## Using the Library

To link against the built library in your own programs:

```bash
# Compile your program
clang++ -std=c++17 -O3 -I./include -I./build_manual/deps/eigen3 \
    your_program.cpp \
    -L./build_manual/lib -lv15scanner \
    -o your_program
```

## Compiler Flags Used

The build script uses these optimized flags:

- `-std=c++17`: C++17 standard
- `-O3`: Maximum optimization
- `-march=native`: CPU-specific optimizations
- `-Wall -Wextra`: Enable warnings
- `-fPIC`: Position-independent code

## Troubleshooting

### Compilation Errors

**Current Known Issues** (as of Jan 24, 2026):

1. **Duplicate `Channel` struct definition**:
   - Defined in both `channel_detector.hpp` and `channel.hpp`
   - Solution: Use header guards or remove duplicate

2. **`ChannelWorkItem` missing default constructor**:
   - Required by `std::vector::resize()`
   - Solution: Add default constructor to struct

3. **Deprecated `std::result_of`**:
   - Used in ThreadPool implementation
   - Solution: Replace with `std::invoke_result` (C++17)

### Eigen Not Found

If the script can't download Eigen automatically:

```bash
# Manual download
cd build_manual/deps
curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz -o eigen.tar.gz
tar -xzf eigen.tar.gz
mv eigen-3.4.0 eigen3
```

### Permission Denied

```bash
chmod +x build_manual.sh
```

## Manual Build Steps

If you need to build manually without the script:

```bash
# 1. Download Eigen3
mkdir -p build_manual/deps
cd build_manual/deps
curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz -o eigen.tar.gz
tar -xzf eigen.tar.gz
mv eigen-3.4.0 eigen3
cd ../..

# 2. Compile source files
mkdir -p build_manual/obj
clang++ -std=c++17 -O3 -march=native -Wall -Wextra -fPIC \
    -I./include -I./build_manual/deps/eigen3 \
    -c src/data_loader.cpp -o build_manual/obj/data_loader.o

clang++ -std=c++17 -O3 -march=native -Wall -Wextra -fPIC \
    -I./include -I./build_manual/deps/eigen3 \
    -c src/channel_detector.cpp -o build_manual/obj/channel_detector.o

clang++ -std=c++17 -O3 -march=native -Wall -Wextra -fPIC \
    -I./include -I./build_manual/deps/eigen3 \
    -c src/indicators.cpp -o build_manual/obj/indicators.o

# ... repeat for other source files

# 3. Create static library
mkdir -p build_manual/lib
ar rcs build_manual/lib/libv15scanner.a build_manual/obj/*.o
ranlib build_manual/lib/libv15scanner.a

# 4. Compile main scanner
mkdir -p build_manual/bin
clang++ -std=c++17 -O3 -march=native \
    -I./include -I./build_manual/deps/eigen3 \
    src/main_scanner.cpp \
    -L./build_manual/lib -lv15scanner \
    -o build_manual/bin/v15_scanner
```

## Comparison with CMake Build

| Feature | Manual Build | CMake Build |
|---------|-------------|-------------|
| Dependencies | None (downloads Eigen) | Requires CMake, make |
| OpenMP | Not included | Auto-detected |
| Python Bindings | Not built | Built with pybind11 |
| Complexity | Simple shell script | Complex CMakeLists.txt |
| Portability | Bash on macOS/Linux | Cross-platform |
| Build Time | ~10-20 seconds | ~30-60 seconds |

## Script Options

```
Usage: ./build_manual.sh [OPTIONS]

Options:
  --skip-tests      Skip compiling test programs
  --skip-examples   Skip compiling example programs
  --clean           Remove build directory
  --help, -h        Show help message
```

## Next Steps

After building:

1. **Run the scanner**:
   ```bash
   ./build_manual/bin/v15_scanner --help
   ```

2. **Run tests**:
   ```bash
   ./build_manual/bin/test_data_loader
   ./build_manual/bin/test_channel_detector
   ```

3. **Fix compilation errors** (see Known Issues above)

4. **Integrate into your project** using the static library

## License

This build script is part of the v15_cpp project.
