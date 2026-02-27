cmake_minimum_required(VERSION 3.15...3.27)

project(v15scanner
    VERSION 1.0.0
    DESCRIPTION "High-performance C++ scanner for v15 trading system"
    LANGUAGES CXX
)

# ============================================================================
# Build Configuration
# ============================================================================

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDEs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Position independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Compiler Flags
# ============================================================================

# Maximum optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Warning flags
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Debug>:-O0>
)

# Platform-specific settings
if(APPLE)
    # macOS specific flags
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
endif()

# ============================================================================
# Dependencies
# ============================================================================

# Find Eigen3 for linear algebra
find_package(Eigen3 3.3 QUIET)

if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found locally, fetching from GitLab...")
    include(FetchContent)
    FetchContent_Declare(
        Eigen3
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
    )
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "Don't build Eigen docs" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Don't build Eigen tests" FORCE)
    set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Don't build pkg-config" FORCE)
    FetchContent_MakeAvailable(Eigen3)
endif()

# Find pybind11 for Python bindings
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found locally, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# Find OpenMP for parallel processing
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
else()
    message(WARNING "OpenMP not found - parallel processing will be disabled")
endif()

# ============================================================================
# Source Files
# ============================================================================

# Collect library source files (exclude main_*.cpp files)
file(GLOB SCANNER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/data_loader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/channel_detector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/indicators.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/scanner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/serialization.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/feature_extractor.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/label_generator.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/flat_writer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/npy_writer.cpp"
)

# Collect all header files
file(GLOB_RECURSE SCANNER_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

# Executable source files
set(MAIN_SCANNER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/main_scanner.cpp")

# Python binding files
file(GLOB PYTHON_BINDING_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/python_bindings/*.cpp"
)

# ============================================================================
# Core Library Target
# ============================================================================

if(SCANNER_SOURCES)
    # Create core scanner library
    add_library(v15scanner ${SCANNER_SOURCES} ${SCANNER_HEADERS})

    # Set target properties
    set_target_properties(v15scanner PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        PUBLIC_HEADER "${SCANNER_HEADERS}"
    )

    # Include directories
    target_include_directories(v15scanner
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    # Link dependencies
    target_link_libraries(v15scanner
        PUBLIC
            Eigen3::Eigen
    )

    # Link OpenMP if available
    if(OpenMP_CXX_FOUND)
        target_link_libraries(v15scanner PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(v15scanner PUBLIC USE_OPENMP)
    endif()

    # Compiler features
    target_compile_features(v15scanner PUBLIC cxx_std_17)

else()
    message(STATUS "No source files found in src/ - skipping library target")
    # Create interface library as placeholder
    add_library(v15scanner INTERFACE)
    target_include_directories(v15scanner
        INTERFACE
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(v15scanner INTERFACE Eigen3::Eigen)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(v15scanner INTERFACE OpenMP::OpenMP_CXX)
    endif()
endif()

# ============================================================================
# Python Module Target
# ============================================================================

if(PYTHON_BINDING_SOURCES)
    # Create Python module
    pybind11_add_module(v15scanner_py ${PYTHON_BINDING_SOURCES})

    # Set target properties
    set_target_properties(v15scanner_py PROPERTIES
        OUTPUT_NAME "v15scanner_cpp"
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )

    # Link to core library
    if(SCANNER_SOURCES)
        target_link_libraries(v15scanner_py PRIVATE v15scanner)
    else()
        # If no core library, link dependencies directly
        target_include_directories(v15scanner_py PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(v15scanner_py PRIVATE Eigen3::Eigen)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(v15scanner_py PRIVATE OpenMP::OpenMP_CXX)
            target_compile_definitions(v15scanner_py PRIVATE USE_OPENMP)
        endif()
    endif()

    # Additional optimization for Python module
    target_compile_options(v15scanner_py PRIVATE
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Release>:-march=native>
        $<$<CONFIG:Release>:-DNDEBUG>
    )

else()
    message(STATUS "No Python binding files found in python_bindings/ - skipping Python module")
endif()

# ============================================================================
# Executable Targets
# ============================================================================

# Scanner executable
if(EXISTS ${MAIN_SCANNER_SOURCE})
    add_executable(v15_scanner ${MAIN_SCANNER_SOURCE})

    # Link to core library
    if(SCANNER_SOURCES)
        target_link_libraries(v15_scanner PRIVATE v15scanner)
    else()
        # If no core library, link dependencies directly
        target_include_directories(v15_scanner PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        )
        target_link_libraries(v15_scanner PRIVATE Eigen3::Eigen)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(v15_scanner PRIVATE OpenMP::OpenMP_CXX)
            target_compile_definitions(v15_scanner PRIVATE USE_OPENMP)
        endif()
    endif()

    # Platform-specific libraries (for getopt)
    if(NOT APPLE AND NOT WIN32)
        # Linux typically has getopt in libc
    elseif(APPLE)
        # macOS has getopt in libc
    endif()

    message(STATUS "v15_scanner executable: YES")
else()
    message(STATUS "v15_scanner executable: NO (main_scanner.cpp not found)")
endif()

# ============================================================================
# Tests (Optional)
# ============================================================================

# Check if tests exist
file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp")

if(TEST_SOURCES)
    option(BUILD_TESTS "Build test executables" OFF)

    if(BUILD_TESTS)
        enable_testing()

        # Build standard unit tests
        set(UNIT_TEST_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_channel_detector.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_data_loader.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_indicators.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_label_generator.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/integration_test.cpp"
        )

        foreach(TEST_FILE ${UNIT_TEST_SOURCES})
            if(EXISTS ${TEST_FILE})
                get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
                add_executable(${TEST_NAME} ${TEST_FILE})
                target_link_libraries(${TEST_NAME} PRIVATE v15scanner)
                add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
                message(STATUS "  Added test: ${TEST_NAME}")
            endif()
        endforeach()

        # Build validation programs (separate from unit tests)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/validate_against_python.cpp")
            add_executable(validate_against_python
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/validate_against_python.cpp"
            )
            target_link_libraries(validate_against_python PRIVATE v15scanner)
            message(STATUS "  Added validation: validate_against_python")
        endif()

        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark.cpp")
            add_executable(benchmark
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark.cpp"
            )
            target_link_libraries(benchmark PRIVATE v15scanner)
            message(STATUS "  Added benchmark: benchmark")
        endif()

        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_resampling.cpp")
            add_executable(test_resampling
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_resampling.cpp"
            )
            target_link_libraries(test_resampling PRIVATE v15scanner)
            message(STATUS "  Added benchmark: benchmark")
        endif()
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

# Install library
if(SCANNER_SOURCES)
    install(TARGETS v15scanner
        EXPORT v15scannerTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/v15scanner
    )
endif()

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/v15scanner
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install Python module
if(PYTHON_BINDING_SOURCES)
    install(TARGETS v15scanner_py
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/python
    )
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== v15scanner Build Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  Eigen3: ${Eigen3_FOUND} ${EIGEN3_VERSION_STRING}")
message(STATUS "  pybind11: ${pybind11_FOUND} ${pybind11_VERSION}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "")
message(STATUS "Targets:")
if(SCANNER_SOURCES)
    message(STATUS "  v15scanner library: YES")
else()
    message(STATUS "  v15scanner library: NO (no source files)")
endif()
if(EXISTS ${MAIN_SCANNER_SOURCE})
    message(STATUS "  v15_scanner executable: YES")
else()
    message(STATUS "  v15_scanner executable: NO")
endif()
if(PYTHON_BINDING_SOURCES)
    message(STATUS "  v15scanner_py module: YES")
else()
    message(STATUS "  v15scanner_py module: NO (no binding files)")
endif()
message(STATUS "")
message(STATUS "Optimization flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "======================================")
message(STATUS "")
