#!/bin/bash
#
# V15 C++ Scanner - Installation Script
# Version: 1.0.0
# Date: 2026-01-25
#
# Usage:
#   ./install.sh              # Build only
#   ./install.sh --install    # Build and install to ~/.local
#   ./install.sh --system     # Build and install system-wide (requires sudo)
#   ./install.sh --python     # Build Python bindings only
#   ./install.sh --clean      # Clean build directory
#   ./install.sh --help       # Show help
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Installation modes
INSTALL_MODE="none"
INSTALL_PREFIX=""
BUILD_PYTHON=true
BUILD_TESTS=false
CLEAN_BUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --install)
            INSTALL_MODE="user"
            INSTALL_PREFIX="$HOME/.local"
            shift
            ;;
        --system)
            INSTALL_MODE="system"
            INSTALL_PREFIX="/usr/local"
            shift
            ;;
        --prefix)
            INSTALL_MODE="custom"
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --python)
            BUILD_PYTHON=true
            shift
            ;;
        --no-python)
            BUILD_PYTHON=false
            shift
            ;;
        --tests)
            BUILD_TESTS=true
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help|-h)
            cat << EOF
V15 C++ Scanner Installation Script

Usage:
  ./install.sh [OPTIONS]

Options:
  --install           Build and install to ~/.local
  --system            Build and install system-wide (requires sudo)
  --prefix PATH       Install to custom location
  --python            Build Python bindings (default)
  --no-python         Skip Python bindings
  --tests             Build tests
  --clean             Clean build directory before building
  --help, -h          Show this help message

Environment Variables:
  BUILD_TYPE          Build type (Release, Debug, RelWithDebInfo) [default: Release]
  NUM_CORES           Number of parallel build jobs [default: auto-detect]

Examples:
  ./install.sh                          # Build only
  ./install.sh --install                # Install to ~/.local
  ./install.sh --system                 # Install to /usr/local (requires sudo)
  ./install.sh --prefix /opt/v15        # Install to /opt/v15
  ./install.sh --clean --install        # Clean build and install
  BUILD_TYPE=Debug ./install.sh         # Debug build

EOF
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Run './install.sh --help' for usage information"
            exit 1
            ;;
    esac
done

# Helper functions
print_step() {
    echo -e "${BLUE}==>${NC} ${GREEN}$1${NC}"
}

print_info() {
    echo -e "${BLUE}  ->${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

print_error() {
    echo -e "${RED}Error:${NC} $1"
}

print_success() {
    echo -e "${GREEN}Success:${NC} $1"
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        print_error "$1 is required but not installed."
        return 1
    fi
    return 0
}

# Verify prerequisites
verify_prerequisites() {
    print_step "Verifying prerequisites..."

    local missing_deps=()

    # Check CMake
    if ! check_command cmake; then
        missing_deps+=("cmake (>= 3.15)")
    else
        CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
        print_info "CMake version: $CMAKE_VERSION"
    fi

    # Check C++ compiler
    if check_command g++; then
        CXX_COMPILER="g++"
        CXX_VERSION=$(g++ --version | head -n1)
        print_info "C++ compiler: $CXX_VERSION"
    elif check_command clang++; then
        CXX_COMPILER="clang++"
        CXX_VERSION=$(clang++ --version | head -n1)
        print_info "C++ compiler: $CXX_VERSION"
    else
        missing_deps+=("C++ compiler (g++ or clang++)")
    fi

    # Check Python (optional for bindings)
    if $BUILD_PYTHON; then
        if check_command python3; then
            PYTHON_VERSION=$(python3 --version | awk '{print $2}')
            print_info "Python version: $PYTHON_VERSION"
        else
            print_warning "Python3 not found - Python bindings will be skipped"
            BUILD_PYTHON=false
        fi
    fi

    # Check Git (for fetching dependencies)
    if ! check_command git; then
        print_warning "Git not found - dependency auto-fetch may fail"
    fi

    # Report missing dependencies
    if [ ${#missing_deps[@]} -gt 0 ]; then
        print_error "Missing required dependencies:"
        for dep in "${missing_deps[@]}"; do
            echo "  - $dep"
        done
        echo ""
        echo "Install dependencies:"
        echo "  macOS:   brew install cmake"
        echo "  Ubuntu:  sudo apt-get install build-essential cmake"
        echo "  RHEL:    sudo yum install gcc-c++ cmake3"
        exit 1
    fi

    print_success "All prerequisites satisfied"
    echo ""
}

# Clean build directory
clean_build() {
    print_step "Cleaning build directory..."

    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        print_info "Removed $BUILD_DIR"
    fi

    # Also clean object files in root
    find "$PROJECT_DIR" -maxdepth 1 -name "*.o" -delete 2>/dev/null || true

    print_success "Build directory cleaned"
    echo ""
}

# Build the scanner
build_scanner() {
    print_step "Building V15 Scanner..."

    # Create build directory
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    # Configure CMake
    print_info "Configuring CMake (Build type: $BUILD_TYPE)..."

    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    )

    # Add tests flag
    if $BUILD_TESTS; then
        CMAKE_ARGS+=(-DBUILD_TESTS=ON)
    fi

    # Enable Link-Time Optimization for Release builds
    if [ "$BUILD_TYPE" = "Release" ]; then
        CMAKE_ARGS+=(-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON)
    fi

    # Run CMake
    if ! cmake "${CMAKE_ARGS[@]}" ..; then
        print_error "CMake configuration failed"
        exit 1
    fi

    # Build
    print_info "Building with $NUM_CORES parallel jobs..."

    if ! cmake --build . -j"$NUM_CORES"; then
        print_error "Build failed"
        exit 1
    fi

    print_success "Build completed"
    echo ""
}

# Verify build outputs
verify_build() {
    print_step "Verifying build outputs..."

    cd "$BUILD_DIR"

    local build_ok=true

    # Check executable
    if [ -f "v15_scanner" ]; then
        print_info "Executable: v15_scanner"

        # Test executable
        if ./v15_scanner --version &>/dev/null; then
            print_info "Executable test: PASS"
        else
            print_warning "Executable test: FAIL (but file exists)"
        fi
    else
        print_error "Executable not found: v15_scanner"
        build_ok=false
    fi

    # Check library
    if [ -f "libv15scanner.a" ] || [ -f "libv15scanner.dylib" ] || [ -f "libv15scanner.so" ]; then
        print_info "Library: $(ls libv15scanner.* 2>/dev/null | head -1)"
    else
        print_warning "Library not found (may be header-only)"
    fi

    # Check Python module
    if $BUILD_PYTHON; then
        PYTHON_MODULE=$(find . -name "v15scanner*.so" -o -name "v15scanner*.dylib" 2>/dev/null | head -1)
        if [ -n "$PYTHON_MODULE" ]; then
            print_info "Python module: $(basename $PYTHON_MODULE)"

            # Test Python import
            if python3 -c "import sys; sys.path.insert(0, '$BUILD_DIR'); import v15scanner_py" &>/dev/null; then
                print_info "Python import test: PASS"
            else
                print_warning "Python import test: FAIL (but module exists)"
            fi
        else
            print_warning "Python module not found (pybind11 may be missing)"
        fi
    fi

    if ! $build_ok; then
        print_error "Build verification failed"
        exit 1
    fi

    print_success "Build verification completed"
    echo ""
}

# Install scanner
install_scanner() {
    print_step "Installing V15 Scanner..."

    if [ "$INSTALL_MODE" = "none" ]; then
        print_info "Skipping installation (build only mode)"
        return
    fi

    cd "$BUILD_DIR"

    # Check if sudo is needed
    local SUDO=""
    if [ "$INSTALL_MODE" = "system" ]; then
        SUDO="sudo"
        print_info "Installing to $INSTALL_PREFIX (requires sudo)"
    else
        print_info "Installing to $INSTALL_PREFIX"
    fi

    # Create installation directories
    $SUDO mkdir -p "$INSTALL_PREFIX/bin"
    $SUDO mkdir -p "$INSTALL_PREFIX/lib"
    $SUDO mkdir -p "$INSTALL_PREFIX/include/v15scanner"

    # Install executable
    if [ -f "v15_scanner" ]; then
        $SUDO cp v15_scanner "$INSTALL_PREFIX/bin/"
        $SUDO chmod 755 "$INSTALL_PREFIX/bin/v15_scanner"
        print_info "Installed: $INSTALL_PREFIX/bin/v15_scanner"
    fi

    # Install library
    for lib in libv15scanner.*; do
        if [ -f "$lib" ]; then
            $SUDO cp "$lib" "$INSTALL_PREFIX/lib/"
            print_info "Installed: $INSTALL_PREFIX/lib/$lib"
        fi
    done

    # Install headers
    if [ -d "$PROJECT_DIR/include" ]; then
        $SUDO cp -r "$PROJECT_DIR/include/"* "$INSTALL_PREFIX/include/v15scanner/"
        print_info "Installed headers to: $INSTALL_PREFIX/include/v15scanner/"
    fi

    # Install Python module
    if $BUILD_PYTHON; then
        PYTHON_MODULE=$(find . -name "v15scanner*.so" -o -name "v15scanner*.dylib" 2>/dev/null | head -1)
        if [ -n "$PYTHON_MODULE" ]; then
            PYTHON_SITE_PACKAGES=$($SUDO python3 -c "import site; print(site.USER_SITE if '$INSTALL_MODE' == 'user' else site.getsitepackages()[0])" 2>/dev/null)
            if [ -n "$PYTHON_SITE_PACKAGES" ]; then
                $SUDO mkdir -p "$PYTHON_SITE_PACKAGES"
                $SUDO cp "$PYTHON_MODULE" "$PYTHON_SITE_PACKAGES/"
                print_info "Installed Python module to: $PYTHON_SITE_PACKAGES/"
            else
                # Fallback to prefix/python
                $SUDO mkdir -p "$INSTALL_PREFIX/python"
                $SUDO cp "$PYTHON_MODULE" "$INSTALL_PREFIX/python/"
                print_info "Installed Python module to: $INSTALL_PREFIX/python/"
            fi
        fi
    fi

    print_success "Installation completed"
    echo ""
}

# Setup environment
setup_environment() {
    print_step "Environment setup..."

    if [ "$INSTALL_MODE" = "user" ]; then
        # Detect shell
        SHELL_RC=""
        if [ -n "$BASH_VERSION" ]; then
            SHELL_RC="$HOME/.bashrc"
        elif [ -n "$ZSH_VERSION" ]; then
            SHELL_RC="$HOME/.zshrc"
        fi

        if [ -n "$SHELL_RC" ]; then
            print_info "Shell configuration file: $SHELL_RC"

            # Check if PATH already includes install prefix
            if ! grep -q "$INSTALL_PREFIX/bin" "$SHELL_RC" 2>/dev/null; then
                echo ""
                print_info "To use the scanner, add to your shell profile ($SHELL_RC):"
                echo ""
                echo "    export PATH=\"$INSTALL_PREFIX/bin:\$PATH\""
                if $BUILD_PYTHON; then
                    PYTHON_PATH="$INSTALL_PREFIX/python"
                    echo "    export PYTHONPATH=\"$PYTHON_PATH:\$PYTHONPATH\""
                fi
                echo ""
                print_info "Or run now:"
                echo ""
                echo "    export PATH=\"$INSTALL_PREFIX/bin:\$PATH\""
                if $BUILD_PYTHON; then
                    echo "    export PYTHONPATH=\"$PYTHON_PATH:\$PYTHONPATH\""
                fi
                echo ""
            else
                print_info "PATH already includes $INSTALL_PREFIX/bin"
            fi
        fi
    elif [ "$INSTALL_MODE" = "system" ]; then
        # System installation - verify PATH
        if echo "$PATH" | grep -q "/usr/local/bin"; then
            print_success "System installation complete - no PATH setup needed"
        else
            print_warning "/usr/local/bin not in PATH - you may need to add it"
        fi
    fi

    echo ""
}

# Print final summary
print_summary() {
    print_step "Installation Summary"
    echo ""

    if [ "$INSTALL_MODE" != "none" ]; then
        echo "Installation location: $INSTALL_PREFIX"
        echo ""
        echo "Installed files:"
        echo "  - Executable: $INSTALL_PREFIX/bin/v15_scanner"
        echo "  - Headers:    $INSTALL_PREFIX/include/v15scanner/"
        if $BUILD_PYTHON; then
            echo "  - Python:     $(find $INSTALL_PREFIX -name "v15scanner*.so" -o -name "v15scanner*.dylib" 2>/dev/null | head -1 || echo "Not found")"
        fi
        echo ""
    else
        echo "Build location: $BUILD_DIR"
        echo ""
        echo "Built files:"
        echo "  - Executable: $BUILD_DIR/v15_scanner"
        if $BUILD_PYTHON; then
            echo "  - Python:     $(find $BUILD_DIR -name "v15scanner*.so" -o -name "v15scanner*.dylib" 2>/dev/null | head -1 || echo "Not found")"
        fi
        echo ""
    fi

    echo "Quick test:"
    if [ "$INSTALL_MODE" != "none" ]; then
        echo "  $INSTALL_PREFIX/bin/v15_scanner --version"
    else
        echo "  $BUILD_DIR/v15_scanner --version"
    fi
    echo ""

    if $BUILD_PYTHON; then
        echo "Python test:"
        if [ "$INSTALL_MODE" != "none" ]; then
            echo "  python3 -c 'import v15scanner_py; print(v15scanner_py.__version__)'"
        else
            echo "  cd $BUILD_DIR && python3 -c 'import v15scanner_py; print(v15scanner_py.__version__)'"
        fi
        echo ""
    fi

    echo "Next steps:"
    echo "  1. Run scanner: v15_scanner --data-dir /path/to/data --output samples.bin"
    echo "  2. Read documentation: cat DEPLOYMENT.md"
    echo "  3. View quick reference: cat QUICKSTART.txt"
    echo ""

    print_success "Installation script completed successfully!"
}

# Main execution
main() {
    echo ""
    echo "========================================"
    echo "V15 C++ Scanner - Installation Script"
    echo "========================================"
    echo ""

    # Clean if requested
    if $CLEAN_BUILD; then
        clean_build
    fi

    # Run installation steps
    verify_prerequisites
    build_scanner
    verify_build
    install_scanner
    setup_environment
    print_summary
}

# Run main function
main
