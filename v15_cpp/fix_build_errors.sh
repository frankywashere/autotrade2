#!/bin/bash

# ============================================================================
# Quick Fix Script for v15_cpp Build Errors
# ============================================================================
# This script applies minimal fixes to get the code compiling.
# Run this before running build_manual.sh
# ============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Applying quick fixes for compilation errors..."

# Fix 1: Add default constructor to ChannelWorkItem
echo "[1/2] Adding default constructor to ChannelWorkItem..."

# Check if already fixed
if grep -q "ChannelWorkItem() =" "${PROJECT_ROOT}/include/scanner.hpp"; then
    echo "  Already fixed"
else
    # Create a backup
    cp "${PROJECT_ROOT}/include/scanner.hpp" "${PROJECT_ROOT}/include/scanner.hpp.backup"

    # Add default constructor after line 113
    sed -i.bak '113 a\
\
    ChannelWorkItem()\
        : primary_tf(""), primary_window(0), channel_idx(0) {}\
' "${PROJECT_ROOT}/include/scanner.hpp"

    echo "  Fixed: Added default constructor"
    echo "  Backup saved to: include/scanner.hpp.backup"
fi

# Fix 2: Remove duplicate Channel struct from channel_detector.hpp
echo "[2/2] Checking for duplicate Channel struct..."

if grep -q "^struct Channel {" "${PROJECT_ROOT}/include/channel_detector.hpp"; then
    echo "  Warning: channel_detector.hpp defines its own Channel struct"
    echo "  This conflicts with channel.hpp"
    echo "  Manual fix required: Remove Channel definition from channel_detector.hpp"
    echo "  and use the one from channel.hpp instead"
else
    echo "  No duplicate found"
fi

echo ""
echo "Quick fixes applied!"
echo ""
echo "Note: You may still see deprecation warnings for std::result_of"
echo "These are warnings, not errors, and won't prevent building."
echo ""
echo "Next step: Run ./build_manual.sh"
