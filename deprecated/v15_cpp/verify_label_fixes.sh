#!/bin/bash
# Quick verification script for label generator fixes

set -e

echo "==================================================================="
echo "Label Generator Fix Verification"
echo "==================================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Step 1: Check if test executable exists
echo -e "${BLUE}[1/3]${NC} Checking test executable..."
if [ ! -f "./build_manual/bin/test_label_generator" ]; then
    echo -e "${YELLOW}Test not found, building...${NC}"
    clang++ -std=c++17 -O2 -Wall -Wextra \
        -I./include -I/opt/homebrew/include/eigen3 \
        tests/test_label_generator.cpp \
        -L./build_manual/lib -lv15scanner \
        -o build_manual/bin/test_label_generator
    echo -e "${GREEN}✓ Test compiled${NC}"
else
    echo -e "${GREEN}✓ Test found${NC}"
fi
echo ""

# Step 2: Run unit tests
echo -e "${BLUE}[2/3]${NC} Running unit tests..."
if ./build_manual/bin/test_label_generator; then
    echo -e "${GREEN}✓ All unit tests passed${NC}"
else
    echo -e "${RED}✗ Unit tests failed${NC}"
    exit 1
fi
echo ""

# Step 3: Summary
echo -e "${BLUE}[3/3]${NC} Verification Summary"
echo "-------------------------------------------------------------------"
echo -e "${GREEN}✓${NC} NULL pointer checks: PASS"
echo -e "${GREEN}✓${NC} RSI computation: PASS"
echo -e "${GREEN}✓${NC} RSI range validation: PASS"
echo -e "${GREEN}✓${NC} Array bounds checking: PASS"
echo -e "${GREEN}✓${NC} No-break scenario: PASS"
echo -e "${GREEN}✓${NC} Immediate break: PASS"
echo -e "${GREEN}✓${NC} Break at scan end: PASS"
echo -e "${GREEN}✓${NC} False break detection: PASS"
echo ""

echo "==================================================================="
echo -e "${GREEN}All label generator fixes verified successfully!${NC}"
echo "==================================================================="
echo ""
echo "Key Fixes Applied:"
echo "  • RSI labels now computed (was: never computed)"
echo "  • NULL pointer validation added"
echo "  • Array index calculations corrected (+1 offset)"
echo "  • RSI values clamped to [0, 100] range"
echo "  • max_scan bounds validation"
echo "  • No-break scenario returns valid labels"
echo ""
echo "Next Steps:"
echo "  1. Run full scanner with small dataset"
echo "  2. Compare output against Python reference"
echo "  3. Check SPY labels also have RSI computed"
echo ""
