#!/bin/bash

# Test all CLI modes with ATR features
# Running minimal 1-epoch tests to verify functionality

PROJECT_DIR="/Users/frank/Desktop/CodingProjects/x7"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Testing ALL CLI modes with ATR features"
echo "=========================================="
echo ""

# Track results
RESULTS_FILE="$PROJECT_DIR/test_results.txt"
echo "Test Results - $(date)" > "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Test counter
TEST_NUM=0

# Function to run a test
run_test() {
    TEST_NUM=$((TEST_NUM + 1))
    local test_name="$1"
    shift
    local args="$@"
    
    echo ""
    echo "=========================================="
    echo "TEST $TEST_NUM: $test_name"
    echo "=========================================="
    echo "Command: python3 train_cli.py $args"
    echo ""
    
    # Run the test and capture output
    if python3 "$PROJECT_DIR/train_cli.py" $args --epochs 1 --no-interactive 2>&1 | tee /tmp/test_output.txt; then
        echo "✓ PASSED: $test_name" | tee -a "$RESULTS_FILE"
        
        # Check for feature dimensions in output
        if grep -q "809" /tmp/test_output.txt; then
            echo "  ✓ Feature dimension: 809 (correct with ATR)" | tee -a "$RESULTS_FILE"
        elif grep -q "776" /tmp/test_output.txt; then
            echo "  ✗ Feature dimension: 776 (missing ATR features!)" | tee -a "$RESULTS_FILE"
        fi
        
        # Check for ATR values
        if grep -qi "atr" /tmp/test_output.txt; then
            echo "  ✓ ATR features detected" | tee -a "$RESULTS_FILE"
        fi
        
    else
        echo "✗ FAILED: $test_name" | tee -a "$RESULTS_FILE"
        echo "  Error output:" | tee -a "$RESULTS_FILE"
        tail -20 /tmp/test_output.txt | tee -a "$RESULTS_FILE"
    fi
    echo "" >> "$RESULTS_FILE"
}

# ==========================================
# Test 1: Loss Types
# ==========================================
echo ""
echo "######################################"
echo "# Testing Different Loss Types"
echo "######################################"

run_test "Loss: Gaussian NLL (default)" \
    --mode quick --duration-loss gaussian_nll

run_test "Loss: Huber" \
    --mode quick --duration-loss huber --huber-delta 1.0

run_test "Loss: Survival" \
    --mode quick --duration-loss survival

# ==========================================
# Test 2: Training Modes
# ==========================================
echo ""
echo "######################################"
echo "# Testing Different Training Modes"
echo "######################################"

run_test "Mode: Standard (single split)" \
    --mode standard

run_test "Mode: Quick" \
    --mode quick

run_test "Mode: Walk-Forward (2 windows)" \
    --mode walk-forward --wf-windows 2 --wf-val-months 2

# ==========================================
# Test 3: Weight Modes
# ==========================================
echo ""
echo "######################################"
echo "# Testing Different Weight Modes"
echo "######################################"

run_test "Weight Mode: Fixed (balanced)" \
    --mode quick --weight-mode fixed_balanced

run_test "Weight Mode: Fixed (duration focus)" \
    --mode quick --weight-mode fixed_duration_focus

run_test "Weight Mode: Learnable" \
    --mode quick --weight-mode learnable

run_test "Weight Mode: Fixed Custom" \
    --mode quick --weight-mode fixed_custom \
    --weight-duration 3.0 --weight-direction 1.0 \
    --weight-next-channel 1.0 --weight-trigger-tf 1.5 \
    --weight-calibration 0.5

# ==========================================
# Test 4: Gradient Balancing
# ==========================================
echo ""
echo "######################################"
echo "# Testing Gradient Balancing Methods"
echo "######################################"

run_test "Gradient Balancing: None" \
    --mode quick --gradient-balancing none

run_test "Gradient Balancing: GradNorm" \
    --mode quick --gradient-balancing gradnorm --gradnorm-alpha 1.5

run_test "Gradient Balancing: PCGrad" \
    --mode quick --gradient-balancing pcgrad

# ==========================================
# Test 5: Two-Stage Training
# ==========================================
echo ""
echo "######################################"
echo "# Testing Two-Stage Training"
echo "######################################"

run_test "Two-Stage: Direction first" \
    --mode quick --two-stage-training \
    --stage1-task direction --stage1-epochs 1

run_test "Two-Stage: Duration first" \
    --mode quick --two-stage-training \
    --stage1-task duration --stage1-epochs 1

# ==========================================
# Test 6: Combination Tests
# ==========================================
echo ""
echo "######################################"
echo "# Testing Complex Combinations"
echo "######################################"

run_test "Combo: Huber + Learnable Weights + GradNorm" \
    --mode quick --duration-loss huber \
    --weight-mode learnable --gradient-balancing gradnorm

run_test "Combo: Survival + PCGrad + Two-Stage" \
    --mode quick --duration-loss survival \
    --gradient-balancing pcgrad --two-stage-training \
    --stage1-task direction --stage1-epochs 1

run_test "Combo: Walk-Forward + Learnable + GradNorm" \
    --mode walk-forward --wf-windows 2 --wf-val-months 2 \
    --weight-mode learnable --gradient-balancing gradnorm

# ==========================================
# Final Summary
# ==========================================
echo ""
echo "=========================================="
echo "ALL TESTS COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
cat "$RESULTS_FILE"
