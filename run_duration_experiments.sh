#!/bin/bash
# Hyperparameter Experiments for Duration Prediction
# Dataset: 996,037 samples, 14,840 features, 185GB (/Volumes/NVME2/samples_full.bin)
#
# Execution order follows the plan:
# Phase 1 - Quick Validation (Exp 1, 5, 6)
# Phase 2 - Scheduler Comparison (Exp 4, 8)
# Phase 3 - Scale Up (Exp 3, 7, 9)
# Phase 4 - Final Optimization (Exp 2, 10)

set -e

SAMPLES="/Volumes/NVME2/samples_full.bin"
OUTPUT_DIR="/Volumes/NVME2/models"
LOG_DIR="logs/experiments"

# Common args for MPS compatibility
COMMON_ARGS="--num-workers 0 --no-feature-analysis"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Helper function to run experiment with logging
run_experiment() {
    local exp_name=$1
    local exp_num=$2
    shift 2

    echo "========================================"
    echo "Starting Experiment $exp_num: $exp_name"
    echo "Time: $(date)"
    echo "========================================"

    local log_file="$LOG_DIR/exp${exp_num}_${exp_name}.log"

    # Run with output directly to log file (avoid pipe issues with tqdm)
    # Use Python's -u flag for unbuffered output
    echo "Logging to: $log_file"
    python3 -u -m v15.pipeline train "$@" $COMMON_ARGS > "$log_file" 2>&1
    local exit_code=$?

    # Show last 20 lines of log
    echo ""
    echo "--- Last 20 lines of log ---"
    tail -20 "$log_file"
    echo "--- End of log snippet ---"

    echo ""
    if [ $exit_code -eq 0 ]; then
        echo "Experiment $exp_num complete. Log saved to: $log_file"
    else
        echo "Experiment $exp_num FAILED (exit code: $exit_code). Check log: $log_file"
    fi
    echo ""

    return $exit_code
}

# Check if samples file exists
if [ ! -f "$SAMPLES" ]; then
    echo "ERROR: Samples file not found: $SAMPLES"
    exit 1
fi

echo "Starting Duration Prediction Hyperparameter Experiments"
echo "Dataset: $SAMPLES"
echo "Output directory: $OUTPUT_DIR"
echo ""

# ============================================================================
# PHASE 1 - Quick Validation (smaller samples, faster feedback)
# ============================================================================
echo "########################################"
echo "PHASE 1 - Quick Validation"
echo "########################################"
echo ""

# Experiment 1: Higher Duration Weight
# Rationale: Codex recommended duration_weight=5.0-10.0 for prioritizing duration head
run_experiment "duration_weight" 1 \
    --samples "$SAMPLES" \
    --output "$OUTPUT_DIR/exp1_duration_weight.pt" \
    --max-samples 50000 \
    --batch-size 128 \
    --epochs 10 \
    --lr 2e-4 \
    --duration-weight 5.0 \
    --duration-loss-type huber

# Experiment 5: Gaussian NLL Loss for Duration
# Rationale: Models uncertainty in duration predictions, handles noisy labels better
run_experiment "gaussian_nll" 5 \
    --samples "$SAMPLES" \
    --output "$OUTPUT_DIR/exp5_gaussian_nll.pt" \
    --max-samples 50000 \
    --batch-size 128 \
    --epochs 10 \
    --lr 2e-4 \
    --duration-weight 5.0 \
    --duration-loss-type gaussian_nll

# Experiment 6: Strong Regularization (Dropout + Weight Decay)
# Rationale: Combat overfitting observed after epoch 1
run_experiment "regularized" 6 \
    --samples "$SAMPLES" \
    --output "$OUTPUT_DIR/exp6_regularized.pt" \
    --max-samples 50000 \
    --batch-size 128 \
    --epochs 15 \
    --lr 2e-4 \
    --weight-decay 0.05 \
    --dropout 0.3 \
    --duration-weight 5.0 \
    --duration-loss-type huber

# ============================================================================
# PHASE 2 - Scheduler Comparison
# ============================================================================
echo "########################################"
echo "PHASE 2 - Scheduler Comparison"
echo "########################################"
echo ""

# Experiment 4: OneCycle Scheduler with Higher Max LR
# Rationale: Codex recommended OneCycle with max_lr=1e-3 for faster convergence
run_experiment "onecycle" 4 \
    --samples "$SAMPLES" \
    --output "$OUTPUT_DIR/exp4_onecycle.pt" \
    --max-samples 50000 \
    --batch-size 256 \
    --epochs 15 \
    --lr 1e-3 \
    --scheduler onecycle \
    --duration-weight 5.0 \
    --duration-loss-type huber

# Experiment 8: Cosine Annealing with Warm Restarts
# Rationale: Escape local minima through periodic LR resets
# Note: Using 'cosine_restarts' as that's what the trainer expects
run_experiment "cosine_restart" 8 \
    --samples "$SAMPLES" \
    --output "$OUTPUT_DIR/exp8_cosine_restart.pt" \
    --max-samples 50000 \
    --batch-size 128 \
    --epochs 15 \
    --lr 5e-4 \
    --scheduler cosine_restarts \
    --duration-weight 5.0 \
    --duration-loss-type huber

# ============================================================================
# PHASE 3 - Scale Up
# ============================================================================
echo "########################################"
echo "PHASE 3 - Scale Up"
echo "########################################"
echo ""

# Experiment 3: Larger Batch Size
# Rationale: More stable gradients, better GPU utilization
run_experiment "batch256" 3 \
    --samples "$SAMPLES" \
    --output "$OUTPUT_DIR/exp3_batch256.pt" \
    --max-samples 50000 \
    --batch-size 256 \
    --epochs 10 \
    --lr 3e-4 \
    --duration-weight 5.0 \
    --duration-loss-type huber

# Experiment 7: Lower LR with More Epochs
# Rationale: Slower, more stable convergence
run_experiment "slow_stable" 7 \
    --samples "$SAMPLES" \
    --output "$OUTPUT_DIR/exp7_slow_stable.pt" \
    --max-samples 100000 \
    --batch-size 128 \
    --epochs 20 \
    --lr 5e-5 \
    --weight-decay 0.01 \
    --duration-weight 5.0 \
    --duration-loss-type huber

# Experiment 9: Large Scale Run
# Rationale: Test with more of the full dataset
run_experiment "large_scale" 9 \
    --samples "$SAMPLES" \
    --output "$OUTPUT_DIR/exp9_large_scale.pt" \
    --max-samples 250000 \
    --batch-size 256 \
    --epochs 10 \
    --lr 1e-3 \
    --scheduler onecycle \
    --weight-decay 0.01 \
    --duration-weight 5.0 \
    --duration-loss-type huber

# ============================================================================
# PHASE 4 - Final Optimization
# ============================================================================
echo "########################################"
echo "PHASE 4 - Final Optimization"
echo "########################################"
echo ""

# Experiment 2: Even Higher Duration Weight
# Rationale: Test upper range of duration weighting
run_experiment "duration_weight_high" 2 \
    --samples "$SAMPLES" \
    --output "$OUTPUT_DIR/exp2_duration_weight_high.pt" \
    --max-samples 50000 \
    --batch-size 128 \
    --epochs 10 \
    --lr 2e-4 \
    --duration-weight 10.0 \
    --duration-loss-type huber

# Experiment 10: Combined Best Practices
# Rationale: Codex's recommended optimal configuration
run_experiment "optimal" 10 \
    --samples "$SAMPLES" \
    --output "$OUTPUT_DIR/exp10_optimal.pt" \
    --max-samples 100000 \
    --batch-size 256 \
    --epochs 15 \
    --lr 1e-3 \
    --scheduler onecycle \
    --weight-decay 0.02 \
    --dropout 0.2 \
    --duration-weight 7.0 \
    --duration-loss-type huber

# ============================================================================
# Summary
# ============================================================================
echo "========================================"
echo "All experiments completed!"
echo "Time: $(date)"
echo "========================================"
echo ""
echo "Results saved in: $OUTPUT_DIR/"
echo "Logs saved in: $LOG_DIR/"
echo ""
echo "Next steps:"
echo "1. Review validation losses in logs to identify best configuration"
echo "2. Check for overfitting patterns (val_loss increasing after epoch N)"
echo "3. Compare duration head loss specifically across experiments"
echo ""
ls -la "$OUTPUT_DIR/"
