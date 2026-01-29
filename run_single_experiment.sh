#!/bin/bash
# Run a single experiment by number
# Usage: ./run_single_experiment.sh <exp_number>
# Example: ./run_single_experiment.sh 1

set -e

SAMPLES="/Volumes/NVME2/samples_full.bin"
OUTPUT_DIR="/Volumes/NVME2/models"
LOG_DIR="logs/experiments"

# Common args for MPS compatibility
COMMON_ARGS="--num-workers 0 --no-feature-analysis"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_number>"
    echo ""
    echo "Available experiments:"
    echo "  1  - Higher Duration Weight (duration_weight=5.0)"
    echo "  2  - Even Higher Duration Weight (duration_weight=10.0)"
    echo "  3  - Larger Batch Size (batch_size=256)"
    echo "  4  - OneCycle Scheduler (lr=1e-3)"
    echo "  5  - Gaussian NLL Loss"
    echo "  6  - Strong Regularization (dropout=0.3, weight_decay=0.05)"
    echo "  7  - Lower LR with More Epochs (lr=5e-5, epochs=20)"
    echo "  8  - Cosine Annealing with Warm Restarts"
    echo "  9  - Large Scale Run (250k samples)"
    echo "  10 - Combined Best Practices"
    exit 1
fi

EXP=$1

# Helper to run and log (avoids pipe issues with tqdm)
run_and_log() {
    local log_file=$1
    shift
    echo "Logging to: $log_file"
    echo "Started at: $(date)"
    # Use Python's -u flag for unbuffered output
    python3 -u "$@" $COMMON_ARGS > "$log_file" 2>&1
    local exit_code=$?
    echo ""
    echo "--- Last 30 lines of log ---"
    tail -30 "$log_file"
    echo "--- End of log snippet ---"
    echo "Finished at: $(date)"
    return $exit_code
}

case $EXP in
    1)
        echo "Running Experiment 1: Higher Duration Weight"
        run_and_log "$LOG_DIR/exp1_duration_weight.log" \
            -m v15.pipeline train \
            --samples "$SAMPLES" \
            --output "$OUTPUT_DIR/exp1_duration_weight.pt" \
            --max-samples 50000 \
            --batch-size 128 \
            --epochs 10 \
            --lr 2e-4 \
            --duration-weight 5.0 \
            --duration-loss-type huber
        ;;
    2)
        echo "Running Experiment 2: Even Higher Duration Weight"
        run_and_log "$LOG_DIR/exp2_duration_weight_high.log" \
            -m v15.pipeline train \
            --samples "$SAMPLES" \
            --output "$OUTPUT_DIR/exp2_duration_weight_high.pt" \
            --max-samples 50000 \
            --batch-size 128 \
            --epochs 10 \
            --lr 2e-4 \
            --duration-weight 10.0 \
            --duration-loss-type huber
        ;;
    3)
        echo "Running Experiment 3: Larger Batch Size"
        run_and_log "$LOG_DIR/exp3_batch256.log" \
            -m v15.pipeline train \
            --samples "$SAMPLES" \
            --output "$OUTPUT_DIR/exp3_batch256.pt" \
            --max-samples 50000 \
            --batch-size 256 \
            --epochs 10 \
            --lr 3e-4 \
            --duration-weight 5.0 \
            --duration-loss-type huber
        ;;
    4)
        echo "Running Experiment 4: OneCycle Scheduler"
        run_and_log "$LOG_DIR/exp4_onecycle.log" \
            -m v15.pipeline train \
            --samples "$SAMPLES" \
            --output "$OUTPUT_DIR/exp4_onecycle.pt" \
            --max-samples 50000 \
            --batch-size 256 \
            --epochs 15 \
            --lr 1e-3 \
            --scheduler onecycle \
            --duration-weight 5.0 \
            --duration-loss-type huber
        ;;
    5)
        echo "Running Experiment 5: Gaussian NLL Loss"
        run_and_log "$LOG_DIR/exp5_gaussian_nll.log" \
            -m v15.pipeline train \
            --samples "$SAMPLES" \
            --output "$OUTPUT_DIR/exp5_gaussian_nll.pt" \
            --max-samples 50000 \
            --batch-size 128 \
            --epochs 10 \
            --lr 2e-4 \
            --duration-weight 5.0 \
            --duration-loss-type gaussian_nll
        ;;
    6)
        echo "Running Experiment 6: Strong Regularization"
        run_and_log "$LOG_DIR/exp6_regularized.log" \
            -m v15.pipeline train \
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
        ;;
    7)
        echo "Running Experiment 7: Lower LR with More Epochs"
        run_and_log "$LOG_DIR/exp7_slow_stable.log" \
            -m v15.pipeline train \
            --samples "$SAMPLES" \
            --output "$OUTPUT_DIR/exp7_slow_stable.pt" \
            --max-samples 100000 \
            --batch-size 128 \
            --epochs 20 \
            --lr 5e-5 \
            --weight-decay 0.01 \
            --duration-weight 5.0 \
            --duration-loss-type huber
        ;;
    8)
        echo "Running Experiment 8: Cosine Annealing with Warm Restarts"
        run_and_log "$LOG_DIR/exp8_cosine_restart.log" \
            -m v15.pipeline train \
            --samples "$SAMPLES" \
            --output "$OUTPUT_DIR/exp8_cosine_restart.pt" \
            --max-samples 50000 \
            --batch-size 128 \
            --epochs 15 \
            --lr 5e-4 \
            --scheduler cosine_restarts \
            --duration-weight 5.0 \
            --duration-loss-type huber
        ;;
    9)
        echo "Running Experiment 9: Large Scale Run"
        run_and_log "$LOG_DIR/exp9_large_scale.log" \
            -m v15.pipeline train \
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
        ;;
    10)
        echo "Running Experiment 10: Combined Best Practices"
        run_and_log "$LOG_DIR/exp10_optimal.log" \
            -m v15.pipeline train \
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
        ;;
    *)
        echo "Unknown experiment number: $EXP"
        echo "Valid range: 1-10"
        exit 1
        ;;
esac

echo ""
echo "Experiment $EXP completed!"
