#!/bin/bash
# Per-TF Duration MAE Experiments
# Run locally on 50K samples to test different approaches
# Usage: bash experiments/run_per_tf_experiments.sh [experiment_name]

set -e

SAMPLES="data/samples_50k.flat"
BASE_DIR="experiments/results"
EPOCHS=30
BATCH_SIZE=64
LR=2e-4
COMMON_ARGS="--samples $SAMPLES --batch-size $BATCH_SIZE --lr $LR --epochs $EPOCHS \
  --weight-decay 0.01 --duration-weight 5.0 --duration-loss-type gaussian_nll \
  --num-workers 0 --no-feature-analysis --scheduler onecycle \
  --grad-clip 0.5 --warmup-steps 200 --early-stopping-patience 15"

run_experiment() {
    local name=$1
    shift
    local output="$BASE_DIR/$name"
    mkdir -p "$output"
    echo "=========================================="
    echo "Running experiment: $name"
    echo "Output: $output"
    echo "=========================================="
    python3 -u -m v15.pipeline train --output "$output" $COMMON_ARGS "$@" 2>&1 | tee "$output/train.log"
    echo ""
    echo "Experiment $name complete."
    echo ""
}

mkdir -p "$BASE_DIR"

case "${1:-all}" in
    baseline)
        # Current production settings
        run_experiment "baseline" \
            --per-tf-loss-weight 0.5 \
            --per-tf-direction-loss-weight 0.3 \
            --per-tf-loss-ramp-epochs 50
        ;;
    high_weight)
        # Higher weight, no ramp — more gradient signal to per-TF heads
        run_experiment "high_weight" \
            --per-tf-loss-weight 2.0 \
            --per-tf-direction-loss-weight 0.3 \
            --per-tf-loss-ramp-epochs 0
        ;;
    no_ramp)
        # Same weight but no ramp — full signal from epoch 1
        run_experiment "no_ramp" \
            --per-tf-loss-weight 0.5 \
            --per-tf-direction-loss-weight 0.3 \
            --per-tf-loss-ramp-epochs 0
        ;;
    v2_head)
        # V2 architecture: TF embedding + bigger hidden dim + LayerNorm
        run_experiment "v2_head" \
            --per-tf-head-version 2 \
            --per-tf-loss-weight 2.0 \
            --per-tf-direction-loss-weight 0.3 \
            --per-tf-loss-ramp-epochs 0
        ;;
    all)
        run_experiment "baseline" \
            --per-tf-loss-weight 0.5 \
            --per-tf-direction-loss-weight 0.3 \
            --per-tf-loss-ramp-epochs 50

        run_experiment "high_weight" \
            --per-tf-loss-weight 2.0 \
            --per-tf-direction-loss-weight 0.3 \
            --per-tf-loss-ramp-epochs 0

        run_experiment "no_ramp" \
            --per-tf-loss-weight 0.5 \
            --per-tf-direction-loss-weight 0.3 \
            --per-tf-loss-ramp-epochs 0

        run_experiment "v2_head" \
            --per-tf-head-version 2 \
            --per-tf-loss-weight 2.0 \
            --per-tf-direction-loss-weight 0.3 \
            --per-tf-loss-ramp-epochs 0
        ;;
    compare)
        # Compare results across all experiments
        echo ""
        echo "=========================================="
        echo "Experiment Comparison"
        echo "=========================================="
        printf "  %-20s %5s  %10s  %8s  %8s  %8s\n" "Name" "Epoch" "Val Loss" "Dur MAE" "PTF MAE" "PTF Dir"
        echo "  -------------------------------------------------------------------"
        for dir in "$BASE_DIR"/*/; do
            name=$(basename "$dir")
            if [ -f "$dir/latest.pt" ]; then
                python3 -c "
import torch
cp = torch.load('$dir/latest.pt', map_location='cpu', weights_only=False)
m = cp.get('metrics', {})
epoch = cp.get('epoch', '?')
val_loss = m.get('val_total', [0])[-1] if 'val_total' in m else 0
dur_mae = m.get('val_duration_mae', [0])[-1] if 'val_duration_mae' in m else 0
ptf_mae = m.get('val_per_tf_duration_mae', [0])[-1] if 'val_per_tf_duration_mae' in m else 0
ptf_dir = m.get('val_per_tf_direction', [0])[-1] if 'val_per_tf_direction' in m else 0
print(f'  {\"$name\":20s} {epoch:5d}  {val_loss:10.4f}  {dur_mae:8.1f}  {ptf_mae:8.1f}  {ptf_dir:8.4f}')
"
            fi
        done
        echo ""
        ;;
    *)
        echo "Usage: $0 [baseline|high_weight|no_ramp|v2_head|all|compare]"
        ;;
esac
