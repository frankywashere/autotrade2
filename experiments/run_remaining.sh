#!/bin/bash
set -e
cd /Users/frank/Desktop/CodingProjects/x14

SAMPLES="data/samples_50k.flat"
BASE_DIR="experiments/results"
COMMON="--samples $SAMPLES --batch-size 64 --lr 2e-4 --epochs 30 --weight-decay 0.01 --duration-weight 5.0 --duration-loss-type gaussian_nll --num-workers 0 --no-feature-analysis --scheduler onecycle --grad-clip 0.5 --warmup-steps 200 --early-stopping-patience 15"

echo "=== v2_low_weight (V2, weight=0.5, no ramp) ==="
mkdir -p "$BASE_DIR/v2_low_weight"
python3 -u -m v15.pipeline train --output "$BASE_DIR/v2_low_weight" $COMMON \
    --per-tf-head-version 2 --per-tf-loss-weight 0.5 --per-tf-direction-loss-weight 0.3 --per-tf-loss-ramp-epochs 0 \
    2>&1 | grep "INFO.*Epoch\|complete"

echo "=== v2_very_high_weight (V2, weight=4.0, no ramp) ==="
mkdir -p "$BASE_DIR/v2_very_high_weight"
python3 -u -m v15.pipeline train --output "$BASE_DIR/v2_very_high_weight" $COMMON \
    --per-tf-head-version 2 --per-tf-loss-weight 4.0 --per-tf-direction-loss-weight 0.3 --per-tf-loss-ramp-epochs 0 \
    2>&1 | grep "INFO.*Epoch\|complete"

echo "=== v1_very_high_weight (V1, weight=4.0, no ramp) ==="
mkdir -p "$BASE_DIR/v1_very_high_weight"
python3 -u -m v15.pipeline train --output "$BASE_DIR/v1_very_high_weight" $COMMON \
    --per-tf-loss-weight 4.0 --per-tf-direction-loss-weight 0.3 --per-tf-loss-ramp-epochs 0 \
    2>&1 | grep "INFO.*Epoch\|complete"

echo "=== ALL DONE ==="
