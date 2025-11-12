#!/bin/bash
#
# Train all 4 timeframe models sequentially with same configuration.
#
# Usage:
#   ./train_all_models.sh
#
# Or customize parameters:
#   EPOCHS=100 BATCH_SIZE=256 ./train_all_models.sh
#

# Configuration (can be overridden via environment variables)
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-128}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-200}
DEVICE=${DEVICE:-cuda}
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-10}

# Timeframes to train
TIMEFRAMES=("15min" "1hour" "4hour" "daily")

echo "======================================================================"
echo "TRAINING ALL MULTI-SCALE MODELS"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Sequence length: $SEQUENCE_LENGTH"
echo "  Device: $DEVICE"
echo "  Pretrain epochs: $PRETRAIN_EPOCHS"
echo ""
echo "Will train ${#TIMEFRAMES[@]} models:"
for tf in "${TIMEFRAMES[@]}"; do
    echo "  - LNN_$tf"
done
echo ""
echo "Estimated time: 60-100 minutes on T4 GPU (sequential)"
echo "======================================================================"
echo ""

# Confirm
read -p "Proceed? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Train each model
for i in "${!TIMEFRAMES[@]}"; do
    tf="${TIMEFRAMES[$i]}"
    num=$((i+1))

    echo ""
    echo "======================================================================"
    echo "TRAINING MODEL $num/${#TIMEFRAMES[@]}: ${tf^^}"
    echo "======================================================================"
    echo ""

    python3 train_model_lazy.py \
        --input_timeframe "$tf" \
        --sequence_length "$SEQUENCE_LENGTH" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --pretrain_epochs "$PRETRAIN_EPOCHS" \
        --device "$DEVICE" \
        --output "models/lnn_$tf.pth"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Model $num/$#TIMEFRAMES[@]} complete: lnn_$tf.pth"
    else
        echo ""
        echo "✗ Error training $tf model"
        read -p "Continue with next model? [y/N]: " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Stopped."
            exit 1
        fi
    fi
done

echo ""
echo "======================================================================"
echo "✅ ALL MODELS TRAINED"
echo "======================================================================"
echo ""
echo "Trained models:"
for tf in "${TIMEFRAMES[@]}"; do
    if [ -f "models/lnn_$tf.pth" ]; then
        echo "  ✓ models/lnn_$tf.pth"
    else
        echo "  ✗ models/lnn_$tf.pth (missing)"
    fi
done

echo ""
echo "======================================================================"
echo "Next steps:"
echo "  1. Backtest all models:"
echo "     python backtest_all_models.py --test_year 2023 --num_simulations 500"
echo ""
echo "  2. Train Meta-LNN coach:"
echo "     python train_meta_lnn.py --mode backtest_no_news"
echo "======================================================================"
echo ""
