# AutoTrade v5.3.1 - Quick Start Guide

**Version**: 5.3.1
**Branch**: `hierarchical-containment`
**Last Updated**: December 10, 2025

---

## 🚀 SETUP

### 1. Clone & Install
```bash
git clone <repo>
cd exp
python -m venv myenv
source myenv/bin/activate  # or `myenv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Required Files
Copy from training machine (if using pre-trained model):
```
models/hierarchical_lnn.pth                    # Trained model (~80MB)
data/feature_cache/tf_meta_*_ev*.json          # Feature metadata with events hash
data/VIX_History.csv                           # Historical VIX data
data/tsla_events_REAL.csv                      # Event data
```

---

## 🎯 TRAINING

### Interactive Mode (Recommended)
```bash
python train_hierarchical.py --interactive
```

### Recommended Settings by Device:

#### **Local Mac (MPS) - Testing Only**
```
Device: MPS
Precision: FP32 (MPS doesn't support TF32)
Batch: 8-16
Workers: 0
Flow: bottom_up
Epochs: 1-5 (testing only)
Time: ~40-60 min/epoch
Use: Quick validation, code testing
```

#### **NVIDIA Consumer GPU (RTX 3060-4090)**
```
Device: CUDA
Precision: FP32 with TF32 ⭐
Batch: 128-512 (depending on VRAM)
Workers: 0-2
Flow: bottom_up
Epochs: 100
Time: ~12-24 hours
Use: Personal training
```

#### **NVIDIA Data Center (A100/H100/H200)**
```
Device: CUDA
Precision: FP32 with TF32 ⭐
Batch: 4096-16384 (H200: use 12288!)
Workers: 16 (WITH --shm-size=64g!)
Flow: bottom_up or bidirectional_bottom
Epochs: 100
Time: ~4-8 hours
Use: Production training

CRITICAL: Add --shm-size=64g to Docker Options on Vast.ai!
```

#### **Multi-GPU (8× RTX 5090, 12× RTX 5060, etc.)**
```
Device: CUDA (auto-selects DDP)
Precision: FP32 with TF32 ⭐
Batch: 256 per GPU
Workers: 0 (or 4 with --shm-size=64g)
Flow: bottom_up
Epochs: 100
Time: ~5-10 hours
Use: Fast parallel training

CRITICAL: Add --shm-size=64g if using workers!
```

---

## ⚠️ CRITICAL SETTINGS (v5.3.1)

### ✅ **MUST DO**:
1. **Learning Rate**: 0.0003 (NOT 0.01 - causes instant NaN!)
2. **Precision**: FP32 with TF32 on CUDA (FP16 AMP causes NaN)
3. **Architecture**: Locked to Geometric + Physics-Only (menu auto-sets)

### ⚠️ **KNOWN ISSUES**:
1. **FP16 AMP**: Causes NaN in duration NLL loss (variance underflow) - DO NOT USE
2. **torch.compile**: Graph breaks on complex architecture - DO NOT USE
3. **LR > 0.001**: Causes gradient explosion with 20M params

### 🔍 **EXPERIMENTAL** (Test in 1 epoch first):
1. **Information Flow**: 4 modes (bottom_up default, try others!)
2. **Batch Size > 8192**: Requires --shm-size=64g on Docker
3. **num_workers > 0**: Minimal benefit (collate bottleneck), needs shm

---

## 💰 GPU RENTAL RECOMMENDATIONS (Vast.ai)

**For 100-Epoch Training**:

| GPU Config | Cost | Time | Best For | Docker Options |
|------------|------|------|----------|----------------|
| **12× RTX 5060** | $8.28 | ~5h | Budget | `--shm-size=64g` |
| **1× H200** | $29-34 | ~17h | Simplicity | `--shm-size=64g` |
| **4× RTX 4090** | $92.55 | ~92h | - | Avoid (slow + expensive) |

**Add to Docker Options**:
```
--shm-size=64g -p 1111:1111 -p 6006:6006 ... (rest of defaults)
```

**Without --shm-size**: Limited to num_workers=0, batch<4096

---

## 📊 INFERENCE (Live Predictions)

### Using the Dashboard
```bash
streamlit run dashboard_v531.py
```

**Workflow**:
1. Load Model (models/hierarchical_lnn.pth)
2. Fetch Live Data (5-10 min first time, cached after)
3. Make Prediction (instant if cached)

**Displays**:
- Primary prediction (high/low with dual confidence)
- Duration scenarios (conservative/expected/aggressive)
- All 11 TF comparison
- Phase 2 forecast (expandable)
- Hierarchical containment (expandable)
- Training history (if available)

### Using Python
```python
from predict import LivePredictor

predictor = LivePredictor('models/hierarchical_lnn.pth')
predictor.fetch_live_data(intraday_days=60, daily_days=400, longer_days=5475)
result = predictor.predict()

print(f"Selected: {result['selected_tf']}")
print(f"High: {result['predicted_high']:.2f}%")
print(f"Low: {result['predicted_low']:.2f}%")
print(f"Validity: {result['v52_validity'][result['selected_tf']]:.0%}")
print(f"Duration: {result['v52_duration'][result['selected_tf']]['expected']:.0f} bars")
```

---

## 📈 EXPECTED PERFORMANCE (After 100 Epochs)

**Test Set**:
- MAE: ~0.20-0.22% (v5.1 baseline was 0.30%)
- RMSE: ~0.40-0.45%

**Duration Prediction**:
- MAE: <5 bars (with parent context)
- Confidence: Calibrated to actual accuracy

**Transition Accuracy**:
- 4-way classification: >70%
- Learns continue/switch/reverse/sideways patterns

---

## 🏗️ ARCHITECTURE SUMMARY

**v5.3.1 Features**:
- 20M parameters
- 11 CfC layers (5min → 3month)
- VIX CfC (90-day regime awareness)
- Event embedding (FOMC, earnings, deliveries)
- Probabilistic duration (mean ± std)
- Validity heads (forward-looking)
- Calibrated confidence (accuracy-based)
- 4-way information flow (bottom/top/bidirectional)
- Hierarchical containment
- Phase 2 compositor (transitions)

**Locked Settings**:
- Base: Geometric projections
- Aggregation: Physics-Only (selects best TF)
- No fusion heads
- No preload mode
- No legacy options

---

## 🐛 TROUBLESHOOTING

**NaN at batch 0**:
- Check: LR ≤ 0.0005
- Check: Precision is FP32 (not FP16)
- Check: Duration log_std clamped to [-3, 3]

**Bus error with workers**:
- Add: --shm-size=64g to Docker Options
- Or: Set num_workers=0

**SLOW_COLLATE warnings**:
- Normal: 1.2-2.5s for 256-8192 samples
- Can't optimize: Collate is serial operation
- Real speedup: TF32 (GPU), not workers (CPU)

**Training very slow**:
- Check: TF32 enabled? (should see "FP32 with TF32" in config summary)
- Check: Batch size appropriate for GPU VRAM
- Check: Not using FP64 by mistake

---

## 📚 DOCUMENTATION

**Technical Spec**: `Technical_Specification_v5.3.md`
**Future Plans**: `docs/v5.3.1_bidirectional_flow.md`, `docs/v5.4_roadmap.md`
**Training Plans**: `/Users/frank/.claude/plans/` (various)
**Deprecated**: `deprecated_code/` directory

---

## 🎯 QUICK REFERENCE

**Best Settings (H200 with --shm-size=64g)**:
```
python train_hierarchical.py --interactive

→ Precision: FP32 with TF32
→ Batch: 12288
→ Workers: 16
→ Flow: bottom_up
→ LR: 0.0003
→ Epochs: 100

Expected: ~4-5 hours, $29 cost, 0.20% test MAE
```

**Files to Copy After Training**:
```
models/hierarchical_lnn.pth
models/hierarchical_training_history.json
data/feature_cache/tf_meta_*_ev*.json (if different from local)
```

**Then Run Dashboard**:
```bash
streamlit run dashboard_v531.py
```

---

**Model Version**: v5.3.1
**Status**: Production Ready
**Performance**: 0.2523% test MAE (1 epoch), expected <0.20% after 100
**Last Updated**: December 10, 2025
