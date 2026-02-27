# Hierarchical CfC Neural Network Architecture

## Overview

The Hierarchical CfC (Closed-form Continuous-time) model is a multi-timeframe channel prediction system that uses Liquid Neural Networks to predict:
1. **Duration** until channel break (probabilistic)
2. **Break direction** (up/down)
3. **Next channel direction** (bear/sideways/bull)
4. **Confidence** (calibrated probability)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT FEATURES (582 dims)                           │
│                                                                             │
│  Per-Timeframe Features (517 dims):                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ TSLA: 28 × 11 TFs = 308 dims                                          │ │
│  │   ├─ Channel geometry (7): direction, position, width, slope, R²      │ │
│  │   ├─ Bounce metrics (4): count, cycles, bars_since, last_touch        │ │
│  │   ├─ RSI (5): current, divergence, upper/lower bounce values          │ │
│  │   ├─ Exit tracking (10): counts, frequency, acceleration              │ │
│  │   └─ Break triggers (2): nearest boundary, RSI alignment              │ │
│  │                                                                         │ │
│  │ SPY: 11 × 11 TFs = 121 dims                                            │ │
│  │   └─ Channel geometry, position, bounce metrics, RSI                  │ │
│  │                                                                         │ │
│  │ Cross-asset: 8 × 11 TFs = 88 dims                                      │ │
│  │   └─ TSLA position in SPY channels, alignment scores                  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  Shared Features (65 dims):                                                │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │ VIX: 6 dims (level, normalized, trends, percentile, regime)           │ │
│  │ TSLA History: 28 dims (past directions, durations, break patterns)    │ │
│  │ SPY History: 28 dims (same as TSLA)                                   │ │
│  │ Alignment: 3 dims (direction match, both near upper/lower)            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEATURE DECOMPOSITION & DISTRIBUTION                     │
│                                                                             │
│  Extract per-TF features for each timeframe + shared context               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │     11 Parallel TF Branches       │
                    └─────────────────┬─────────────────┘
                                      │
        ┌─────────┬─────────┬────────┼────────┬─────────┬─────────┐
        ▼         ▼         ▼        ▼        ▼         ▼         ▼
    ┌──────┐ ┌──────┐ ┌──────┐  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
    │ 5min │ │15min │ │30min │  │  1h  │ │  2h  │ │  ... │ │3month│
    │Branch│ │Branch│ │Branch│  │Branch│ │Branch│ │Branch│ │Branch│
    └───┬──┘ └───┬──┘ └───┬──┘  └───┬──┘ └───┬──┘ └───┬──┘ └───┬──┘
        │        │        │         │        │        │        │
        │   Each Branch Architecture:                           │
        │   ┌─────────────────────────────────────────────┐     │
        │   │ Per-TF features (47) + Shared (65) = 112    │     │
        │   │              ▼                              │     │
        │   │      Linear Projection (→ 64)               │     │
        │   │              ▼                              │     │
        │   │         LayerNorm + GELU                    │     │
        │   │              ▼                              │     │
        │   │    CfC (Liquid Neural Network)              │     │
        │   │      96 units → 64 output                   │     │
        │   │    (Sparse recurrent architecture)          │     │
        │   │              ▼                              │     │
        │   │      LayerNorm + Dropout                    │     │
        │   │              ▼                              │     │
        │   │        Embedding (64 dims)                  │     │
        │   └─────────────────────────────────────────────┘     │
        │        │        │         │        │        │        │
        ▼        ▼        ▼         ▼        ▼        ▼        ▼
    ┌────────────────────────────────────────────────────────────┐
    │            TF Embeddings [batch, 11, 64]                   │
    └────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │           CROSS-TIMEFRAME ATTENTION                         │
    │                                                             │
    │  Multi-head Attention (4 heads)                             │
    │  ┌───────────────────────────────────────────────────────┐ │
    │  │ Query, Key, Value: TF Embeddings                      │ │
    │  │ Learns which TFs matter for current prediction        │ │
    │  │                                                        │ │
    │  │ Example attention patterns:                           │ │
    │  │   - Predicting 5min break: Attend to daily/weekly     │ │
    │  │   - High volatility: Attend to shorter TFs            │ │
    │  │   - Trend change: Attend to longer TFs                │ │
    │  └───────────────────────────────────────────────────────┘ │
    │                          ▼                                  │
    │         Mean Pool + Linear Projection                       │
    │                          ▼                                  │
    │           Context Vector (128 dims)                         │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                   PREDICTION HEADS (Shared)                 │
    │                                                             │
    │  ┌─────────────────┐  ┌──────────────────┐                 │
    │  │ Duration Head   │  │ Break Dir Head   │                 │
    │  │ (Gaussian NLL)  │  │ (Binary CE)      │                 │
    │  ├─────────────────┤  ├──────────────────┤                 │
    │  │ 128→64→32       │  │ 128→64→32        │                 │
    │  │   ↓             │  │   ↓              │                 │
    │  │ mean & std      │  │ [down, up]       │                 │
    │  └─────────────────┘  └──────────────────┘                 │
    │                                                             │
    │  ┌─────────────────┐  ┌──────────────────┐                 │
    │  │ Next Dir Head   │  │ Confidence Head  │                 │
    │  │ (3-class CE)    │  │ (Calibration)    │                 │
    │  ├─────────────────┤  ├──────────────────┤                 │
    │  │ 128→64→32       │  │ 128→64→32        │                 │
    │  │   ↓             │  │   ↓              │                 │
    │  │[bear,side,bull] │  │ sigmoid → [0,1]  │                 │
    │  └─────────────────┘  └──────────────────┘                 │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                         OUTPUTS                             │
    │                                                             │
    │  • duration_mean:     Expected bars until break             │
    │  • duration_std:      Uncertainty in duration               │
    │  • break_direction:   0=down, 1=up                          │
    │  • next_direction:    0=bear, 1=sideways, 2=bull            │
    │  • confidence:        Calibrated probability [0, 1]         │
    │  • attention_weights: Which TFs were important [11×11]      │
    └─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. TF Branch (Liquid Neural Network)

Each timeframe has a dedicated CfC branch that processes temporal dynamics:

**Why CfC?**
- Models continuous-time dynamics (natural for financial data)
- Strong extrapolation beyond training distribution
- Learns causal relationships automatically
- Compact parameters vs LSTM/GRU

**Architecture:**
```python
Input (112 dims) → Linear(64) → LayerNorm → GELU
                → CfC(96 units → 64 output)
                → LayerNorm → Dropout
                → Embedding (64 dims)
```

**Parameters per branch:** ~35,584
**Total for 11 branches:** 391,424 parameters

### 2. Cross-Timeframe Attention

**Purpose:** Learn context-dependent importance of each timeframe

**Mechanism:**
- Multi-head self-attention (4 heads) over 11 TF embeddings
- Each head attends to different TF relationships
- Mean pooling + projection to 128 dims

**Learned patterns (examples):**
- Breaking 5min channel near daily boundary → Attend to daily
- High VIX regime → Attend to shorter TFs (more volatility)
- Trend reversal → Attend to longer TFs (macro context)

**Parameters:** 25,216

### 3. Prediction Heads

All heads are **shared** across timeframes (same weights for all TFs):

#### a) Duration Head (Gaussian NLL)
Predicts distribution over duration:
- **mean**: Expected bars until break
- **std**: Uncertainty in prediction

**Loss:** Negative log-likelihood of Gaussian
```
NLL = 0.5 * (log(2π·σ²) + ((y - μ) / σ)²)
```

**Parameters:** 10,530

#### b) Break Direction Head (Binary Classification)
Predicts which way channel will break:
- Class 0: Down (below lower bound)
- Class 1: Up (above upper bound)

**Loss:** Binary cross-entropy

**Parameters:** 10,530

#### c) Next Channel Direction Head (3-Class)
Predicts the next channel's trend after break:
- Class 0: Bear (downward sloping)
- Class 1: Sideways (neutral)
- Class 2: Bull (upward sloping)

**Loss:** Categorical cross-entropy

**Parameters:** 10,563

#### d) Confidence Head (Calibration)
Outputs calibrated confidence score [0, 1]:
- 0.8 = model is correct ~80% of the time
- 0.5 = model is uncertain

**Loss:** Brier score (MSE between confidence and correctness)
```
Brier = mean((confidence - correct)²)
```

**Parameters:** 10,497

## Model Statistics

| Component            | Parameters | % of Total |
|---------------------|-----------|-----------|
| TF Branches (11×)   | 391,424   | 85.3%     |
| Cross-TF Attention  | 25,216    | 5.5%      |
| Duration Head       | 10,530    | 2.3%      |
| Break Direction     | 10,530    | 2.3%      |
| Next Direction      | 10,563    | 2.3%      |
| Confidence Head     | 10,497    | 2.3%      |
| **Total**           | **458,760** | **100%**  |

## Training Strategy

### Loss Function
```python
Total Loss = w₁·Duration_NLL
           + w₂·BreakDir_CE
           + w₃·NextDir_CE
           + w₄·Confidence_Brier

Default weights: w₁=1.0, w₂=1.0, w₃=1.0, w₄=0.5
```

### Optimization
- **Optimizer:** AdamW (weight decay for regularization)
- **Learning Rate:** 1e-3 → 1e-5 (cosine annealing)
- **Batch Size:** 32-128 (depending on GPU memory)
- **Gradient Clipping:** 1.0 (prevent exploding gradients in CfC)

### Data Pipeline
1. Extract 582-dim feature vector from `FullFeatures`
2. Generate labels from `ChannelLabels`:
   - duration_bars → target for duration head
   - break_direction → target for break dir head
   - new_channel_direction → target for next dir head
   - Compute correctness for confidence calibration
3. Normalize features (standardize per-feature)
4. Shuffle and batch

## Key Design Decisions

### 1. Hierarchical Processing
**Why?** Different timeframes have different temporal dynamics:
- 5min: High frequency, noisy
- Daily: Smoother, trend-following
- Monthly: Long-term structure

Each TF gets its own CfC to learn appropriate temporal patterns.

### 2. Attention Fusion
**Why?** Timeframe importance is context-dependent:
- Near longer TF boundary → That TF matters most
- High volatility → Shorter TFs matter
- Low volatility → Longer TFs matter

Attention learns these relationships from data.

### 3. Shared Prediction Heads
**Why?** Transfer learning within the model:
- Same prediction logic across all TFs
- Reduces parameters (vs separate heads per TF)
- Enforces consistent decision-making

### 4. Probabilistic Duration
**Why?** Point estimates are overconfident:
- Gaussian distribution captures uncertainty
- Wide std = model is unsure
- Narrow std = confident prediction
- Can sample from distribution for Monte Carlo

### 5. Confidence Calibration
**Why?** Know when to trust predictions:
- High confidence + correct → Actionable signal
- Low confidence → Skip trade
- Enables risk-aware decision making

## Usage Example

```python
import torch
from v7.models import create_model, create_loss
from v7.features.full_features import extract_full_features, features_to_tensor_dict

# Create model
model = create_model(
    hidden_dim=64,
    cfc_units=96,
    num_attention_heads=4,
    dropout=0.1,
    device='cuda'
)

# Extract features
features = extract_full_features(tsla_df, spy_df, vix_df)
feature_dict = features_to_tensor_dict(features)

# Flatten to single vector
x = torch.cat([
    torch.from_numpy(feature_dict[f'tsla_{tf}']).flatten(),
    torch.from_numpy(feature_dict[f'spy_{tf}']).flatten(),
    torch.from_numpy(feature_dict[f'cross_{tf}']).flatten(),
    # ... for all TFs and shared features
])
x = x.unsqueeze(0)  # Add batch dimension

# Predict
predictions = model.predict(x)

print(f"Expected duration: {predictions['duration_mean'].item():.1f} bars")
print(f"Uncertainty: ±{predictions['duration_std'].item():.1f} bars")
print(f"Break direction: {'UP' if predictions['break_direction'].item() == 1 else 'DOWN'}")
print(f"Next channel: {['BEAR', 'SIDEWAYS', 'BULL'][predictions['next_direction'].item()]}")
print(f"Confidence: {predictions['confidence'].item():.2%}")

# Attention analysis
attn = predictions['attention_weights'][0]  # [11, 11]
print("\nMost important timeframes:")
importance = attn.mean(dim=0)  # Average attention received
for i, tf in enumerate(['5min', '15min', '30min', '1h', '2h', '3h', '4h',
                         'daily', 'weekly', 'monthly', '3month']):
    print(f"  {tf}: {importance[i].item():.2%}")
```

## Extensions & Future Work

### 1. Multi-Asset Support
Currently supports TSLA/SPY. Could extend to:
- Multiple stocks simultaneously
- Stock-specific branches with cross-stock attention
- Sector rotation signals

### 2. Temporal Sequences
Current model processes single timesteps. Could extend to:
- Sequence input (last N bars)
- Maintain CfC hidden states across time
- Learn temporal patterns within TFs

### 3. Uncertainty Estimation
Add Bayesian layers for epistemic uncertainty:
- Monte Carlo Dropout at inference
- Ensemble of models
- Separate aleatory vs epistemic uncertainty

### 4. Online Learning
Add capabilities for continuous learning:
- Store recent prediction errors
- Fine-tune on recent data
- Adaptive confidence based on recent performance

### 5. Interpretability
Enhance explainability:
- Feature importance (which features drive predictions)
- Counterfactual analysis (what-if scenarios)
- Rule extraction from CfC (symbolic regression)

## References

- **Liquid Neural Networks:** Hasani et al., "Liquid Time-Constant Networks" (NeurIPS 2020)
- **CfC:** Hasani et al., "Closed-form Continuous-time Neural Networks" (Nature Machine Intelligence 2022)
- **Attention:** Vaswani et al., "Attention is All You Need" (NeurIPS 2017)
- **Calibration:** Guo et al., "On Calibration of Modern Neural Networks" (ICML 2017)
