# Physics-Inspired ML Architecture: Wigner Crystal Implementation Plan (physics2)

## Executive Summary

This plan maps Generalized Wigner Crystal (GWC) physics to a hierarchical stock prediction system. The physics describes **multi-scale phase transitions in strongly correlated systems** - directly analogous to **multi-timeframe regime changes in markets**.

---

## Part 1: Physics → ML Concept Mapping

| Physics Concept | Mathematical Form | ML Application |
|----------------|-------------------|----------------|
| **Screened Coulomb** | `V(r) = Σ (-1)^k / √[(kd)² + r²]` | Temporal attention decay with oscillating bias |
| **Hubbard Hamiltonian** | `H = -t·hop + U·self + V·interact` | Multi-component loss function |
| **Order Parameter** | `S(k) = |Σ ⟨n_i⟩ e^{ik·r_i}|` | Fourier-based trend strength |
| **Phase Transition (Tc)** | Specific heat peak | Regime change detection |
| **Pinball Phase** | Localized + delocalized coexistence | Support/resistance + volatility |
| **Charge Frustration** | Competing V1, V2, V3 | Multi-timeframe signal conflict |

---

## Part 2: Architecture Overview

### Current vs Proposed

**Current Architecture**:
```
Input (14,322 features)
    ↓
11 CfC Layers (bottom-up hidden flow)
    ↓
Simple Concatenation (33 + 12 = 45 dims)
    ↓
MLP Fusion (45 → 128 → 64)
    ↓
Multi-task Heads (15+ outputs)
```

**Proposed Physics-Inspired Architecture**:
```
Input (14,322 features)
    ↓
11 CfC Layers + Screened Coulomb Attention (per-layer)
    ↓
Order Parameter Computation (Fourier trend strength)
    ↓
Phase Transition Detection (regime indicators)
    ↓
Cross-Timeframe Coulomb Attention Fusion (V_ij matrix)
    ↓
Hubbard Loss (kinetic + on-site + interaction)
    ↓
Multi-task Heads + Pinball Phase Head + Energy Confidence
```

---

## Part 3: Component Implementations

### Component 1: Screened Coulomb Attention

**Physics Basis**: Double-gate screened Coulomb potential (Equation 2 from paper)
```
V(r) = (e²/4πεε₀a) Σ_{k=-∞}^{∞} (-1)^k / √[(kd/a)² + (|r|/a)²]
```

**Key Properties**:
- Oscillatory decay with distance (alternating signs from (-1)^k)
- Learnable screening length d/a
- Learnable dielectric constant ε

```python
class ScreenedCoulombAttention(nn.Module):
    """
    Attention mechanism inspired by double-gate screened Coulomb potential.

    The oscillating term (-1)^k models how:
    - Recent data: POSITIVE influence (confirms current trend)
    - Slightly older: NEGATIVE influence (creates doubt/pullback)
    - Even older: POSITIVE influence (supports longer-term thesis)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, num_screens: int = 5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.num_screens = num_screens

        # Learnable physics parameters (per head)
        self.d_over_a = nn.Parameter(torch.ones(num_heads) * 2.0)  # Screening length
        self.epsilon = nn.Parameter(torch.ones(num_heads) * 1.0)   # Dielectric

        # Standard QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, time_indices: torch.Tensor = None):
        """
        x: [batch, seq_len, hidden_dim]
        time_indices: [batch, seq_len] or None (uses position indices)
        """
        B, L, D = x.shape

        # QKV projections
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Physics-based positional bias (Coulomb potential)
        if time_indices is None:
            time_indices = torch.arange(L, device=x.device).float()

        # Temporal distance matrix
        r = torch.abs(time_indices.unsqueeze(-1) - time_indices.unsqueeze(-2))

        # Screened Coulomb potential for each head
        coulomb_bias = torch.zeros(self.num_heads, L, L, device=x.device)
        for h in range(self.num_heads):
            d_a = self.d_over_a[h]
            eps = self.epsilon[h]

            for k in range(-self.num_screens, self.num_screens + 1):
                denom = torch.sqrt((k * d_a)**2 + r**2 + 1e-6)
                coulomb_bias[h] += ((-1)**k) / denom

            coulomb_bias[h] = coulomb_bias[h] / eps

        # Add Coulomb bias to attention scores
        scores = scores + coulomb_bias.unsqueeze(0)

        # Softmax and output
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)

        return self.out_proj(out), attn
```

---

### Component 2: Cross-Timeframe Interaction Hierarchy (V₁, V₂, V₃)

**Physics Basis**: Extended Hubbard model with V_ij interactions

**Key Insight from Paper**: Truncating at different ranges gives DIFFERENT ground states:
- V₁ only → Triangular crystal ✓
- V₁ + V₂ → Dimer crystal ✗ (WRONG!)
- V₁ + V₂ + V₃ → Stripe phase ✗ (WRONG!)
- Full LR → Triangular crystal ✓ (CORRECT!)

```python
class CrossTimeframeCoulombAttention(nn.Module):
    """
    Cross-attention between timeframes using physics-inspired V_ij matrix.

    Maps to timeframe hierarchy:
    - V₁ (NN): Adjacent timeframes (5min↔15min) - strength 1.0
    - V₂ (NNN): Skip-one (5min↔30min) - strength ~0.5
    - V₃ (NNNN): Skip-two (5min↔1h) - strength ~0.25
    - V_LR: All pairwise with decay
    """

    def __init__(self, hidden_dim: int, num_timeframes: int = 11):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_timeframes = num_timeframes

        # Initialize V_ij with physics-inspired 1/distance decay
        init_interactions = torch.zeros(num_timeframes, num_timeframes)
        for i in range(num_timeframes):
            for j in range(num_timeframes):
                if i != j:
                    init_interactions[i, j] = 1.0 / (abs(i - j))

        self.V_ij = nn.Parameter(init_interactions)

        # Projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states: dict) -> torch.Tensor:
        """
        hidden_states: dict[timeframe -> tensor [batch, hidden_dim]]
        Returns: fused representation [batch, hidden_dim]
        """
        tfs = list(hidden_states.keys())
        B = hidden_states[tfs[0]].shape[0]

        # Stack hidden states
        H = torch.stack([hidden_states[tf] for tf in tfs], dim=1)  # [B, T, D]

        Q = self.query(H)
        K = self.key(H)
        V_vals = self.value(H)

        # Attention with V_ij physics bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        scores = scores + self.V_ij.unsqueeze(0)  # Add interaction bias

        attn = F.softmax(scores, dim=-1)
        fused = torch.matmul(attn, V_vals)

        # Pool across timeframes
        fused = fused.mean(dim=1)

        return self.output(fused)
```

---

### Component 3: Phase Classification Auxiliary Task

**Physics Basis**: GWC phases (triangular, dimer, stripe, pinball, liquid)

**Market Mapping**:
| Phase | Market Behavior | Physics Analog |
|-------|----------------|----------------|
| Trending | Persistent directional movement | Crystal (ordered) |
| Consolidating | Bouncing between S/R | Liquid (bounded) |
| Choppy | Random with no pattern | Gas (chaos) |
| Mixed | Some TFs trending, others ranging | Pinball phase |
| Crash/Melt-up | Structure breakdown | Phase transition |

```python
class PhaseClassifier(nn.Module):
    """
    Classify market phase based on hidden state configuration.
    Auxiliary task that improves main prediction quality.
    """

    PHASES = ['trending', 'consolidating', 'ranging', 'mixed', 'chaotic']

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, len(self.PHASES))
        )

    def forward(self, fused_hidden: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(fused_hidden)
        return F.softmax(logits, dim=-1)
```

---

### Component 4: Energy-Based Confidence Scoring

**Physics Basis**: Boltzmann distribution P(state) ∝ exp(-E/T)

**Market Mapping**:
- Low energy = stable configuration → High confidence
- High energy = unstable configuration → Low confidence

```python
class EnergyBasedConfidence(nn.Module):
    """
    Compute confidence using physics-inspired energy function.

    Energy = -t × kinetic + U × local + V × interaction

    Low energy (stable) → High confidence
    High energy (unstable) → Low confidence
    """

    def __init__(self, num_timeframes: int = 11):
        super().__init__()
        self.t_kinetic = nn.Parameter(torch.tensor(1.0))
        self.U_local = nn.Parameter(torch.tensor(0.5))
        self.V_interaction = nn.Parameter(torch.randn(num_timeframes, num_timeframes) * 0.1)

    def compute_energy(self, hidden_states: dict) -> torch.Tensor:
        """
        Lower energy = more stable/predictable state.
        """
        all_h = torch.stack([hidden_states[tf] for tf in hidden_states], dim=1)

        # Kinetic: smoothness between timeframes
        kinetic = torch.diff(all_h, dim=1).pow(2).sum(dim=-1).mean(dim=-1)

        # Local: activation magnitude
        local = all_h.pow(2).sum(dim=-1).mean(dim=-1)

        # Interaction: cross-timeframe correlations
        norms = all_h.norm(dim=-1)
        interaction = torch.einsum('bi,ij,bj->b', norms, self.V_interaction, norms)

        energy = -self.t_kinetic * kinetic + self.U_local * local + interaction
        return energy

    def forward(self, hidden_states: dict, temperature: float = 1.0) -> torch.Tensor:
        energy = self.compute_energy(hidden_states)
        confidence = torch.sigmoid(torch.exp(-energy / temperature))
        return confidence, energy
```

---

### Component 5: Hubbard-Inspired Loss Function

**Physics Basis**: Extended Hubbard Hamiltonian
```
H = -Σ t_ij (hopping) + U Σ n↑n↓ (on-site) + Σ V_ij n_i n_j (interaction)
```

```python
class HubbardLoss(nn.Module):
    """
    Three-component loss function inspired by Hubbard Hamiltonian.

    - Kinetic (t): Prediction accuracy
    - On-site (U): Self-consistency within timeframe
    - Interaction (V): Cross-timeframe correlation consistency
    """

    def __init__(self, num_timeframes: int = 11):
        super().__init__()
        self.t_hop = nn.Parameter(torch.tensor(1.0))
        self.U_onsite = nn.Parameter(torch.tensor(0.5))

        # V_ij with 1/distance initialization
        V_init = torch.zeros(num_timeframes, num_timeframes)
        for i in range(num_timeframes):
            for j in range(num_timeframes):
                if i != j:
                    V_init[i, j] = 0.1 / abs(i - j)
        self.V_interaction = nn.Parameter(V_init)

    def forward(self, final_preds, targets, tf_preds, hidden_states):
        losses = {}

        # KINETIC: Primary prediction accuracy
        L_kinetic = F.mse_loss(final_preds['high'], targets['high'])
        L_kinetic += F.mse_loss(final_preds['low'], targets['low'])
        losses['kinetic'] = self.t_hop * L_kinetic

        # ON-SITE: High confidence should mean small prediction spread
        L_onsite = 0
        for tf, preds in tf_preds.items():
            pred_range = torch.abs(preds['high'] - preds['low'])
            L_onsite += torch.mean(preds['conf'] * pred_range)
        losses['onsite'] = self.U_onsite * L_onsite

        # INTERACTION: Similar hidden states → similar predictions
        L_interaction = 0
        tfs = list(hidden_states.keys())
        for i in range(len(tfs)):
            for j in range(i + 1, len(tfs)):
                similarity = F.cosine_similarity(
                    hidden_states[tfs[i]], hidden_states[tfs[j]], dim=-1
                )
                pred_diff = (tf_preds[tfs[i]]['high'] - tf_preds[tfs[j]]['high'])**2
                L_interaction += self.V_interaction[i, j] * torch.mean(similarity * pred_diff)
        losses['interaction'] = L_interaction

        total = losses['kinetic'] + losses['onsite'] + losses['interaction']
        return total, losses
```

---

### Component 6: Order Parameter Feature

**Physics Basis**: Fourier transform of charge distribution
```
S(k) = (1/N) Σ ⟨n_i n_j⟩ e^{ik·(r_i - r_j)}
```

```python
def compute_order_parameter(prices: torch.Tensor, window: int = 64) -> dict:
    """
    Fourier-based order parameter for trend strength detection.

    High S(k) = crystalline order (strong trend)
    Low S(k) = liquid disorder (ranging/noise)
    """
    returns = prices[..., 1:] - prices[..., :-1]

    fft = torch.fft.rfft(returns, dim=-1)
    power = torch.abs(fft)**2

    # Order parameter: concentration of power at dominant frequency
    max_power = power.max(dim=-1).values
    total_power = power.sum(dim=-1) + 1e-10
    order_param = max_power / total_power

    # Dominant frequency (period of oscillation)
    dominant_freq = torch.argmax(power, dim=-1)

    # Spectral entropy (disorder measure)
    power_norm = power / (total_power.unsqueeze(-1) + 1e-10)
    spectral_entropy = -torch.sum(power_norm * torch.log(power_norm + 1e-10), dim=-1)

    return {
        'order_param': order_param,
        'dominant_freq': dominant_freq,
        'spectral_entropy': spectral_entropy
    }
```

---

### Component 7: Pinball Phase Head (Consolidation Detection)

**Physics Basis**: Partially melted GWC with pinned + delocalized charges

```python
class PinballPhaseHead(nn.Module):
    """
    Detect consolidation patterns (pinball phase).

    - "Pins" = strong support/resistance levels
    - "Delocalized" = volatile intermediate zone
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # upper_pin, lower_pin, delocalization, pattern_type
        )

    def forward(self, fused_hidden: torch.Tensor):
        raw = self.head(fused_hidden)

        return {
            'upper_pin_pct': F.softplus(raw[:, 0:1]),      # Distance to resistance
            'lower_pin_pct': -F.softplus(raw[:, 1:2]),     # Distance to support
            'delocalization': torch.sigmoid(raw[:, 2:3]),  # Volatility in zone
            'pattern_type': raw[:, 3:4]                     # Pattern classification
        }
```

---

## Part 4: Auto-Generated Consolidation Labels

```python
class ConsolidationLabeler:
    """
    Auto-detect consolidation patterns for pinball phase training labels.
    """

    def label_dataset(self, df: pd.DataFrame, min_touches: int = 3) -> pd.DataFrame:
        labels = pd.DataFrame(index=df.index)

        for window in [50, 100, 200]:
            rolling_high = df['high'].rolling(window).max()
            rolling_low = df['low'].rolling(window).min()

            # Count boundary touches
            upper_touch = (df['high'] >= rolling_high * 0.99).rolling(window).sum()
            lower_touch = (df['low'] <= rolling_low * 1.01).rolling(window).sum()

            # Consolidation = multiple touches on both sides
            is_consolidating = (upper_touch >= min_touches) & (lower_touch >= min_touches)

            # Delocalization = normalized volatility within range
            range_size = rolling_high - rolling_low
            internal_vol = df['close'].rolling(window).std()
            delocalization = internal_vol / (range_size + 1e-10)

            labels[f'is_consolidating_{window}'] = is_consolidating.astype(float)
            labels[f'upper_pin_{window}'] = (rolling_high - df['close']) / df['close'] * 100
            labels[f'lower_pin_{window}'] = (df['close'] - rolling_low) / df['close'] * 100
            labels[f'delocalization_{window}'] = delocalization.clip(0, 1)

        return labels
```

---

## Part 5: Main Model (WignerLNN)

```python
class WignerLNN(nn.Module):
    """
    Physics-inspired Hierarchical Liquid Neural Network.

    Full architecture with all physics components:
    - Screened Coulomb attention per layer
    - Cross-timeframe V_ij fusion
    - Phase classification
    - Energy-based confidence
    - Pinball phase detection
    """

    TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h',
                  'daily', 'weekly', 'monthly', '3month']

    def __init__(self, input_sizes: dict, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = hidden_size

        # CfC layers with Coulomb attention
        self.timeframe_layers = nn.ModuleDict()
        self.timeframe_attention = nn.ModuleDict()

        for i, tf in enumerate(self.TIMEFRAMES):
            tf_input = input_sizes.get(tf, 128)
            if i > 0:
                tf_input += hidden_size

            wiring = AutoNCP(hidden_size * 2, hidden_size)
            self.timeframe_layers[tf] = CfC(tf_input, wiring, batch_first=True)
            self.timeframe_attention[tf] = ScreenedCoulombAttention(hidden_size)

        # Per-layer prediction heads
        self.timeframe_heads = nn.ModuleDict()
        for tf in self.TIMEFRAMES:
            self.timeframe_heads[f'{tf}_high'] = nn.Linear(hidden_size, 1)
            self.timeframe_heads[f'{tf}_low'] = nn.Linear(hidden_size, 1)
            self.timeframe_heads[f'{tf}_conf'] = nn.Linear(hidden_size, 1)

        # Cross-timeframe fusion
        self.cross_tf_attention = CrossTimeframeCoulombAttention(hidden_size, len(self.TIMEFRAMES))

        # Physics components
        self.phase_classifier = PhaseClassifier(hidden_size)
        self.energy_confidence = EnergyBasedConfidence(len(self.TIMEFRAMES))
        self.pinball_head = PinballPhaseHead(hidden_size)

        # Final prediction heads
        self.final_high = nn.Linear(hidden_size, 1)
        self.final_low = nn.Linear(hidden_size, 1)
        self.final_conf = nn.Linear(hidden_size, 1)

        # Multi-task heads (same as original)
        # ... (hit_band, hit_target, breakout, continuation, etc.)

    def forward(self, timeframe_data: dict, market_state=None):
        hidden_states = {}
        timeframe_predictions = {}
        prev_hidden = None

        # Process each timeframe
        for i, tf in enumerate(self.TIMEFRAMES):
            x_tf = timeframe_data.get(tf)
            if x_tf is None:
                continue

            B, L, D = x_tf.shape

            # Concatenate previous hidden
            if i > 0 and prev_hidden is not None:
                prev_expanded = prev_hidden.unsqueeze(1).expand(-1, L, -1)
                x_tf = torch.cat([x_tf, prev_expanded], dim=-1)

            # CfC forward
            h_init = torch.zeros(B, self.hidden_size, device=x_tf.device)
            layer_out, h_new = self.timeframe_layers[tf](x_tf, h_init)

            # Coulomb attention within sequence
            attended, attn = self.timeframe_attention[tf](layer_out)

            hidden = attended[:, -1, :]
            hidden_states[tf] = hidden
            prev_hidden = hidden

            # Per-layer predictions
            timeframe_predictions[tf] = {
                'high': self.timeframe_heads[f'{tf}_high'](hidden),
                'low': self.timeframe_heads[f'{tf}_low'](hidden),
                'conf': torch.sigmoid(self.timeframe_heads[f'{tf}_conf'](hidden))
            }

        # Cross-timeframe Coulomb fusion
        fused = self.cross_tf_attention(hidden_states)

        # Physics outputs
        phase_probs = self.phase_classifier(fused)
        energy_conf, energy = self.energy_confidence(hidden_states)
        pinball = self.pinball_head(fused)

        # Final predictions
        final_high = self.final_high(fused)
        final_low = self.final_low(fused)
        final_conf = torch.sigmoid(self.final_conf(fused))

        predictions = torch.cat([final_high, final_low, final_conf], dim=-1)

        output_dict = {
            'hidden_states': hidden_states,
            'timeframe_predictions': timeframe_predictions,
            'phase': phase_probs,
            'energy_confidence': energy_conf,
            'energy': energy,
            'pinball': pinball,
            'fused_hidden': fused,
        }

        return predictions, output_dict
```

---

## Part 6: Implementation Plan

### File Structure
```
src/ml/
├── physics_model.py          # WignerLNN main model
├── coulomb_attention.py      # Screened Coulomb attention
├── hubbard_loss.py           # Physics-inspired loss
├── order_parameter.py        # Fourier-based features
├── phase_classifier.py       # Phase detection
├── energy_confidence.py      # Energy-based confidence
├── pinball_head.py           # Consolidation detection
├── consolidation_labels.py   # Auto-label generation
```

### Implementation Order

| Step | Component | Dependencies |
|------|-----------|--------------|
| 1 | `order_parameter.py` | None |
| 2 | `phase_classifier.py` | None |
| 3 | `energy_confidence.py` | None |
| 4 | `consolidation_labels.py` | None |
| 5 | `coulomb_attention.py` | None |
| 6 | `hubbard_loss.py` | None |
| 7 | `pinball_head.py` | None |
| 8 | `physics_model.py` | Steps 1-7 |
| 9 | Training script updates | Step 8 |
| 10 | Dataset updates | Step 4 |

---

## Part 7: Training Changes

```python
# In train_wigner.py

from src.ml.physics_model import WignerLNN
from src.ml.hubbard_loss import HubbardLoss

# Initialize model
model = WignerLNN(input_sizes, hidden_size=128)

# Initialize physics loss
hubbard_loss = HubbardLoss(num_timeframes=11)

# Training loop
for batch in dataloader:
    predictions, output_dict = model(batch['features'], batch['market_state'])

    # Hubbard loss
    loss, loss_components = hubbard_loss(
        final_preds={'high': predictions[:, 0], 'low': predictions[:, 1]},
        targets=batch['targets'],
        tf_preds=output_dict['timeframe_predictions'],
        hidden_states=output_dict['hidden_states']
    )

    # Additional losses for auxiliary tasks
    phase_loss = F.cross_entropy(output_dict['phase'], batch['phase_labels'])
    pinball_loss = pinball_criterion(output_dict['pinball'], batch['consolidation_labels'])

    total_loss = loss + 0.1 * phase_loss + 0.1 * pinball_loss

    total_loss.backward()
    optimizer.step()
```

---

## Summary

This plan implements 7 physics-inspired components:

1. **Screened Coulomb Attention** - Oscillatory temporal attention
2. **Cross-Timeframe V_ij Interaction** - Physics-based timeframe fusion
3. **Phase Classification** - Market regime detection
4. **Energy-Based Confidence** - Boltzmann-inspired uncertainty
5. **Hubbard Loss** - Three-component physics loss
6. **Order Parameter** - Fourier trend strength
7. **Pinball Phase Head** - Consolidation pattern detection

This is a **full rebuild** approach that replaces the current HierarchicalLNN with a physics-inspired WignerLNN.
