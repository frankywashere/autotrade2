"""
Physics-Inspired Attention Modules for HierarchicalLNN

Based on Generalized Wigner Crystal (GWC) paper concepts:
- Screened Coulomb potential for distance-based attention decay
- Phase classification for market regime detection
- V₁, V₂, V₃ interaction hierarchy
- Energy-based confidence scoring

Key insight: These modules learn WHEN each timeframe should matter more,
not static importance weights. The attention is dynamic and context-dependent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import math


class CoulombTimeframeAttention(nn.Module):
    """
    Dynamic attention weights between timeframes inspired by screened Coulomb potential.

    Key insight: Learns WHEN each timeframe should matter more, not static importance.

    V(r) = Σ (-1)^k / √[(kd)² + r²]

    - Adjacent timeframes have strong base interaction
    - Distant timeframes have weaker but non-zero interaction
    - Dynamic component: attention weights computed from hidden state content
    """

    TIMEFRAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']

    def __init__(self, hidden_size: int, n_timeframes: int = 11, n_screens: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_timeframes = n_timeframes

        # Learnable screening distances (d in the paper's equation)
        # These control how quickly attention decays with timeframe distance
        self.screen_distances = nn.Parameter(torch.tensor([1.0, 3.0, 7.0]))

        # Learnable overall scale (like dielectric constant ε)
        self.scale = nn.Parameter(torch.tensor(1.0))

        # Query/Key projections for dynamic attention
        # This is the key difference from static weights:
        # attention depends on WHAT the timeframes are seeing, not just WHICH timeframe
        self.W_q = nn.Linear(hidden_size, hidden_size // 4)
        self.W_k = nn.Linear(hidden_size, hidden_size // 4)
        self.W_v = nn.Linear(hidden_size, hidden_size)

        # Output projection
        self.W_o = nn.Linear(hidden_size, hidden_size)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Pre-compute static distance matrix
        self.register_buffer('distance_matrix', self._compute_distance_matrix())

    def _compute_distance_matrix(self) -> torch.Tensor:
        """Compute pairwise timeframe distances."""
        n = self.n_timeframes
        distances = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                distances[i, j] = abs(i - j)
        return distances

    def _coulomb_kernel(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Compute Coulomb-inspired bias from distances.

        Implements: V(r) = Σ_k (-1)^k / √[(kd)² + r²]

        This creates oscillating, decaying weights:
        - Strong for adjacent timeframes
        - Weaker but non-zero for distant timeframes
        - Alternating sign captures "screening" effects
        """
        weights = torch.zeros_like(distances)

        for k, d in enumerate(self.screen_distances):
            sign = (-1) ** k
            denominator = torch.sqrt(d**2 + distances**2 + 1e-6)
            weights = weights + sign / denominator

        # Scale and shift to reasonable range
        weights = weights / (self.scale + 1e-6)
        return weights

    def forward(
        self,
        hidden_states: Dict[str, torch.Tensor],
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Coulomb-modulated attention to hidden states.

        Args:
            hidden_states: Dict[timeframe -> tensor of shape [batch, hidden_size]]
            return_attention: If True, also return attention weights for visualization

        Returns:
            attended_states: Dict[timeframe -> attended tensor]
        """
        tf_list = list(hidden_states.keys())
        n_tf = len(tf_list)

        # Stack hidden states: [batch, n_tf, hidden]
        h_stack = torch.stack([hidden_states[tf] for tf in tf_list], dim=1)
        batch_size = h_stack.shape[0]

        # Compute Q, K, V
        Q = self.W_q(h_stack)  # [batch, n_tf, hidden/4]
        K = self.W_k(h_stack)  # [batch, n_tf, hidden/4]
        V = self.W_v(h_stack)  # [batch, n_tf, hidden]

        # Dot-product attention scores
        d_k = Q.shape[-1]
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)  # [batch, n_tf, n_tf]

        # Add Coulomb bias - this is the physics-inspired part
        # The bias encodes prior knowledge about timeframe relationships
        coulomb_bias = self._coulomb_kernel(self.distance_matrix[:n_tf, :n_tf])  # [n_tf, n_tf]
        attn_scores = attn_scores + coulomb_bias.unsqueeze(0)

        # Softmax over source timeframes
        attn_probs = F.softmax(attn_scores, dim=-1)

        # Apply attention
        attended = torch.bmm(attn_probs, V)  # [batch, n_tf, hidden]

        # Output projection + residual + layer norm
        output = self.W_o(attended)
        output = self.layer_norm(h_stack + output)

        # Convert back to dict
        result = {tf: output[:, i, :] for i, tf in enumerate(tf_list)}

        if return_attention:
            result['_attention_weights'] = attn_probs
            result['_coulomb_bias'] = coulomb_bias

        return result


class MarketPhaseClassifier(nn.Module):
    """
    Explicit market phase classification inspired by GWC phase diagram.

    Phases:
    - 0: TRENDING_UP (crystal-like order, positive momentum)
    - 1: TRENDING_DOWN (crystal-like order, negative momentum)
    - 2: CONSOLIDATING (liquid-like, bounded oscillation)
    - 3: VOLATILE/CHOPPY (gas-like, high entropy)
    - 4: TRANSITIONING (phase transition in progress)
    """

    PHASE_NAMES = ['trending_up', 'trending_down', 'consolidating', 'volatile', 'transitioning']
    NUM_PHASES = 5

    def __init__(self, hidden_size: int, n_timeframes: int = 11):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_timeframes = n_timeframes

        # Input: concatenated hidden states from all timeframes
        input_dim = hidden_size * n_timeframes

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.NUM_PHASES)
        )

        # Per-phase temperature for confidence calibration
        self.phase_temp = nn.Parameter(torch.ones(self.NUM_PHASES))

    def forward(self, hidden_states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Classify market phase from all timeframe hidden states.

        Args:
            hidden_states: Dict[timeframe -> tensor [batch, hidden_size]]

        Returns:
            phase_logits: [batch, 5] raw logits
            phase_probs: [batch, 5] calibrated probabilities
            phase_id: [batch] predicted phase index
            phase_entropy: [batch] entropy of phase distribution (uncertainty)
        """
        # Concatenate all hidden states
        tf_list = list(hidden_states.keys())
        h_concat = torch.cat([hidden_states[tf] for tf in tf_list], dim=-1)

        logits = self.classifier(h_concat)

        # Temperature-scaled softmax for calibration
        calibrated_logits = logits / (self.phase_temp.unsqueeze(0) + 1e-6)
        probs = F.softmax(calibrated_logits, dim=-1)

        phase_id = torch.argmax(probs, dim=-1)

        # Entropy as uncertainty measure
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        return {
            'phase_logits': logits,
            'phase_probs': probs,
            'phase_id': phase_id,
            'phase_entropy': entropy,
            'phase_names': self.PHASE_NAMES
        }


class TimeframeInteractionHierarchy(nn.Module):
    """
    Explicit V₁, V₂, V₃ interaction strengths between timeframes.

    Paper insight: Truncating long-range interactions gives WRONG ground states!
    We must model ALL interactions, not just nearest neighbors.

    V₁ = nearest neighbor (5min ↔ 15min): Strong
    V₂ = next-nearest (5min ↔ 30min): Medium
    V₃ = skip-2 (5min ↔ 1h): Weaker
    V_LR = long-range tail: Exponential decay but non-zero
    """

    def __init__(self, hidden_size: int, n_timeframes: int = 11):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_timeframes = n_timeframes

        # Learnable interaction strengths (initialized with physics-inspired values)
        self.V1 = nn.Parameter(torch.tensor(1.0))    # Nearest neighbor
        self.V2 = nn.Parameter(torch.tensor(0.5))    # Next-nearest
        self.V3 = nn.Parameter(torch.tensor(0.25))   # Skip-2
        self.V_lr_decay = nn.Parameter(torch.tensor(0.3))  # Long-range decay rate

        # Bilinear interaction transforms for each distance level
        self.bilinear_v1 = nn.Bilinear(hidden_size, hidden_size, hidden_size)
        self.bilinear_v2 = nn.Bilinear(hidden_size, hidden_size, hidden_size)
        self.bilinear_v3 = nn.Bilinear(hidden_size, hidden_size, hidden_size)

        # For long-range: simpler linear combination
        self.lr_combine = nn.Linear(hidden_size * 2, hidden_size)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(hidden_size)

    def _get_interaction_strength(self, distance: int) -> torch.Tensor:
        """Get V_ij based on timeframe distance."""
        if distance == 0:
            return torch.tensor(0.0, device=self.V1.device)
        elif distance == 1:
            return self.V1
        elif distance == 2:
            return self.V2
        elif distance == 3:
            return self.V3
        else:
            # Exponential decay for long-range
            return self.V3 * torch.exp(-self.V_lr_decay * (distance - 3))

    def forward(self, hidden_states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply hierarchical interactions between timeframes.

        Each timeframe's representation is updated based on its
        interactions with ALL other timeframes, weighted by V_ij.
        """
        tf_list = list(hidden_states.keys())
        n = len(tf_list)

        output_states = {}

        for i, tf_i in enumerate(tf_list):
            h_i = hidden_states[tf_i]
            interaction_sum = torch.zeros_like(h_i)

            for j, tf_j in enumerate(tf_list):
                if i == j:
                    continue

                h_j = hidden_states[tf_j]
                distance = abs(i - j)

                # Get interaction strength
                V_ij = self._get_interaction_strength(distance)

                # Compute interaction contribution based on distance
                if distance == 1:
                    contrib = self.bilinear_v1(h_i, h_j)
                elif distance == 2:
                    contrib = self.bilinear_v2(h_i, h_j)
                elif distance == 3:
                    contrib = self.bilinear_v3(h_i, h_j)
                else:
                    # Long-range: linear combination
                    contrib = self.lr_combine(torch.cat([h_i, h_j], dim=-1))

                interaction_sum = interaction_sum + V_ij * contrib

            # Residual connection + layer norm
            output_states[tf_i] = self.layer_norm(h_i + interaction_sum)

        return output_states


class EnergyBasedConfidence(nn.Module):
    """
    Compute "energy" of market configuration for confidence scoring.

    Physics analogy: Lower energy = more stable = higher confidence

    E = -t × kinetic + U × local + Σ V_ij × interaction

    Market translation:
    - Kinetic: momentum coherence (aligned TFs = lower energy)
    - Local: volatility clustering (high vol = higher energy)
    - Interaction: timeframe disagreement (disagreement = higher energy)
    """

    def __init__(self, hidden_size: int, n_timeframes: int = 11):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_timeframes = n_timeframes

        # Learnable energy weights (like t, U in Hamiltonian)
        self.kinetic_weight = nn.Parameter(torch.tensor(1.0))
        self.local_weight = nn.Parameter(torch.tensor(0.5))
        self.interaction_weight = nn.Parameter(torch.tensor(0.3))

        # Learnable "temperature" for Boltzmann confidence
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Energy estimation networks
        self.kinetic_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.local_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Pairwise disagreement scorer
        self.disagreement_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def compute_energy(self, hidden_states: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute configuration energy from hidden states.

        Returns dict with:
            total_energy: [batch] total energy (lower = more stable)
            kinetic_energy: [batch] momentum coherence term
            local_energy: [batch] local volatility term
            interaction_energy: [batch] disagreement term
        """
        tf_list = list(hidden_states.keys())
        h_stack = torch.stack([hidden_states[tf] for tf in tf_list], dim=1)  # [batch, n_tf, hidden]
        batch_size = h_stack.shape[0]
        n_tf = h_stack.shape[1]

        # Kinetic term: momentum/trend coherence
        # Strong aligned momentum = negative = lower energy
        kinetic_per_tf = self.kinetic_net(h_stack).squeeze(-1)  # [batch, n_tf]
        kinetic_energy = -self.kinetic_weight * kinetic_per_tf.mean(dim=1)

        # Local term: volatility/uncertainty at each timeframe
        local_per_tf = self.local_net(h_stack).squeeze(-1)  # [batch, n_tf]
        local_energy = self.local_weight * F.relu(local_per_tf).mean(dim=1)

        # Interaction term: pairwise disagreement
        interaction_energy = torch.zeros(batch_size, device=h_stack.device)
        n_pairs = 0
        for i in range(n_tf):
            for j in range(i + 1, n_tf):
                h_pair = torch.cat([h_stack[:, i, :], h_stack[:, j, :]], dim=-1)
                disagreement = F.relu(self.disagreement_net(h_pair).squeeze(-1))

                # Weight by distance (adjacent TFs disagreeing is worse)
                distance_weight = 1.0 / (abs(i - j) + 1)
                interaction_energy = interaction_energy + distance_weight * disagreement
                n_pairs += 1

        interaction_energy = self.interaction_weight * interaction_energy / max(n_pairs, 1)

        total_energy = kinetic_energy + local_energy + interaction_energy

        return {
            'total_energy': total_energy,
            'kinetic_energy': kinetic_energy,
            'local_energy': local_energy,
            'interaction_energy': interaction_energy
        }

    def forward(
        self,
        hidden_states: Dict[str, torch.Tensor],
        base_confidence: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute energy-based confidence modulation.

        Returns:
            energy: Raw energy value
            energy_confidence: Boltzmann confidence exp(-E/T)
            adjusted_confidence: base_confidence * energy_confidence (if provided)
        """
        energy_dict = self.compute_energy(hidden_states)
        energy = energy_dict['total_energy']

        # Boltzmann-style confidence: low energy = high confidence
        # P ∝ exp(-E/T)
        energy_confidence = torch.sigmoid(-energy / (torch.abs(self.temperature) + 0.1))

        result = {
            'energy': energy,
            'energy_breakdown': energy_dict,
            'energy_confidence': energy_confidence,
            'temperature': self.temperature
        }

        if base_confidence is not None:
            # Modulate base confidence by energy-based confidence
            # If energy is high (unstable), reduce confidence
            result['adjusted_confidence'] = base_confidence * energy_confidence

        return result
