"""
Test-Time Training (TTT) Module for Channel Prediction Model

TTT enables the model to adapt during inference by performing gradient updates
on a self-supervised loss. This helps the model adapt to regime changes in
financial time series without requiring labeled data at inference time.

Key concepts:
- Only a subset of parameters are updated (LayerNorms by default)
- Updates happen every N bars to balance adaptation vs. stability
- Loss types: consistency (temporal), reconstruction, prediction_agreement
- Safeguards: drift limits, gradient clipping, warmup steps

Usage:
    from v7.training.ttt import TTTConfig, TTTAdapter, TTTMode

    # Create config
    config = TTTConfig(
        mode=TTTMode.ADAPTIVE,
        learning_rate=1e-4,
        update_frequency=12,
        loss_type='consistency'
    )

    # Create adapter
    adapter = TTTAdapter(model, config)
    adapter.initialize()

    # During inference
    for features in data_stream:
        predictions, stats = adapter.step(features)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime
import numpy as np


class TTTMode(Enum):
    """Test-Time Training mode."""
    STATIC = auto()      # No adaptation - standard inference (default)
    ADAPTIVE = auto()    # Full TTT - adapt on every update cycle
    MIXED = auto()       # Adapt only when confidence below threshold


@dataclass
class TTTConfig:
    """
    Configuration for Test-Time Training (TTT).

    TTT enables online adaptation during inference by updating a subset of
    model parameters using self-supervised losses. This allows the model to
    adapt to distribution shifts without labeled data.
    """
    # Core settings
    enabled: bool = False
    mode: TTTMode = TTTMode.STATIC
    learning_rate: float = 1e-4
    update_frequency: int = 12  # Update every N forward passes

    # Loss configuration
    loss_type: str = 'consistency'  # 'consistency', 'reconstruction', 'prediction_agreement'
    consistency_weight: float = 1.0
    reconstruction_weight: float = 0.5
    agreement_weight: float = 0.3

    # Parameter selection
    parameter_subset: str = 'layernorm_only'  # 'layernorm_only', 'layernorm_and_attention', 'all_adaptable'

    # Safeguards
    max_drift_pct: float = 0.15  # Max 15% drift from base weights
    max_grad_norm: float = 1.0  # Gradient clipping
    warmup_steps: int = 10  # Steps before first TTT update

    # For MIXED mode
    confidence_threshold: float = 0.6  # Adapt when confidence below this

    # EMA for consistency loss
    ema_decay: float = 0.99


class ConsistencyLoss(nn.Module):
    """
    Temporal consistency loss for Test-Time Training.

    Penalizes large changes between consecutive predictions, encouraging
    smooth adaptation. This is based on the assumption that market conditions
    don't change drastically between consecutive time steps.

    Loss = EMA-weighted MSE between current and previous predictions
    """

    def __init__(
        self,
        duration_weight: float = 1.0,
        direction_weight: float = 1.0,
        confidence_weight: float = 0.5,
        ema_decay: float = 0.99
    ):
        super().__init__()
        self.duration_weight = duration_weight
        self.direction_weight = direction_weight
        self.confidence_weight = confidence_weight
        self.ema_decay = ema_decay

        # Buffers to store EMA of previous predictions
        self._prev_duration: Optional[torch.Tensor] = None
        self._prev_direction: Optional[torch.Tensor] = None
        self._prev_confidence: Optional[torch.Tensor] = None
        self._initialized = False

    def reset(self):
        """Reset stored predictions (call at start of new sequence)."""
        self._prev_duration = None
        self._prev_direction = None
        self._prev_confidence = None
        self._initialized = False

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        update_ema: bool = True
    ) -> torch.Tensor:
        """
        Compute consistency loss between current and EMA of previous predictions.

        Args:
            predictions: Current model outputs containing:
                - 'duration_mean': [batch, 11]
                - 'direction_logits': [batch, 11]
                - 'confidence': [batch, 11]
            update_ema: Whether to update the EMA buffers

        Returns:
            Scalar consistency loss (0.0 if first call)
        """
        duration = predictions.get('duration_mean')
        direction = predictions.get('direction_logits')
        confidence = predictions.get('confidence')

        if duration is None:
            return torch.tensor(0.0, device=next(iter(predictions.values())).device)

        device = duration.device

        if not self._initialized:
            # First call - initialize buffers, return zero loss
            if update_ema:
                self._prev_duration = duration.detach().clone()
                self._prev_direction = direction.detach().clone() if direction is not None else None
                self._prev_confidence = confidence.detach().clone() if confidence is not None else None
                self._initialized = True
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Compute MSE between current and EMA-smoothed previous predictions
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        if self._prev_duration is not None:
            duration_loss = F.mse_loss(duration, self._prev_duration.to(device))
            loss = loss + self.duration_weight * duration_loss

        if direction is not None and self._prev_direction is not None:
            direction_loss = F.mse_loss(direction, self._prev_direction.to(device))
            loss = loss + self.direction_weight * direction_loss

        if confidence is not None and self._prev_confidence is not None:
            confidence_loss = F.mse_loss(confidence, self._prev_confidence.to(device))
            loss = loss + self.confidence_weight * confidence_loss

        # Update EMA buffers
        if update_ema:
            self._prev_duration = (
                self.ema_decay * self._prev_duration +
                (1 - self.ema_decay) * duration.detach()
            )
            if direction is not None and self._prev_direction is not None:
                self._prev_direction = (
                    self.ema_decay * self._prev_direction +
                    (1 - self.ema_decay) * direction.detach()
                )
            if confidence is not None and self._prev_confidence is not None:
                self._prev_confidence = (
                    self.ema_decay * self._prev_confidence +
                    (1 - self.ema_decay) * confidence.detach()
                )

        return loss


class ReconstructionLoss(nn.Module):
    """
    Feature reconstruction loss for Test-Time Training.

    Adds noise to input features and trains the model's internal representations
    to be robust to this perturbation. Good representations should produce
    similar outputs despite input noise.
    """

    def __init__(self, noise_std: float = 0.1):
        super().__init__()
        self.noise_std = noise_std

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        clean_output: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute reconstruction loss by comparing clean vs noisy outputs.

        Args:
            model: The model to evaluate
            x: Input features [batch, features]
            clean_output: Outputs from clean input

        Returns:
            Scalar reconstruction loss
        """
        device = x.device

        # Add noise to input
        noise = torch.randn_like(x) * self.noise_std
        x_noisy = x + noise

        # Get outputs from noisy input
        noisy_output = model(x_noisy, return_attention=True)

        # Loss: noisy outputs should match clean outputs
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        if 'duration_mean' in clean_output and 'duration_mean' in noisy_output:
            loss = loss + F.mse_loss(
                noisy_output['duration_mean'],
                clean_output['duration_mean'].detach()
            )

        if 'direction_logits' in clean_output and 'direction_logits' in noisy_output:
            loss = loss + F.mse_loss(
                noisy_output['direction_logits'],
                clean_output['direction_logits'].detach()
            ) * 0.5

        return loss


class PredictionAgreementLoss(nn.Module):
    """
    Prediction agreement loss for Test-Time Training.

    Encourages per-TF predictions to agree with the aggregate prediction.
    The aggregate synthesizes information from all timeframes via cross-TF
    attention, so individual TF predictions should be consistent with this view.
    """

    def __init__(
        self,
        duration_weight: float = 1.0,
        direction_weight: float = 1.0
    ):
        super().__init__()
        self.duration_weight = duration_weight
        self.direction_weight = direction_weight

    def forward(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute agreement loss between per-TF and aggregate predictions.

        Args:
            predictions: Model outputs containing per-TF and aggregate predictions

        Returns:
            Scalar agreement loss
        """
        if 'aggregate' not in predictions:
            return torch.tensor(0.0, requires_grad=True)

        agg = predictions['aggregate']
        device = predictions.get('duration_mean', agg.get('duration_mean')).device
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Duration agreement
        if 'duration_mean' in predictions and 'duration_mean' in agg:
            per_tf_duration = predictions['duration_mean']  # [batch, 11]
            agg_duration = agg['duration_mean']  # [batch, 1]

            # Per-TF should be close to aggregate (weighted by confidence if available)
            if 'confidence' in predictions:
                weights = F.softmax(predictions['confidence'], dim=1).detach()
                diff = (per_tf_duration - agg_duration.expand_as(per_tf_duration)) ** 2
                loss = loss + self.duration_weight * (diff * weights).sum(dim=1).mean()
            else:
                loss = loss + self.duration_weight * F.mse_loss(
                    per_tf_duration.mean(dim=1, keepdim=True),
                    agg_duration
                )

        # Direction agreement
        if 'direction_logits' in predictions and 'direction_logits' in agg:
            per_tf_dir = predictions['direction_logits']  # [batch, 11]
            agg_dir = agg['direction_logits']  # [batch, 1]

            loss = loss + self.direction_weight * F.mse_loss(
                per_tf_dir.mean(dim=1, keepdim=True),
                agg_dir
            )

        return loss


class TTTAdapter:
    """
    Manages Test-Time Training adaptation for HierarchicalCfCModel.

    This adapter:
    1. Identifies adaptable parameters based on configuration
    2. Creates separate optimizer for TTT parameters
    3. Computes self-supervised loss for adaptation
    4. Performs gradient updates during inference
    5. Enforces drift limits to prevent catastrophic forgetting
    """

    def __init__(self, model: nn.Module, config: TTTConfig):
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device

        # State tracking
        self.step_count = 0
        self.update_count = 0
        self.loss_history: List[float] = []
        self.last_update_time: Optional[datetime] = None

        # Will be populated by initialize()
        self.adaptable_params: List[nn.Parameter] = []
        self.base_weights: Dict[str, torch.Tensor] = {}
        self.param_drift: Dict[str, float] = {}
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Loss functions
        self.consistency_loss = ConsistencyLoss(ema_decay=config.ema_decay)
        self.reconstruction_loss = ReconstructionLoss()
        self.agreement_loss = PredictionAgreementLoss()

        # Previous output for consistency loss
        self.previous_output: Optional[Dict[str, torch.Tensor]] = None

    def initialize(self):
        """Initialize TTT state: identify params, snapshot weights, create optimizer."""
        self._identify_adaptable_params()
        self._snapshot_base_weights()
        self._create_optimizer()
        self._freeze_non_ttt_params()

    def _identify_adaptable_params(self):
        """Find parameters to adapt based on config.parameter_subset."""
        self.adaptable_params = []

        subset = self.config.parameter_subset

        for name, module in self.model.named_modules():
            if subset == 'layernorm_only':
                # Only LayerNorm parameters
                if isinstance(module, nn.LayerNorm):
                    for param in module.parameters():
                        self.adaptable_params.append(param)

            elif subset == 'layernorm_and_attention':
                # LayerNorm + attention output projection
                if isinstance(module, nn.LayerNorm):
                    for param in module.parameters():
                        self.adaptable_params.append(param)
                elif 'output_proj' in name and 'attention' in name.lower():
                    for param in module.parameters():
                        self.adaptable_params.append(param)

            elif subset == 'all_adaptable':
                # All norms + projections
                if isinstance(module, nn.LayerNorm):
                    for param in module.parameters():
                        self.adaptable_params.append(param)
                elif 'proj' in name:
                    for param in module.parameters():
                        self.adaptable_params.append(param)

        # Remove duplicates while preserving order
        seen = set()
        unique_params = []
        for p in self.adaptable_params:
            if id(p) not in seen:
                seen.add(id(p))
                unique_params.append(p)
        self.adaptable_params = unique_params

    def _snapshot_base_weights(self):
        """
        Store copy of original weights for drift limiting.

        Saves ALL parameters in self.adaptable_params, not just LayerNorm.
        Uses parameter id() as unique key since adaptable_params can include
        attention/projection layers depending on config.parameter_subset.
        """
        self.base_weights = {}
        for idx, param in enumerate(self.adaptable_params):
            # Use both index and id for a unique, stable key
            key = f"param_{idx}_{id(param)}"
            self.base_weights[key] = param.data.clone()

        # Also store a mapping from key to parameter for easy access
        self._param_key_map = {
            f"param_{idx}_{id(param)}": param
            for idx, param in enumerate(self.adaptable_params)
        }

    def _create_optimizer(self):
        """Create optimizer for TTT parameters only."""
        if self.adaptable_params:
            self.optimizer = torch.optim.Adam(
                self.adaptable_params,
                lr=self.config.learning_rate
            )

    def _freeze_non_ttt_params(self):
        """Freeze all parameters except TTT-adaptable ones."""
        adaptable_ids = {id(p) for p in self.adaptable_params}

        for param in self.model.parameters():
            if id(param) in adaptable_ids:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _unfreeze_all_params(self):
        """Restore all parameters to trainable."""
        for param in self.model.parameters():
            param.requires_grad = True

    def prepare_for_inference(self):
        """Set model to appropriate mode for TTT inference."""
        if self.config.mode == TTTMode.STATIC:
            self.model.eval()
        else:
            # Keep model in train mode for LayerNorm running stats
            # but manually control gradient computation
            self.model.train()
            # Disable dropout for stability
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.eval()

    def compute_ttt_loss(
        self,
        x: torch.Tensor,
        output: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute self-supervised TTT loss based on config.loss_type.

        Args:
            x: Input features
            output: Model outputs

        Returns:
            total_loss: Combined TTT loss
            loss_dict: Individual loss components for logging
        """
        loss_dict = {}
        device = x.device

        if self.config.loss_type == 'consistency':
            loss = self.consistency_loss(output)
            loss_dict['consistency'] = loss.item() if loss.requires_grad else 0.0

        elif self.config.loss_type == 'reconstruction':
            loss = self.reconstruction_loss(self.model, x, output)
            loss_dict['reconstruction'] = loss.item() if loss.requires_grad else 0.0

        elif self.config.loss_type == 'prediction_agreement':
            loss = self.agreement_loss(output)
            loss_dict['agreement'] = loss.item() if loss.requires_grad else 0.0

        else:
            raise ValueError(f"Unknown TTT loss type: {self.config.loss_type}")

        loss_dict['total'] = loss.item() if loss.requires_grad else 0.0
        return loss, loss_dict

    def _enforce_drift_limits(self):
        """
        Clamp ALL parameters in self.base_weights to stay within max_drift_pct.

        This works with all parameter types (LayerNorm, attention, projections)
        by iterating through the stored base_weights dictionary directly.
        """
        for key, base in self.base_weights.items():
            # Get the corresponding parameter from our mapping
            if not hasattr(self, '_param_key_map') or key not in self._param_key_map:
                continue

            param = self._param_key_map[key]
            max_delta = self.config.max_drift_pct * (base.abs().mean() + 1e-6)
            param.data = torch.clamp(
                param.data,
                base - max_delta,
                base + max_delta
            )

            # Track drift for monitoring
            drift = (param.data - base).abs().mean() / (base.abs().mean() + 1e-6)
            self.param_drift[key] = drift.item()

    def step(
        self,
        x: torch.Tensor,
        force_update: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Perform one TTT step: forward pass + optional parameter update.

        Args:
            x: Input features [batch_size, total_features]
            force_update: If True, update regardless of update_frequency

        Returns:
            predictions: Model predictions (detached, safe to use downstream)
            stats: Dictionary with TTT statistics
        """
        self.step_count += 1
        stats: Dict[str, Any] = {'step': self.step_count, 'updated': False}

        x = x.to(self.device)

        # Check if TTT is active
        if self.config.mode == TTTMode.STATIC:
            with torch.no_grad():
                output = self.model(x, return_attention=True)
            return self._detach_output(output), stats

        # Forward pass with gradients for TTT
        output = self.model(x, return_attention=True)

        # Check if we should update
        should_update = (
            self.step_count >= self.config.warmup_steps and
            (force_update or self.step_count % self.config.update_frequency == 0)
        )

        # For MIXED mode, check confidence
        if self.config.mode == TTTMode.MIXED and should_update:
            avg_confidence = output.get('confidence', torch.tensor([0.5])).mean()
            if avg_confidence >= self.config.confidence_threshold:
                should_update = False
                stats['skipped_reason'] = 'confidence_above_threshold'

        if should_update and self.optimizer is not None:
            # Compute TTT loss
            loss, loss_dict = self.compute_ttt_loss(x, output)
            stats['loss'] = loss_dict

            if loss.requires_grad:
                # Backward and update
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.adaptable_params,
                        self.config.max_grad_norm
                    )
                    stats['grad_norm'] = grad_norm.item()

                self.optimizer.step()

                # Enforce drift limits
                self._enforce_drift_limits()

                self.update_count += 1
                self.last_update_time = datetime.now()
                stats['updated'] = True

                self.loss_history.append(loss_dict['total'])

        # Store for next iteration's consistency loss
        self.previous_output = self._detach_output(output)

        return self._detach_output(output), stats

    def _detach_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Detach all tensors in output dict to prevent gradient leakage."""
        detached = {}
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                detached[key] = value.detach()
            elif isinstance(value, dict):
                detached[key] = self._detach_output(value)
            elif isinstance(value, list):
                detached[key] = [
                    v.detach() if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
            else:
                detached[key] = value
        return detached

    def reset(self):
        """Reset TTT state for new session."""
        self.step_count = 0
        self.update_count = 0
        self.loss_history = []
        self.param_drift = {}
        self.previous_output = None
        self.consistency_loss.reset()

        # Restore ALL base weights (works with any parameter type)
        if hasattr(self, '_param_key_map'):
            for key, base in self.base_weights.items():
                if key in self._param_key_map:
                    param = self._param_key_map[key]
                    param.data.copy_(base)

        # Reset optimizer state
        self._create_optimizer()

        # Reset hidden states if model supports it
        if hasattr(self.model, 'reset_hidden_states'):
            self.model.reset_hidden_states()

    def switch_mode(self, new_mode: TTTMode, reset_weights: bool = True):
        """
        Handle mode transitions with proper state management.

        Args:
            new_mode: Target TTT mode
            reset_weights: If True, reload base weights on transition
        """
        old_mode = self.config.mode

        if old_mode == new_mode:
            return

        if reset_weights or new_mode == TTTMode.STATIC:
            self.reset()

        self.config.mode = new_mode
        self.prepare_for_inference()

    def get_status(self) -> Dict[str, Any]:
        """Get TTT status for dashboard display."""
        return {
            'mode': self.config.mode.name,
            'enabled': self.config.mode != TTTMode.STATIC,
            'update_count': self.update_count,
            'step_count': self.step_count,
            'cumulative_loss': sum(self.loss_history),
            'avg_loss': np.mean(self.loss_history) if self.loss_history else 0.0,
            'recent_loss': np.mean(self.loss_history[-10:]) if self.loss_history else 0.0,
            'last_update': self.last_update_time,
            'max_drift': max(self.param_drift.values()) if self.param_drift else 0.0,
            'num_adaptable_params': len(self.adaptable_params),
            'total_adaptable_params': sum(p.numel() for p in self.adaptable_params),
        }

    def check_realignment_needed(self) -> bool:
        """
        Check if automatic realignment is needed.

        Triggers:
        1. Param drift exceeds threshold
        2. Loss is consistently high (potential runaway)
        """
        # Check drift threshold
        max_drift = max(self.param_drift.values()) if self.param_drift else 0
        if max_drift > self.config.max_drift_pct:
            return True

        # Check for runaway loss
        if len(self.loss_history) > 10:
            recent_avg = np.mean(self.loss_history[-10:])
            if recent_avg > 5.0:  # Arbitrary threshold
                return True

        return False


def create_ttt_adapter(
    model: nn.Module,
    mode: str = 'static',
    learning_rate: float = 1e-4,
    update_frequency: int = 12,
    loss_type: str = 'consistency',
    parameter_subset: str = 'layernorm_only'
) -> TTTAdapter:
    """
    Factory function to create a TTT adapter.

    Args:
        model: HierarchicalCfCModel instance
        mode: 'static', 'adaptive', or 'mixed'
        learning_rate: Learning rate for TTT updates
        update_frequency: Update every N forward passes
        loss_type: 'consistency', 'reconstruction', or 'prediction_agreement'
        parameter_subset: Which params to adapt

    Returns:
        Configured TTTAdapter
    """
    mode_map = {
        'static': TTTMode.STATIC,
        'adaptive': TTTMode.ADAPTIVE,
        'mixed': TTTMode.MIXED
    }

    config = TTTConfig(
        enabled=mode != 'static',
        mode=mode_map.get(mode.lower(), TTTMode.STATIC),
        learning_rate=learning_rate,
        update_frequency=update_frequency,
        loss_type=loss_type,
        parameter_subset=parameter_subset
    )

    adapter = TTTAdapter(model, config)
    adapter.initialize()
    adapter.prepare_for_inference()

    return adapter
