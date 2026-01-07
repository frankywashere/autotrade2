"""
Temperature Scheduler for End-to-End Window Selection

Implements temperature annealing for the softmax-based window selection mechanism.
Based on Phase 2b design principles:

- Start with high temperature (soft selection, exploration)
  - High T = flat softmax distribution = model considers all windows equally
  - Enables gradient flow from all windows during early training

- Anneal to low temperature (hard selection, exploitation)
  - Low T = sharp softmax distribution = model commits to best window
  - Approaches argmax behavior during late training/inference

Temperature Annealing Schedule:
- Exponential decay: T(t) = T_initial * (T_final / T_initial) ^ (t / T_anneal)
- Alternative schedules: linear, cosine, step

The scheduler integrates with the Trainer to:
1. Update temperature each epoch/step
2. Pass current temperature to model during forward pass
3. Support both epoch-based and step-based annealing
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class AnnealingSchedule(Enum):
    """Temperature annealing schedule types."""
    EXPONENTIAL = "exponential"  # T_i * (T_f / T_i) ^ progress
    LINEAR = "linear"            # T_i + (T_f - T_i) * progress
    COSINE = "cosine"            # T_f + 0.5 * (T_i - T_f) * (1 + cos(pi * progress))
    STEP = "step"                # Discrete steps at specified milestones


@dataclass
class TemperatureConfig:
    """Configuration for temperature annealing."""

    # Temperature bounds
    initial_temp: float = 5.0     # Start high for exploration
    final_temp: float = 0.1       # End low for exploitation

    # Annealing duration
    anneal_epochs: Optional[int] = None   # Epochs to anneal over (epoch-based)
    anneal_steps: Optional[int] = None    # Steps to anneal over (step-based)

    # Schedule type
    schedule: AnnealingSchedule = AnnealingSchedule.EXPONENTIAL

    # Step schedule milestones (for STEP schedule)
    # Format: [(epoch_or_step, temperature), ...]
    milestones: Optional[list] = None

    # Warmup: hold at initial temperature for N epochs/steps before annealing
    warmup_epochs: int = 0
    warmup_steps: int = 0


class TemperatureScheduler:
    """
    Temperature scheduler for softmax-based window selection.

    Anneals temperature during training to transition from exploration
    (soft selection across all windows) to exploitation (hard selection
    of the best window).

    Example usage:
        # Epoch-based annealing
        scheduler = TemperatureScheduler(
            initial_temp=5.0,
            final_temp=0.1,
            anneal_epochs=50
        )

        for epoch in range(100):
            temp = scheduler.get_temperature(epoch=epoch)
            # Pass temp to model during forward pass

        # Step-based annealing
        scheduler = TemperatureScheduler(
            initial_temp=5.0,
            final_temp=0.1,
            anneal_steps=10000
        )

        for step in range(total_steps):
            temp = scheduler.get_temperature(step=step)
    """

    def __init__(
        self,
        initial_temp: float = 5.0,
        final_temp: float = 0.1,
        anneal_epochs: Optional[int] = None,
        anneal_steps: Optional[int] = None,
        schedule: AnnealingSchedule = AnnealingSchedule.EXPONENTIAL,
        warmup_epochs: int = 0,
        warmup_steps: int = 0,
        milestones: Optional[list] = None,
    ):
        """
        Initialize temperature scheduler.

        Args:
            initial_temp: Starting temperature (high = soft selection)
            final_temp: Ending temperature (low = hard selection)
            anneal_epochs: Number of epochs to anneal over (for epoch-based)
            anneal_steps: Number of steps to anneal over (for step-based)
            schedule: Annealing schedule type
            warmup_epochs: Hold at initial temp for this many epochs
            warmup_steps: Hold at initial temp for this many steps
            milestones: For STEP schedule, list of (epoch/step, temp) tuples

        Note: Specify either anneal_epochs OR anneal_steps, not both.
              If neither is specified, defaults to 50 epochs.
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs
        self.warmup_steps = warmup_steps
        self.milestones = milestones or []

        # Determine annealing mode
        if anneal_steps is not None:
            self.anneal_steps = anneal_steps
            self.anneal_epochs = None
            self.step_based = True
        elif anneal_epochs is not None:
            self.anneal_epochs = anneal_epochs
            self.anneal_steps = None
            self.step_based = False
        else:
            # Default to 50 epochs
            self.anneal_epochs = 50
            self.anneal_steps = None
            self.step_based = False

        # Validate
        if self.initial_temp <= 0 or self.final_temp <= 0:
            raise ValueError("Temperatures must be positive")
        if self.initial_temp < self.final_temp:
            # Allow but warn - unusual to increase temperature
            pass

        # Sort milestones for step schedule
        if self.schedule == AnnealingSchedule.STEP and self.milestones:
            self.milestones = sorted(self.milestones, key=lambda x: x[0])

        # Track current state
        self._current_temp = initial_temp
        self._current_epoch = 0
        self._current_step = 0

    @classmethod
    def from_config(cls, config: TemperatureConfig) -> 'TemperatureScheduler':
        """Create scheduler from config dataclass."""
        return cls(
            initial_temp=config.initial_temp,
            final_temp=config.final_temp,
            anneal_epochs=config.anneal_epochs,
            anneal_steps=config.anneal_steps,
            schedule=config.schedule,
            warmup_epochs=config.warmup_epochs,
            warmup_steps=config.warmup_steps,
            milestones=config.milestones,
        )

    def get_temperature(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ) -> float:
        """
        Get current temperature based on training progress.

        Args:
            epoch: Current epoch (for epoch-based annealing)
            step: Current global step (for step-based annealing)

        Returns:
            Current temperature value

        Note: Uses step if step_based=True, otherwise uses epoch.
              If neither is provided, uses internal tracking.
        """
        # Determine progress unit
        if self.step_based:
            current = step if step is not None else self._current_step
            warmup = self.warmup_steps
            total = self.anneal_steps
        else:
            current = epoch if epoch is not None else self._current_epoch
            warmup = self.warmup_epochs
            total = self.anneal_epochs

        # Update internal state
        if self.step_based and step is not None:
            self._current_step = step
        elif not self.step_based and epoch is not None:
            self._current_epoch = epoch

        # During warmup, return initial temperature
        if current < warmup:
            self._current_temp = self.initial_temp
            return self.initial_temp

        # Adjust for warmup
        adjusted_current = current - warmup

        # Compute progress (0 to 1)
        progress = min(adjusted_current / total, 1.0) if total > 0 else 1.0

        # Compute temperature based on schedule
        temp = self._compute_temperature(progress)
        self._current_temp = temp
        return temp

    def _compute_temperature(self, progress: float) -> float:
        """
        Compute temperature for given progress [0, 1].

        Args:
            progress: Training progress from 0 (start) to 1 (end)

        Returns:
            Temperature value
        """
        if self.schedule == AnnealingSchedule.EXPONENTIAL:
            # Exponential decay: T_i * (T_f / T_i) ^ progress
            # This gives smooth exponential transition
            if self.initial_temp > 0 and self.final_temp > 0:
                ratio = self.final_temp / self.initial_temp
                return self.initial_temp * (ratio ** progress)
            else:
                return self.final_temp

        elif self.schedule == AnnealingSchedule.LINEAR:
            # Linear interpolation: T_i + (T_f - T_i) * progress
            return self.initial_temp + (self.final_temp - self.initial_temp) * progress

        elif self.schedule == AnnealingSchedule.COSINE:
            # Cosine annealing: T_f + 0.5 * (T_i - T_f) * (1 + cos(pi * progress))
            # Smooth transition that slows down at extremes
            return self.final_temp + 0.5 * (self.initial_temp - self.final_temp) * (
                1 + math.cos(math.pi * progress)
            )

        elif self.schedule == AnnealingSchedule.STEP:
            # Step-wise schedule based on milestones
            if not self.milestones:
                return self.final_temp

            # Find the appropriate milestone
            # milestones is [(epoch/step, temp), ...]
            current_temp = self.initial_temp
            for threshold, temp in self.milestones:
                if progress >= threshold / (self.anneal_epochs or self.anneal_steps or 1):
                    current_temp = temp
                else:
                    break
            return current_temp

        else:
            # Default to exponential
            ratio = self.final_temp / self.initial_temp
            return self.initial_temp * (ratio ** progress)

    def step(self, epoch: Optional[int] = None, global_step: Optional[int] = None):
        """
        Update scheduler state after an epoch or step.

        This method is called by the Trainer after each epoch/step.

        Args:
            epoch: Current epoch number
            global_step: Current global step number
        """
        if epoch is not None:
            self._current_epoch = epoch
        if global_step is not None:
            self._current_step = global_step

        # Update temperature
        self.get_temperature(epoch=epoch, step=global_step)

    @property
    def current_temperature(self) -> float:
        """Get the last computed temperature."""
        return self._current_temp

    def state_dict(self) -> dict:
        """
        Get scheduler state for checkpointing.

        Returns:
            Dictionary with scheduler state
        """
        return {
            'initial_temp': self.initial_temp,
            'final_temp': self.final_temp,
            'anneal_epochs': self.anneal_epochs,
            'anneal_steps': self.anneal_steps,
            'schedule': self.schedule.value,
            'warmup_epochs': self.warmup_epochs,
            'warmup_steps': self.warmup_steps,
            'milestones': self.milestones,
            'step_based': self.step_based,
            'current_temp': self._current_temp,
            'current_epoch': self._current_epoch,
            'current_step': self._current_step,
        }

    def load_state_dict(self, state_dict: dict):
        """
        Load scheduler state from checkpoint.

        Args:
            state_dict: Dictionary with scheduler state
        """
        self.initial_temp = state_dict.get('initial_temp', self.initial_temp)
        self.final_temp = state_dict.get('final_temp', self.final_temp)
        self.anneal_epochs = state_dict.get('anneal_epochs', self.anneal_epochs)
        self.anneal_steps = state_dict.get('anneal_steps', self.anneal_steps)
        self.warmup_epochs = state_dict.get('warmup_epochs', self.warmup_epochs)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.milestones = state_dict.get('milestones', self.milestones)
        self.step_based = state_dict.get('step_based', self.step_based)
        self._current_temp = state_dict.get('current_temp', self.initial_temp)
        self._current_epoch = state_dict.get('current_epoch', 0)
        self._current_step = state_dict.get('current_step', 0)

        if 'schedule' in state_dict:
            self.schedule = AnnealingSchedule(state_dict['schedule'])

    def __repr__(self) -> str:
        mode = "step" if self.step_based else "epoch"
        total = self.anneal_steps if self.step_based else self.anneal_epochs
        return (
            f"TemperatureScheduler("
            f"T={self.initial_temp:.2f}->{self.final_temp:.2f}, "
            f"{self.schedule.value}, "
            f"{total} {mode}s, "
            f"current={self._current_temp:.3f})"
        )


def create_default_scheduler(num_epochs: int = 100) -> TemperatureScheduler:
    """
    Create a default temperature scheduler for typical training.

    Default settings:
    - Initial temperature: 5.0 (soft selection)
    - Final temperature: 0.1 (near-hard selection)
    - Exponential decay over 80% of training
    - 5-epoch warmup at high temperature

    Args:
        num_epochs: Total number of training epochs

    Returns:
        Configured TemperatureScheduler
    """
    return TemperatureScheduler(
        initial_temp=5.0,
        final_temp=0.1,
        anneal_epochs=int(num_epochs * 0.8),  # Anneal over 80% of training
        schedule=AnnealingSchedule.EXPONENTIAL,
        warmup_epochs=5,
    )


if __name__ == '__main__':
    """Demonstrate temperature scheduler behavior."""

    print("Temperature Scheduler Demo")
    print("=" * 50)

    # Create schedulers with different schedules
    schedules = [
        AnnealingSchedule.EXPONENTIAL,
        AnnealingSchedule.LINEAR,
        AnnealingSchedule.COSINE,
    ]

    for schedule in schedules:
        scheduler = TemperatureScheduler(
            initial_temp=5.0,
            final_temp=0.1,
            anneal_epochs=50,
            schedule=schedule,
            warmup_epochs=5,
        )

        print(f"\n{schedule.value.upper()} Schedule:")
        print("-" * 30)

        for epoch in [0, 5, 10, 25, 40, 50, 55, 100]:
            temp = scheduler.get_temperature(epoch=epoch)
            print(f"  Epoch {epoch:3d}: T = {temp:.4f}")

    # Demonstrate step-based annealing
    print("\n" + "=" * 50)
    print("Step-based Annealing:")
    print("-" * 30)

    scheduler = TemperatureScheduler(
        initial_temp=5.0,
        final_temp=0.1,
        anneal_steps=10000,
        schedule=AnnealingSchedule.EXPONENTIAL,
        warmup_steps=500,
    )

    for step in [0, 500, 1000, 2500, 5000, 7500, 10000, 15000]:
        temp = scheduler.get_temperature(step=step)
        print(f"  Step {step:5d}: T = {temp:.4f}")

    print("\n" + "=" * 50)
    print(f"Default scheduler: {create_default_scheduler(100)}")
