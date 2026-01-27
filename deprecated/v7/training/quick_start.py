"""
Quick Start - Minimal Training Example

This is the absolute minimum code needed to train a model.
Perfect for getting started quickly.

Run: python quick_start.py
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training import (
    prepare_dataset_from_scratch,
    create_dataloaders,
    Trainer,
    TrainingConfig
)
from features.feature_ordering import FEATURE_ORDER


class SimpleModel(nn.Module):
    """Minimal model - just for demonstration."""

    def __init__(self):
        super().__init__()
        # We'll infer dimensions from first batch
        self.initialized = False

    def _initialize(self, total_dim):
        """Lazy initialization based on input size."""
        self.fc1 = nn.Linear(total_dim, 128)
        self.fc2 = nn.Linear(128, 64)

        # Output heads
        self.duration = nn.Linear(64, 1)
        self.break_dir = nn.Linear(64, 2)
        self.new_dir = nn.Linear(64, 3)
        self.perm_break = nn.Linear(64, 1)

        self.initialized = True

    def forward(self, features):
        # Concatenate all features using CANONICAL ordering
        # CRITICAL: Must use FEATURE_ORDER, NOT features.values()!
        x = torch.cat([features[k] for k in FEATURE_ORDER if k in features], dim=1)

        # Initialize on first forward pass
        if not self.initialized:
            self._initialize(x.shape[1])
            self.to(x.device)

        # Forward
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        return {
            'duration': self.duration(x),
            'break_direction': self.break_dir(x),
            'new_direction': self.new_dir(x),
            'permanent_break': self.perm_break(x)
        }


def main():
    print("Quick Start Training Example")
    print("=" * 60)

    # Setup paths
    data_dir = Path(__file__).parent.parent.parent / "data"
    cache_dir = data_dir / "feature_cache"

    # Step 1: Prepare dataset (uses cache if available)
    print("\n1. Preparing dataset...")
    train_samples, val_samples, test_samples = prepare_dataset_from_scratch(
        data_dir=data_dir,
        cache_dir=cache_dir,
        window=50,
        step=50,  # Larger step = fewer samples = faster
        force_rebuild=False
    )

    # Step 2: Create dataloaders
    print("\n2. Creating dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples,
        val_samples,
        test_samples,
        batch_size=32,
        num_workers=0  # Set to 4 for faster loading
    )

    # Step 3: Create model
    print("\n3. Creating model...")
    model = SimpleModel()

    # Step 4: Configure training
    print("\n4. Configuring training...")
    config = TrainingConfig(
        num_epochs=10,  # Just 10 epochs for quick demo
        learning_rate=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        save_dir=Path('./checkpoints'),
        log_dir=Path('./logs')
    )

    # Step 5: Train
    print("\n5. Training...")
    print(f"   Device: {config.device}")
    print(f"   Train samples: {len(train_samples)}")
    print(f"   Val samples: {len(val_samples)}")

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )

    history = trainer.train()

    # Done!
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best model saved to: {config.save_dir / 'best_model.pt'}")
    print("=" * 60)


if __name__ == '__main__':
    main()
