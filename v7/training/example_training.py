"""
Example Training Script

Demonstrates complete usage of the training pipeline:
1. Prepare dataset from raw data
2. Define a simple model architecture
3. Configure training
4. Train the model
5. Evaluate on test set

This is a working example you can run directly or modify for your needs.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

from .dataset import prepare_dataset_from_scratch, create_dataloaders
from .trainer import Trainer, TrainingConfig


class SimpleChannelPredictor(nn.Module):
    """
    Simple baseline model for channel prediction.

    Architecture:
    - Concatenate all feature groups
    - Pass through shared MLP backbone
    - Split into task-specific heads for each prediction

    This is a simple baseline. You can replace with more sophisticated
    architectures (Transformers, Graph Networks, etc.)
    """

    def __init__(
        self,
        feature_dims: dict,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super().__init__()

        # Calculate total input dimension
        self.feature_dims = feature_dims
        self.total_input_dim = sum(feature_dims.values())

        # Shared backbone
        layers = []
        input_dim = self.total_input_dim

        for i in range(num_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Task-specific heads
        # Duration prediction (regression)
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Break direction (2-class classification: UP/DOWN)
        self.break_direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)
        )

        # New channel direction (3-class: BEAR/SIDEWAYS/BULL)
        self.new_direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3)
        )

        # Permanent break (binary classification)
        self.permanent_break_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, features: dict) -> dict:
        """
        Forward pass.

        Args:
            features: Dict of feature tensors with keys like:
                - 'tsla_5min': [batch_size, feature_dim]
                - 'tsla_15min': [batch_size, feature_dim]
                - 'spy_5min': [batch_size, feature_dim]
                - 'vix': [batch_size, feature_dim]
                - etc.

        Returns:
            Dict with predictions:
                - 'duration': [batch_size, 1]
                - 'break_direction': [batch_size, 2]
                - 'new_direction': [batch_size, 3]
                - 'permanent_break': [batch_size, 1]
        """
        # Concatenate all features
        feature_list = []
        for key in sorted(features.keys()):  # Sort for consistent ordering
            feature_list.append(features[key])

        x = torch.cat(feature_list, dim=1)

        # Shared backbone
        x = self.backbone(x)

        # Task-specific predictions
        predictions = {
            'duration': self.duration_head(x),
            'break_direction': self.break_direction_head(x),
            'new_direction': self.new_direction_head(x),
            'permanent_break': self.permanent_break_head(x)
        }

        return predictions


def get_feature_dims(train_loader):
    """
    Infer feature dimensions from first batch.

    Args:
        train_loader: DataLoader

    Returns:
        Dict mapping feature names to dimensions
    """
    features, _ = next(iter(train_loader))
    feature_dims = {k: v.shape[1] for k, v in features.items()}
    return feature_dims


def main():
    """Main training pipeline."""

    # ==================== Setup Paths ====================
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    cache_dir = project_root / "data" / "feature_cache"
    checkpoint_dir = project_root / "checkpoints"
    log_dir = project_root / "logs"

    # ==================== Prepare Dataset ====================
    print("=" * 80)
    print("STEP 1: Preparing Dataset")
    print("=" * 80)

    train_samples, val_samples, test_samples = prepare_dataset_from_scratch(
        data_dir=data_dir,
        cache_dir=cache_dir,
        window=50,
        step=25,  # Step=25 for more samples
        min_cycles=1,
        train_end="2022-12-31",
        val_end="2023-12-31",
        include_history=False,  # Set True for full features (slower, ~5min)
        force_rebuild=False  # Set True to rebuild cache
    )

    # ==================== Create DataLoaders ====================
    print("\n" + "=" * 80)
    print("STEP 2: Creating DataLoaders")
    print("=" * 80)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_samples,
        val_samples,
        test_samples,
        batch_size=32,
        num_workers=0,  # Set to 0 for debugging, increase for faster loading
        augment_train=True,
        pin_memory=torch.cuda.is_available()
    )

    # Get feature dimensions
    feature_dims = get_feature_dims(train_loader)
    print(f"\nFeature dimensions:")
    for k, v in sorted(feature_dims.items()):
        print(f"  {k}: {v}")
    print(f"  Total: {sum(feature_dims.values())}")

    # ==================== Create Model ====================
    print("\n" + "=" * 80)
    print("STEP 3: Creating Model")
    print("=" * 80)

    model = SimpleChannelPredictor(
        feature_dims=feature_dims,
        hidden_dim=256,
        num_layers=3,
        dropout=0.2
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: SimpleChannelPredictor")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # ==================== Configure Training ====================
    print("\n" + "=" * 80)
    print("STEP 4: Configuring Training")
    print("=" * 80)

    config = TrainingConfig(
        # Model
        model_class=SimpleChannelPredictor,
        model_kwargs={'feature_dims': feature_dims},

        # Training
        num_epochs=50,
        learning_rate=0.001,
        weight_decay=0.0001,
        batch_size=32,
        gradient_clip=1.0,

        # Loss weights (tune these based on importance)
        duration_weight=1.0,
        break_direction_weight=2.0,  # More weight on break direction
        new_direction_weight=1.0,
        permanent_break_weight=1.0,

        # Optimization
        optimizer='adam',
        scheduler='cosine',
        scheduler_kwargs={},

        # Mixed precision (faster on GPU)
        use_amp=torch.cuda.is_available(),

        # Early stopping
        early_stopping_patience=10,
        early_stopping_metric='val_loss',
        early_stopping_mode='min',

        # Checkpointing
        save_dir=checkpoint_dir,
        save_every_n_epochs=5,
        save_best_only=True,

        # Logging
        log_dir=log_dir,
        log_every_n_steps=10,
        use_tensorboard=False,  # Set True to enable TensorBoard

        # Device
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42
    )

    print(f"\nTraining Configuration:")
    print(f"  Device: {config.device}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Optimizer: {config.optimizer}")
    print(f"  Scheduler: {config.scheduler}")
    print(f"  Mixed Precision: {config.use_amp}")

    # ==================== Train Model ====================
    print("\n" + "=" * 80)
    print("STEP 5: Training Model")
    print("=" * 80)

    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Train
    history = trainer.train()

    # ==================== Evaluate on Test Set ====================
    print("\n" + "=" * 80)
    print("STEP 6: Evaluating on Test Set")
    print("=" * 80)

    if test_loader is not None:
        test_metrics = trainer.validate()  # Uses val method, but on test set
        print(f"\nTest Set Results:")
        print(f"  Test Loss: {test_metrics['total']:.4f}")
        print(f"  Break Direction Accuracy: {test_metrics['break_direction_acc']:.3f}")
        print(f"  New Direction Accuracy: {test_metrics['new_direction_acc']:.3f}")
        print(f"  Permanent Break Accuracy: {test_metrics['permanent_break_acc']:.3f}")

    # ==================== Save Training Summary ====================
    print("\n" + "=" * 80)
    print("STEP 7: Saving Training Summary")
    print("=" * 80)

    summary = {
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()},
        'train_history': history['train'],
        'val_history': history['val'],
        'final_epoch': trainer.current_epoch,
        'best_val_metric': trainer.best_val_metric,
    }

    summary_path = log_dir / 'training_summary.json'
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nTraining complete!")
    print(f"  Best validation {config.early_stopping_metric}: {trainer.best_val_metric:.4f}")
    print(f"  Best model saved to: {config.save_dir / 'best_model.pt'}")
    print(f"  Training summary saved to: {summary_path}")

    # ==================== Print Usage Instructions ====================
    print("\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print(f"\n1. Load best model:")
    print(f"   checkpoint = torch.load('{config.save_dir / 'best_model.pt'}')")
    print(f"   model.load_state_dict(checkpoint['model_state_dict'])")
    print(f"\n2. View training curves (if TensorBoard enabled):")
    print(f"   tensorboard --logdir {log_dir}")
    print(f"\n3. Use model for inference:")
    print(f"   predictions = model(features)")


if __name__ == '__main__':
    main()
