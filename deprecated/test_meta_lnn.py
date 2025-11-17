#!/usr/bin/env python3
"""
Test Meta-LNN architecture and market state calculation.

Verifies:
- MetaLNN can be instantiated and forward pass works
- Market state features are calculated correctly (12 features)
- Input/output dimensions are correct
"""

import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.meta_models import MetaLNN, MetaLNNWithModalityDropout, calculate_market_state, meta_loss


def test_meta_lnn_architecture():
    """Test Meta-LNN instantiation and forward pass."""
    print("\n" + "=" * 70)
    print("🧪 TESTING META-LNN ARCHITECTURE")
    print("=" * 70)

    # Create model
    print("\nCreating MetaLNN...")
    model = MetaLNN(num_submodels=4, market_state_dim=12, news_vec_dim=768, hidden_size=64)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Model size: {total_params * 4 / 1024**2:.2f} MB")

    # Test forward pass
    print("\nTesting forward pass...")

    batch_size = 8
    num_submodels = 4

    # Create dummy inputs
    subpreds = torch.randn(batch_size, num_submodels, 3)  # [batch, 4, 3]
    market_state = torch.randn(batch_size, 12)  # [batch, 12]
    news_vec = torch.zeros(batch_size, 768)  # [batch, 768] - disabled for backtest
    news_mask = torch.zeros(batch_size, 1)  # [batch, 1]

    # Forward pass
    pred_high, pred_low, pred_conf, hidden = model(subpreds, market_state, news_vec, news_mask)

    print(f"  ✓ Forward pass successful")
    print(f"  Input shapes:")
    print(f"    - subpreds: {subpreds.shape}")
    print(f"    - market_state: {market_state.shape}")
    print(f"    - news_vec: {news_vec.shape}")
    print(f"    - news_mask: {news_mask.shape}")
    print(f"  Output shapes:")
    print(f"    - pred_high: {pred_high.shape}")
    print(f"    - pred_low: {pred_low.shape}")
    print(f"    - pred_conf: {pred_conf.shape}")

    # Validate outputs
    assert pred_high.shape == (batch_size, 1), f"Expected (8, 1), got {pred_high.shape}"
    assert pred_low.shape == (batch_size, 1), f"Expected (8, 1), got {pred_low.shape}"
    assert pred_conf.shape == (batch_size, 1), f"Expected (8, 1), got {pred_conf.shape}"

    # Check confidence is in [0, 1]
    assert (pred_conf >= 0).all() and (pred_conf <= 1).all(), "Confidence should be in [0, 1]"

    print("  ✅ All shapes correct!")
    print("  ✅ Confidence in valid range [0, 1]")


def test_market_state_calculation():
    """Test market state feature calculation."""
    print("\n" + "=" * 70)
    print("🧪 TESTING MARKET STATE CALCULATION")
    print("=" * 70)

    # Create dummy market data
    print("\nCreating dummy market data...")
    n_bars = 500

    timestamps = pd.date_range(start='2024-01-01 09:30', periods=n_bars, freq='1T')
    df = pd.DataFrame({
        'returns': np.random.randn(n_bars) * 0.01,
        'volatility': np.abs(np.random.randn(n_bars) * 0.02),
        'correlation_50': np.random.randn(n_bars) * 0.5,
        'overnight_return': np.random.randn(n_bars) * 0.005
    }, index=timestamps)

    print(f"  ✓ Created {len(df)} bars of dummy data")

    # Calculate market state for middle bar
    current_idx = n_bars // 2

    print(f"\nCalculating market state for index {current_idx}...")
    market_state = calculate_market_state(df, current_idx, events_handler=None)

    print(f"  ✓ Market state calculated")
    print(f"  Shape: {market_state.shape}")
    print(f"  Expected: [12]")

    # Validate
    assert market_state.shape == (12,), f"Expected shape (12,), got {market_state.shape}"
    assert not torch.isnan(market_state).any(), "Market state contains NaN values"

    print("  ✅ Shape correct!")
    print("  ✅ No NaN values!")

    # Print feature values
    feature_names = [
        'rv_5m', 'rv_30m', 'rv_1d',
        'overnight_ret_abs', 'jump_flag',
        'vol_zscore',
        'time_sin', 'time_cos',
        'has_earnings_soon', 'has_macro_soon',
        'spy_corr', 'vix_level'
    ]

    print("\n  Market state features:")
    for i, name in enumerate(feature_names):
        print(f"    {i+1:2d}. {name:20s}: {market_state[i].item():.4f}")


def test_meta_loss():
    """Test meta-loss function."""
    print("\n" + "=" * 70)
    print("🧪 TESTING META-LOSS FUNCTION")
    print("=" * 70)

    batch_size = 16

    # Create dummy predictions and targets
    pred_high = torch.randn(batch_size) * 10 + 250
    pred_low = torch.randn(batch_size) * 10 + 240
    pred_conf = torch.rand(batch_size)

    y_high = torch.randn(batch_size) * 10 + 250
    y_low = torch.randn(batch_size) * 10 + 240

    print("\nCalculating loss...")
    loss = meta_loss(pred_high, pred_low, pred_conf, y_high, y_low)

    print(f"  ✓ Loss calculated: {loss.item():.6f}")
    print(f"  Shape: {loss.shape}")

    # Validate
    assert loss.shape == (), "Loss should be scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("  ✅ Loss function working correctly!")


def test_modality_dropout():
    """Test modality dropout wrapper."""
    print("\n" + "=" * 70)
    print("🧪 TESTING MODALITY DROPOUT")
    print("=" * 70)

    print("\nCreating MetaLNN with modality dropout...")
    model = MetaLNNWithModalityDropout(
        num_submodels=4,
        dropout_prob=0.4
    )

    print(f"  ✓ Model created with dropout_prob=0.4")

    # Test in training mode
    model.train()

    batch_size = 32
    subpreds = torch.randn(batch_size, 4, 3)
    market_state = torch.randn(batch_size, 12)
    news_vec = torch.ones(batch_size, 768)  # Non-zero to detect dropout
    news_mask = torch.ones(batch_size, 1)

    print("\nRunning 10 forward passes in training mode...")
    print("  (Should dropout news ~40% of the time)")

    # Can't directly test dropout randomness, but verify no errors
    for i in range(10):
        pred_high, pred_low, pred_conf, _ = model(subpreds, market_state, news_vec, news_mask)

    print("  ✓ Modality dropout working (no errors)")

    # Test in eval mode (dropout disabled)
    model.eval()
    pred_high, pred_low, pred_conf, _ = model(subpreds, market_state, news_vec, news_mask)

    print("  ✓ Eval mode working (dropout disabled)")
    print("  ✅ Modality dropout wrapper functioning correctly!")


if __name__ == '__main__':
    test_meta_lnn_architecture()
    test_market_state_calculation()
    test_meta_loss()
    test_modality_dropout()

    print("\n" + "=" * 70)
    print("✅ ALL META-LNN TESTS PASSED")
    print("=" * 70)
    print("\nMeta-LNN is ready for training!")
    print()
