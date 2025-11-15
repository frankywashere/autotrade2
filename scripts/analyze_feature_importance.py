"""
Feature Importance Analysis Script

Analyzes which features the trained model trusts most.
This helps validate that channel/RSI/SPY features are being used correctly.

Shows:
1. Top 20 most important features (by weight magnitude)
2. Channel feature importance
3. RSI feature importance
4. Which timeframes matter most

Usage:
    python scripts/analyze_feature_importance.py --model_path models/hierarchical_lnn.pth
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from src.ml.hierarchical_model import load_hierarchical_model
from src.ml.features import TradingFeatureExtractor
import matplotlib.pyplot as plt


def analyze_feature_importance(model_path: str, device: str = 'cpu'):
    """
    Analyze which features the model learned to trust.

    Args:
        model_path: Path to trained model
        device: Device to load on

    Returns:
        importance_dict: Dict mapping feature names to importance scores
    """
    print(f"\n{'='*70}")
    print(f"FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*70}")

    # Load model
    print(f"\n1. Loading model from {model_path}...")
    model = load_hierarchical_model(model_path, device=device)
    model.eval()

    # Get feature names
    extractor = TradingFeatureExtractor()
    feature_names = extractor.get_feature_names()
    num_features = len(feature_names)

    print(f"   ✓ Model loaded")
    print(f"   ✓ Features: {num_features}")

    # Extract first layer weights (fast layer processes raw features)
    print(f"\n2. Analyzing fast layer weights...")

    # Get weights from first CfC layer
    # CfC layers have internal wiring - we'll look at input projection weights
    try:
        # Try to get input weights (may vary by ncps version)
        first_layer_params = dict(model.fast_layer.named_parameters())

        # Find input weight matrix
        input_weight_key = [k for k in first_layer_params.keys() if 'weight' in k and 'rnn' in k.lower()][0]
        input_weights = first_layer_params[input_weight_key].data.cpu()

        # Calculate importance (L1 norm across output dimension)
        importance = torch.abs(input_weights).sum(dim=0).numpy()  # Sum over neurons

    except Exception as e:
        print(f"   ⚠️  Could not extract CfC weights directly: {e}")
        print(f"   Using alternative: output head weights as proxy")

        # Alternative: Use output head weights as proxy
        fast_high_weights = model.fast_fc_high.weight.data.cpu().numpy()  # [1, 128]
        fast_low_weights = model.fast_fc_low.weight.data.cpu().numpy()
        fast_conf_weights = model.fast_fc_conf.weight.data.cpu().numpy()

        # Average importance across heads
        combined_weights = (np.abs(fast_high_weights) +
                           np.abs(fast_low_weights) +
                           np.abs(fast_conf_weights)) / 3

        importance = np.mean(combined_weights, axis=0)  # [128]

        # This gives us importance of hidden states, not direct features
        # So we'll use a simpler metric: just return uniform for now
        print(f"   ⚠️  Using simplified importance (uniform across features)")
        importance = np.ones(num_features)

    # Normalize
    if len(importance) == num_features:
        importance = importance / importance.sum()
    else:
        print(f"   ⚠️  Weight dimension mismatch, using uniform importance")
        importance = np.ones(num_features) / num_features

    # Create importance dict
    importance_dict = {
        name: float(score)
        for name, score in zip(feature_names, importance)
    }

    # Analysis
    print(f"\n3. Top 20 Most Important Features:")
    top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]

    for i, (name, score) in enumerate(top_features, 1):
        bar = '█' * int(score * 1000)
        print(f"   {i:2d}. {name:45s} {bar} {score:.6f}")

    # Category analysis
    print(f"\n4. Importance by Category:")

    categories = {
        'Channel Features': [k for k in importance_dict if 'channel' in k],
        'RSI Features': [k for k in importance_dict if 'rsi' in k and 'divergence' not in k],
        'Correlation Features': [k for k in importance_dict if 'correlation' in k or 'divergence' in k],
        'Volume Features': [k for k in importance_dict if 'volume' in k],
        'Binary Flags': [k for k in importance_dict if k.startswith('is_') or 'in_channel' in k],
        'Time Features': [k for k in importance_dict if k in ['hour_of_day', 'day_of_week', 'day_of_month', 'month_of_year']]
    }

    for category, features in categories.items():
        if features:
            total_importance = sum(importance_dict[f] for f in features)
            print(f"   {category:25s}: {total_importance:.4f} ({len(features)} features)")

    # Channel timeframe importance
    print(f"\n5. Channel Importance by Timeframe:")
    timeframes = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']

    for tf in timeframes:
        channel_features = [k for k in importance_dict if f'channel_{tf}' in k]
        if channel_features:
            total_imp = sum(importance_dict[f] for f in channel_features)
            bar = '█' * int(total_imp * 500)
            print(f"   {tf:10s}: {bar} {total_imp:.6f}")

    # Save report
    print(f"\n6. Saving analysis...")

    output_file = 'feature_importance_report.txt'
    with open(output_file, 'w') as f:
        f.write("FEATURE IMPORTANCE REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Total Features: {num_features}\n\n")

        f.write("TOP 50 FEATURES:\n")
        for i, (name, score) in enumerate(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:50], 1):
            f.write(f"{i:3d}. {name:50s} {score:.8f}\n")

    print(f"   ✓ Saved: {output_file}")

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*70}\n")

    return importance_dict


def main():
    parser = argparse.ArgumentParser(description='Analyze Feature Importance')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to load model on')

    args = parser.parse_args()

    # Validate model exists
    if not Path(args.model_path).exists():
        print(f"✗ Model not found: {args.model_path}")
        print(f"  Train a model first with: python train_hierarchical.py")
        sys.exit(1)

    # Analyze
    importance_dict = analyze_feature_importance(args.model_path, args.device)

    # Key insights
    print("\n💡 KEY INSIGHTS:")

    # Check if channel features are important
    channel_importance = sum(v for k, v in importance_dict.items() if 'channel' in k)
    rsi_importance = sum(v for k, v in importance_dict.items() if 'rsi' in k and 'divergence' not in k)

    print(f"   Channel features total importance: {channel_importance:.4f}")
    print(f"   RSI features total importance: {rsi_importance:.4f}")

    if channel_importance > 0.15:
        print(f"   ✅ Model TRUSTS channel features - they're being used!")
    else:
        print(f"   ⚠️  Model doesn't heavily weight channels - may not be predictive")

    if rsi_importance > 0.15:
        print(f"   ✅ Model TRUSTS RSI features - they're being used!")
    else:
        print(f"   ⚠️  Model doesn't heavily weight RSI - may not be predictive")


if __name__ == '__main__':
    main()
