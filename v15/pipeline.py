"""
V15 Pipeline - CLI entry point for scanning, training, and evaluation.

Usage:
    python -m v15.pipeline scan --data-dir data --output samples.pkl
    python -m v15.pipeline train --samples samples.pkl --output model.pt
    python -m v15.pipeline eval --model model.pt --samples test.pkl
    python -m v15.pipeline analyze --samples samples.pkl
    python -m v15.pipeline infer --model model.pt --data-dir data
    python -m v15.pipeline dashboard
    python -m v15.pipeline info
"""
import argparse
import logging
import sys
from pathlib import Path
import pickle
import json

from .config import SCANNER_CONFIG, TRAINING_CONFIG, TOTAL_FEATURES
from .exceptions import V15Error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def cmd_scan(args):
    """
    DEPRECATED: Python scanner has been removed.

    Use the C++ scanner instead for 10x faster performance:
        cd v15_cpp/build && ./v15_scanner --data-dir ../../data --output samples.bin

    Then train with:
        python -m v15.pipeline train --samples samples.bin --output model.pt
    """
    logger.error("Python scanner has been removed. Use v15_cpp/build/v15_scanner instead.")
    logger.error("Example: cd v15_cpp/build && ./v15_scanner --data-dir ../../data --output samples.bin")
    sys.exit(1)


def load_training_samples(path: str):
    """Load samples from either .pkl or .bin format."""
    from pathlib import Path
    path = Path(path)

    if path.suffix == '.bin' or path.suffix == '':
        # Try binary format first
        with open(path, 'rb') as f:
            magic = f.read(8)
        if magic == b'V15SAMP\x00':
            from .binary_loader import load_samples as load_bin_samples
            _, _, _, samples = load_bin_samples(str(path))
            return samples

    # Fall back to pickle format
    with open(path, 'rb') as f:
        return pickle.load(f)


def cmd_train(args):
    """Train the model."""
    from .training import ChannelDataset, create_dataloaders, Trainer
    from .models import create_model

    logger.info(f"Loading samples from {args.samples}")
    samples = load_training_samples(args.samples)

    # Split train/val
    split_idx = int(len(samples) * 0.8)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    logger.info(f"Train: {len(train_samples)}, Val: {len(val_samples)}")

    # Log window selection strategy
    logger.info(f"Window selection strategy: {args.strategy}")
    if args.end_to_end:
        logger.info(f"End-to-end learning enabled (weight={args.window_selection_weight})")

    # Create dataloaders with window selection strategy
    train_loader, val_loader = create_dataloaders(
        train_samples, val_samples,
        batch_size=args.batch_size,
        target_tf=args.target_tf,
        strategy=args.strategy
    )

    # Detect actual feature count from samples (C++ scanner may produce different count than config)
    actual_feature_count = len(samples[0].tf_features) if samples else TOTAL_FEATURES
    if actual_feature_count != TOTAL_FEATURES:
        logger.info(f"Note: Samples have {actual_feature_count} features (config says {TOTAL_FEATURES})")

    # Create model with actual feature count
    model = create_model({'input_dim': actual_feature_count})
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create training config with window selection options
    from .training.trainer import TrainingConfig
    config = TrainingConfig(
        lr=args.lr,
        max_epochs=args.epochs,
        checkpoint_dir=args.output,
        use_end_to_end_loss=args.end_to_end,
        window_selection_weight=args.window_selection_weight,
        strategy=args.strategy,
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    history = trainer.train()

    logger.info("Training complete!")


def cmd_analyze(args):
    """Analyze feature correlations."""
    from .features.validation import analyze_correlations, get_feature_stats
    import numpy as np

    logger.info(f"Loading samples from {args.samples}")
    samples = load_training_samples(args.samples)

    # Extract feature matrix
    feature_names = sorted(samples[0].tf_features.keys())
    feature_matrix = np.array([
        [s.tf_features.get(name, 0.0) for name in feature_names]
        for s in samples
    ])

    logger.info(f"Feature matrix: {feature_matrix.shape}")

    # Analyze correlations
    logger.info("Analyzing correlations...")
    corr_results = analyze_correlations(feature_matrix, feature_names)

    logger.info(f"Highly correlated pairs: {len(corr_results['highly_correlated_pairs'])}")
    logger.info(f"Suggested drops: {len(corr_results['suggested_drops'])}")

    # Feature stats
    stats = get_feature_stats(feature_matrix, feature_names)

    # Save report
    if args.output:
        report = {
            'n_samples': len(samples),
            'n_features': len(feature_names),
            'highly_correlated_pairs': [
                (a, b, float(c)) for a, b, c in corr_results['highly_correlated_pairs'][:100]
            ],
            'suggested_drops': corr_results['suggested_drops'][:50],
        }
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {args.output}")


def cmd_infer(args):
    """Make predictions with trained model."""
    from .inference import Predictor
    from .data import load_market_data

    predictor = Predictor.load(args.model)
    tsla, spy, vix = load_market_data(args.data_dir)

    # Use last N bars
    prediction = predictor.predict(
        tsla.iloc[-args.lookback:],
        spy.iloc[-args.lookback:],
        vix.iloc[-args.lookback:]
    )

    print(f"Prediction at {prediction.timestamp}")
    print(f"  Duration: {prediction.duration_mean:.0f} ± {prediction.duration_std:.0f} bars")
    print(f"  Direction: {prediction.direction} ({prediction.direction_prob:.1%})")
    print(f"  New Channel: {prediction.new_channel}")
    print(f"  Confidence: {prediction.confidence:.1%}")


def cmd_dashboard(args):
    """Launch Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / 'dashboard.py'
    subprocess.run(['streamlit', 'run', str(dashboard_path)])


def cmd_info(args):
    """Show V15 system information."""
    from .deprecated import print_deprecation_guide
    from .config import TOTAL_FEATURES, TIMEFRAMES

    print(f"V15 Channel Prediction System")
    print(f"  Total Features: {TOTAL_FEATURES}")
    print(f"  Timeframes: {len(TIMEFRAMES)}")
    print()
    print_deprecation_guide()


def main():
    parser = argparse.ArgumentParser(description='V15 Channel Prediction Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Extract features from data')
    scan_parser.add_argument('--data-dir', required=True, help='Data directory')
    scan_parser.add_argument('--output', required=True, help='Output pickle file')
    scan_parser.add_argument('--step', type=int, default=SCANNER_CONFIG['step'])
    scan_parser.add_argument('--warmup', type=int, default=SCANNER_CONFIG['warmup_bars'])
    scan_parser.add_argument('--workers', type=int, default=SCANNER_CONFIG['workers'])
    scan_parser.add_argument('--max-samples', type=int, default=None,
        help='Maximum number of samples to generate (for testing)')
    scan_parser.add_argument('--incremental', action='store_true',
        help='Write results incrementally to disk to reduce memory usage')
    scan_parser.add_argument('--incremental-chunk', type=int, default=1000,
        help='Number of samples to buffer before writing to disk (default: 1000)')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--samples', required=True, help='Samples pickle file')
    train_parser.add_argument('--output', required=True, help='Output directory')
    train_parser.add_argument('--batch-size', type=int, default=TRAINING_CONFIG['batch_size'])
    train_parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['learning_rate'])
    train_parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['max_epochs'])
    train_parser.add_argument('--target-tf', default='daily', help='Target timeframe for labels')
    train_parser.add_argument('--strategy', default='bounce_first',
        choices=['bounce_first', 'label_validity', 'balanced_score', 'quality_score', 'learned'],
        help='Window selection strategy')
    train_parser.add_argument('--end-to-end', action='store_true',
        help='Enable end-to-end window selection learning')
    train_parser.add_argument('--window-selection-weight', type=float, default=0.1,
        help='Weight for window selection loss (only with --end-to-end)')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze feature correlations')
    analyze_parser.add_argument('--samples', required=True, help='Samples pickle file')
    analyze_parser.add_argument('--output', help='Output JSON report')

    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Make predictions with trained model')
    infer_parser.add_argument('--model', required=True, help='Path to trained model')
    infer_parser.add_argument('--data-dir', required=True, help='Data directory')
    infer_parser.add_argument('--lookback', type=int, default=500, help='Number of bars to use for prediction')

    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch Streamlit dashboard')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show V15 system information')

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == 'scan':
            cmd_scan(args)
        elif args.command == 'train':
            cmd_train(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'infer':
            cmd_infer(args)
        elif args.command == 'dashboard':
            cmd_dashboard(args)
        elif args.command == 'info':
            cmd_info(args)
    except V15Error as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
