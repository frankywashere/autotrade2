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


def load_training_samples(path: str, max_samples: int = None):
    """Load samples from either .pkl or .bin format.

    Args:
        path: Path to the sample file (.bin or .pkl)
        max_samples: Maximum number of samples to load (None for all)
    """
    from pathlib import Path
    path = Path(path)

    if path.suffix == '.bin' or path.suffix == '':
        # Try binary format first
        with open(path, 'rb') as f:
            magic = f.read(8)
        if magic == b'V15SAMP\x00':
            from .binary_loader import load_samples as load_bin_samples
            _, _, _, samples = load_bin_samples(str(path), max_samples=max_samples)
            return samples

    # Fall back to pickle format
    with open(path, 'rb') as f:
        samples = pickle.load(f)
    if max_samples is not None and max_samples < len(samples):
        samples = samples[:max_samples]
    return samples


def cmd_train(args):
    """Train the model."""
    from .training import ChannelDataset, create_dataloaders, Trainer
    from .models import create_model

    # Check if streaming mode requested
    streaming = getattr(args, 'streaming', False)
    chunk_size = getattr(args, 'chunk_size', 15000)
    max_samples = getattr(args, 'max_samples', None)

    if streaming:
        # Use streaming data loader for large datasets (low RAM usage)
        logger.info(f"Using streaming data loader (chunk_size={chunk_size:,})")
        logger.info(f"Loading from {args.samples}")

        from .training.streaming_dataset import create_streaming_dataloaders

        train_loader, val_loader, actual_feature_count = create_streaming_dataloaders(
            binary_path=args.samples,
            batch_size=args.batch_size,
            chunk_size=chunk_size,
            target_tf=args.target_tf,
            val_split=0.2,
            num_workers=args.num_workers,
            prefetch=True,
        )

        logger.info(f"Features: {actual_feature_count:,}")
    else:
        # Original in-memory loading
        logger.info(f"Loading samples from {args.samples}" + (f" (max {max_samples})" if max_samples else ""))
        samples = load_training_samples(args.samples, max_samples=max_samples)

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
            strategy=args.strategy,
            num_workers=args.num_workers
        )

        # Detect actual feature count from samples (C++ scanner may produce different count than config)
        actual_feature_count = len(samples[0].tf_features) if samples else TOTAL_FEATURES
    if actual_feature_count != TOTAL_FEATURES:
        logger.info(f"Note: Samples have {actual_feature_count} features (config says {TOTAL_FEATURES})")

    # Create model with actual feature count
    model = create_model({
        'input_dim': actual_feature_count,
        'hidden_dim': args.hidden_dim,
        'embed_dim': args.embed_dim,
        'n_attention_heads': args.n_attention_heads,
        'dropout': args.dropout,
        'use_gating': args.use_gating,
        'share_tf_weights': args.share_tf_weights,
        'use_window_selector': args.end_to_end,
        'enable_tsla_heads': args.enable_tsla_heads,
        'enable_spy_heads': args.enable_spy_heads,
        'enable_cross_correlation_heads': args.enable_cross_correlation_heads,
        'enable_durability_heads': args.enable_durability_heads,
        'enable_rsi_heads': args.enable_rsi_heads,
    })
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create training config with window selection options
    from .training.trainer import TrainingConfig
    config = TrainingConfig(
        # Basic training hyperparameters
        lr=args.lr,
        max_epochs=args.epochs,
        checkpoint_dir=args.output,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        early_stopping_patience=args.early_stopping_patience,
        scheduler=args.scheduler,
        analyze_features=not args.no_feature_analysis,
        # Loss function settings
        duration_loss_type=args.duration_loss_type,
        direction_loss_type=args.direction_loss_type,
        focal_gamma=args.focal_gamma,
        huber_delta=args.huber_delta,
        # Core task weights
        duration_weight=args.duration_weight,
        direction_weight=args.direction_weight,
        new_channel_weight=args.new_channel_weight,
        # Window selection settings
        use_end_to_end_loss=args.end_to_end,
        window_selection_weight=args.window_selection_weight,
        strategy=args.strategy,
        entropy_weight=args.entropy_weight,
        consistency_weight=args.consistency_weight,
        use_gumbel_softmax=args.use_gumbel_softmax,
        gumbel_temperature=args.gumbel_temperature,
        gumbel_temperature_min=args.gumbel_temperature_min,
        # TSLA break scan head weights
        tsla_bars_to_break_weight=args.tsla_bars_to_break_weight,
        tsla_break_direction_weight=args.tsla_break_direction_weight,
        tsla_break_magnitude_weight=args.tsla_break_magnitude_weight,
        tsla_returned_weight=args.tsla_returned_weight,
        tsla_bounces_weight=args.tsla_bounces_weight,
        tsla_channel_continued_weight=args.tsla_channel_continued_weight,
        tsla_durability_weight=args.tsla_durability_weight,
        tsla_bars_to_permanent_weight=args.tsla_bars_to_permanent_weight,
        # SPY break scan head weights
        spy_bars_to_break_weight=args.spy_bars_to_break_weight,
        spy_break_direction_weight=args.spy_break_direction_weight,
        spy_break_magnitude_weight=args.spy_break_magnitude_weight,
        spy_returned_weight=args.spy_returned_weight,
        spy_bounces_weight=args.spy_bounces_weight,
        spy_channel_continued_weight=args.spy_channel_continued_weight,
        spy_durability_weight=args.spy_durability_weight,
        spy_bars_to_permanent_weight=args.spy_bars_to_permanent_weight,
        # Cross-correlation head weights
        cross_direction_aligned_weight=args.cross_direction_aligned_weight,
        cross_who_broke_first_weight=args.cross_who_broke_first_weight,
        cross_break_lag_weight=args.cross_break_lag_weight,
        cross_both_permanent_weight=args.cross_both_permanent_weight,
        cross_return_aligned_weight=args.cross_return_aligned_weight,
        cross_durability_spread_weight=args.cross_durability_spread_weight,
        # TSLA RSI head weights
        tsla_rsi_at_break_weight=args.tsla_rsi_at_break_weight,
        tsla_rsi_overbought_weight=args.tsla_rsi_overbought_weight,
        tsla_rsi_oversold_weight=args.tsla_rsi_oversold_weight,
        tsla_rsi_divergence_weight=args.tsla_rsi_divergence_weight,
        # SPY RSI head weights
        spy_rsi_at_break_weight=args.spy_rsi_at_break_weight,
        spy_rsi_overbought_weight=args.spy_rsi_overbought_weight,
        spy_rsi_oversold_weight=args.spy_rsi_oversold_weight,
        spy_rsi_divergence_weight=args.spy_rsi_divergence_weight,
        # Cross-asset RSI head weights
        cross_rsi_aligned_weight=args.cross_rsi_aligned_weight,
        cross_rsi_spread_weight=args.cross_rsi_spread_weight,
        # Per-timeframe loss
        per_tf_loss_weight=args.per_tf_loss_weight,
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
    train_parser.add_argument('--max-samples', type=int, default=None,
        help='Maximum number of samples to load (for quick experiments)')
    train_parser.add_argument('--streaming', action='store_true',
        help='Use streaming data loader (low RAM, supports full dataset)')
    train_parser.add_argument('--chunk-size', type=int, default=15000,
        help='Samples per chunk for streaming loader (default: 15000 = ~2.8GB RAM)')
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
    train_parser.add_argument('--entropy-weight', type=float, default=0.1,
        help='Entropy regularization for window selection')
    train_parser.add_argument('--consistency-weight', type=float, default=0.05,
        help='Consistency with heuristic window selection')
    train_parser.add_argument('--use-gumbel-softmax', action='store_true',
        help='Use Gumbel-Softmax for differentiable selection')
    train_parser.add_argument('--gumbel-temperature', type=float, default=1.0,
        help='Initial Gumbel-Softmax temperature')
    train_parser.add_argument('--gumbel-temperature-min', type=float, default=0.1,
        help='Minimum Gumbel-Softmax temperature')
    train_parser.add_argument('--weight-decay', type=float, default=1e-5,
        help='Weight decay for optimizer')
    train_parser.add_argument('--warmup-steps', type=int, default=1000,
        help='Number of warmup steps for scheduler')
    train_parser.add_argument('--grad-clip', type=float, default=1.0,
        help='Gradient clipping value')
    train_parser.add_argument('--early-stopping-patience', type=int, default=10,
        help='Number of epochs without improvement before early stopping')
    train_parser.add_argument('--num-workers', type=int, default=0,
        help='Number of DataLoader workers (0 for main process, default 0 for MPS compatibility)')
    train_parser.add_argument('--no-feature-analysis', action='store_true',
        help='Skip feature correlation/constant analysis before training')
    train_parser.add_argument('--scheduler', type=str, default='onecycle',
        choices=['onecycle', 'cosine_restarts', 'none'],
        help='Learning rate scheduler type')
    train_parser.add_argument('--enable-tsla-heads', action='store_true',
        help='Enable TSLA break scan prediction heads')
    train_parser.add_argument('--enable-spy-heads', action='store_true',
        help='Enable SPY break scan prediction heads')
    train_parser.add_argument('--enable-cross-correlation-heads', action='store_true',
        help='Enable cross-asset correlation heads')
    train_parser.add_argument('--cross-direction-aligned-weight', type=float, default=1.0,
        help='Loss weight for cross-correlation direction aligned head (default: 1.0)')
    train_parser.add_argument('--cross-who-broke-first-weight', type=float, default=1.0,
        help='Loss weight for cross-correlation who-broke-first head (default: 1.0)')
    train_parser.add_argument('--cross-break-lag-weight', type=float, default=1.0,
        help='Loss weight for cross-correlation break lag head (default: 1.0)')
    train_parser.add_argument('--cross-both-permanent-weight', type=float, default=1.0,
        help='Loss weight for cross-correlation both-permanent head (default: 1.0)')
    train_parser.add_argument('--cross-return-aligned-weight', type=float, default=1.0,
        help='Loss weight for cross-correlation return aligned head (default: 1.0)')
    train_parser.add_argument('--cross-durability-spread-weight', type=float, default=0.3,
        help='Loss weight for cross-correlation durability spread head (default: 0.3)')
    train_parser.add_argument('--enable-durability-heads', action='store_true',
        help='Enable durability prediction heads')
    train_parser.add_argument('--enable-rsi-heads', action='store_true',
        help='Enable RSI prediction heads')
    train_parser.add_argument('--duration-loss-type', type=str, default='gaussian_nll',
        choices=['gaussian_nll', 'huber', 'mse'],
        help='Loss function for duration prediction')
    train_parser.add_argument('--direction-loss-type', type=str, default='bce',
        choices=['bce', 'focal'],
        help='Loss function for direction prediction')
    train_parser.add_argument('--focal-gamma', type=float, default=2.0,
        help='Gamma parameter for focal loss (only with --direction-loss-type focal)')
    train_parser.add_argument('--huber-delta', type=float, default=1.0,
        help='Delta parameter for Huber loss (only with --duration-loss-type huber)')
    # Core loss weights
    train_parser.add_argument('--duration-weight', type=float, default=1.0,
        help='Weight for duration loss')
    train_parser.add_argument('--direction-weight', type=float, default=1.0,
        help='Weight for direction loss')
    train_parser.add_argument('--new-channel-weight', type=float, default=1.0,
        help='Weight for new channel loss')
    # Model architecture arguments
    train_parser.add_argument('--hidden-dim', type=int, default=256,
        help='Hidden dimension size (default: 256)')
    train_parser.add_argument('--embed-dim', type=int, default=128,
        help='Embedding dimension size (default: 128)')
    train_parser.add_argument('--n-attention-heads', type=int, default=8,
        help='Number of attention heads (default: 8)')
    train_parser.add_argument('--dropout', type=float, default=0.1,
        help='Dropout rate (default: 0.1)')
    train_parser.add_argument('--use-gating', action='store_true',
        help='Enable feature gating')
    train_parser.add_argument('--share-tf-weights', action='store_true',
        help='Share TF encoder weights')
    # SPY break scan head weights
    train_parser.add_argument('--spy-bars-to-break-weight', type=float, default=1.0,
        help='Weight for SPY bars-to-break prediction head (default: 1.0)')
    train_parser.add_argument('--spy-break-direction-weight', type=float, default=1.0,
        help='Weight for SPY break direction prediction head (default: 1.0)')
    train_parser.add_argument('--spy-break-magnitude-weight', type=float, default=1.0,
        help='Weight for SPY break magnitude prediction head (default: 1.0)')
    train_parser.add_argument('--spy-returned-weight', type=float, default=1.0,
        help='Weight for SPY returned prediction head (default: 1.0)')
    train_parser.add_argument('--spy-bounces-weight', type=float, default=1.0,
        help='Weight for SPY bounces prediction head (default: 1.0)')
    train_parser.add_argument('--spy-channel-continued-weight', type=float, default=0.5,
        help='Weight for SPY channel continued prediction head (default: 0.5)')
    train_parser.add_argument('--spy-durability-weight', type=float, default=0.5,
        help='Weight for SPY durability prediction head (default: 0.5)')
    train_parser.add_argument('--spy-bars-to-permanent-weight', type=float, default=0.5,
        help='Weight for SPY bars-to-permanent prediction head (default: 0.5)')
    # TSLA break scan head weights
    train_parser.add_argument('--tsla-bars-to-break-weight', type=float, default=1.0,
        help='Weight for TSLA bars-to-break prediction head (default: 1.0)')
    train_parser.add_argument('--tsla-break-direction-weight', type=float, default=1.0,
        help='Weight for TSLA break direction prediction head (default: 1.0)')
    train_parser.add_argument('--tsla-break-magnitude-weight', type=float, default=1.0,
        help='Weight for TSLA break magnitude prediction head (default: 1.0)')
    train_parser.add_argument('--tsla-returned-weight', type=float, default=1.0,
        help='Weight for TSLA returned prediction head (default: 1.0)')
    train_parser.add_argument('--tsla-bounces-weight', type=float, default=1.0,
        help='Weight for TSLA bounces prediction head (default: 1.0)')
    train_parser.add_argument('--tsla-channel-continued-weight', type=float, default=1.0,
        help='Weight for TSLA channel continued prediction head (default: 1.0)')
    train_parser.add_argument('--tsla-durability-weight', type=float, default=0.5,
        help='Weight for TSLA durability prediction head (default: 0.5)')
    train_parser.add_argument('--tsla-bars-to-permanent-weight', type=float, default=0.5,
        help='Weight for TSLA bars-to-permanent prediction head (default: 0.5)')
    # TSLA RSI head weights
    train_parser.add_argument('--tsla-rsi-at-break-weight', type=float, default=1.0,
        help='Weight for TSLA RSI at break prediction head (default: 1.0)')
    train_parser.add_argument('--tsla-rsi-overbought-weight', type=float, default=1.0,
        help='Weight for TSLA RSI overbought prediction head (default: 1.0)')
    train_parser.add_argument('--tsla-rsi-oversold-weight', type=float, default=1.0,
        help='Weight for TSLA RSI oversold prediction head (default: 1.0)')
    train_parser.add_argument('--tsla-rsi-divergence-weight', type=float, default=1.0,
        help='Weight for TSLA RSI divergence prediction head (default: 1.0)')
    # SPY RSI head weights
    train_parser.add_argument('--spy-rsi-at-break-weight', type=float, default=1.0,
        help='Weight for SPY RSI at break prediction head (default: 1.0)')
    train_parser.add_argument('--spy-rsi-overbought-weight', type=float, default=1.0,
        help='Weight for SPY RSI overbought prediction head (default: 1.0)')
    train_parser.add_argument('--spy-rsi-oversold-weight', type=float, default=1.0,
        help='Weight for SPY RSI oversold prediction head (default: 1.0)')
    train_parser.add_argument('--spy-rsi-divergence-weight', type=float, default=1.0,
        help='Weight for SPY RSI divergence prediction head (default: 1.0)')
    # Cross-asset RSI head weights
    train_parser.add_argument('--cross-rsi-aligned-weight', type=float, default=1.0,
        help='Weight for cross-asset RSI aligned prediction head (default: 1.0)')
    train_parser.add_argument('--cross-rsi-spread-weight', type=float, default=1.0,
        help='Weight for cross-asset RSI spread prediction head (default: 1.0)')
    # Per-timeframe loss weight
    train_parser.add_argument('--per-tf-loss-weight', type=float, default=0.0,
        help='Weight for per-timeframe duration loss (0.0 = disabled, try 0.5 to enable)')

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
