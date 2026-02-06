"""
Post-training temperature scaling for calibrating per-TF direction probabilities.

Usage:
    python -m v15.pipeline calibrate --checkpoint path/to/best.pt --data path/to/data.flat

This script:
1. Loads model + validation split from .flat data
2. Collects all per-TF direction logits and ground truth across val set
3. Optimizes temperature T via scipy.minimize_scalar to minimize NLL
4. Computes ECE (Expected Calibration Error) before/after
5. Saves temperature_calibration.json next to checkpoint
"""
import json
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def collect_direction_logits_and_targets(
    model,
    data_loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect all per-TF direction logits and ground truth from a data loader.

    Returns:
        logits: [N] float array of raw logits
        targets: [N] float array of 0/1 targets
    """
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch_data in data_loader:
            features, labels = batch_data
            features = features.to(device)

            # Forward with per-TF outputs
            _, per_tf_preds = model.forward_with_per_tf(features, validate=False)

            direction_logits = per_tf_preds['direction_logits']  # [batch, n_tfs]

            # Get targets and validity
            per_tf_dir = labels.get('per_tf_direction')
            per_tf_dir_valid = labels.get('per_tf_direction_valid')

            if per_tf_dir is None or per_tf_dir_valid is None:
                continue

            per_tf_dir = per_tf_dir.to(device)
            per_tf_dir_valid = per_tf_dir_valid.to(device).bool()

            if not per_tf_dir_valid.any():
                continue

            valid_logits = direction_logits[per_tf_dir_valid].cpu().numpy()
            valid_targets = per_tf_dir[per_tf_dir_valid].cpu().numpy().astype(float)

            all_logits.append(valid_logits)
            all_targets.append(valid_targets)

    if not all_logits:
        return np.array([]), np.array([])

    return np.concatenate(all_logits), np.concatenate(all_targets)


def compute_nll(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    """Compute negative log-likelihood with temperature scaling."""
    scaled = logits / temperature
    # Numerically stable sigmoid + NLL
    # NLL = -[y * log(sigmoid(x)) + (1-y) * log(1-sigmoid(x))]
    # = -[y * x - log(1 + exp(x))]  (using log_sigmoid identity)
    log_probs = -np.logaddexp(0, -scaled)  # log(sigmoid(x))
    log_1m_probs = -np.logaddexp(0, scaled)  # log(1 - sigmoid(x))
    nll = -(targets * log_probs + (1 - targets) * log_1m_probs)
    return nll.mean()


def compute_ece(logits: np.ndarray, targets: np.ndarray, temperature: float, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    scaled = logits / temperature
    probs = 1.0 / (1.0 + np.exp(-scaled))

    # For binary: confidence = max(p, 1-p), accuracy = (pred == target)
    predictions = (probs > 0.5).astype(float)
    confidences = np.maximum(probs, 1.0 - probs)
    accuracies = (predictions == targets).astype(float)

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin

    return ece


def calibrate_temperature(
    checkpoint_path: str,
    data_path: str,
    val_split: float = 0.2,
    batch_size: int = 128,
    seed: int = 42,
) -> dict:
    """
    Run temperature calibration on validation data.

    Args:
        checkpoint_path: Path to model checkpoint
        data_path: Path to .flat data directory
        val_split: Fraction of data used for validation
        batch_size: Batch size for inference
        seed: Random seed

    Returns:
        Dict with calibration results
    """
    from scipy.optimize import minimize_scalar
    from .inference import Predictor
    from .training.flat_dataset import create_flat_dataloaders

    checkpoint_path = Path(checkpoint_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    predictor = Predictor.load(str(checkpoint_path), device=str(device))
    model = predictor.model

    # Create val loader
    _, val_loader, _ = create_flat_dataloaders(
        flat_dir=data_path,
        batch_size=batch_size,
        val_split=val_split,
        seed=seed,
    )

    logger.info("Collecting direction logits from validation set...")
    logits, targets = collect_direction_logits_and_targets(model, val_loader, device)

    if len(logits) == 0:
        logger.error("No valid per-TF direction samples found in validation data")
        return {'error': 'no_valid_samples'}

    logger.info(f"Collected {len(logits):,} valid logit-target pairs")

    # Compute metrics before calibration (T=1.0)
    nll_before = compute_nll(logits, targets, temperature=1.0)
    ece_before = compute_ece(logits, targets, temperature=1.0)

    logger.info(f"Before calibration: NLL={nll_before:.4f}, ECE={ece_before:.4f}")

    # Optimize temperature
    result = minimize_scalar(
        lambda t: compute_nll(logits, targets, t),
        bounds=(0.1, 10.0),
        method='bounded',
    )
    optimal_temp = result.x

    # Compute metrics after calibration
    nll_after = compute_nll(logits, targets, temperature=optimal_temp)
    ece_after = compute_ece(logits, targets, temperature=optimal_temp)

    logger.info(f"Optimal temperature: {optimal_temp:.4f}")
    logger.info(f"After calibration: NLL={nll_after:.4f}, ECE={ece_after:.4f}")
    logger.info(f"Improvement: NLL {nll_before - nll_after:+.4f}, ECE {ece_before - ece_after:+.4f}")

    # Save calibration
    output = {
        'temperature': float(optimal_temp),
        'nll_before': float(nll_before),
        'nll_after': float(nll_after),
        'ece_before': float(ece_before),
        'ece_after': float(ece_after),
        'n_samples': int(len(logits)),
    }

    output_path = checkpoint_path.parent / 'temperature_calibration.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved calibration to {output_path}")

    return output
