"""
ML model loading — extracts model initialization from state.py.

Loads GBT (Surfer ML), EL/ER sub-models, and Intraday LightGBM filter.
All models are read-only after loading — safe to share across adapter instances.
"""

import logging
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class MLModels:
    """Container for all loaded ML models."""
    gbt_model: Any = None            # GBT shim with .predict()
    gbt_feature_names: list = field(default_factory=list)
    el_model: Any = None             # ExtremeLoserDetector
    er_model: Any = None             # ExtendedRunPredictor
    fast_rev_model: Any = None       # MomentumReversalDetector
    intraday_model: Any = None       # LightGBM filter
    intraday_features: list = field(default_factory=list)
    intraday_threshold: float = 0.5
    load_errors: list = field(default_factory=list)


def _find_model_path(filename: str) -> Optional[Path]:
    """Search for model file in standard locations."""
    candidates = [
        Path('surfer_models') / filename,
        Path(__file__).parent.parent.parent / 'surfer_models' / filename,
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 200:
            return p
    return None


def load_ml_models() -> MLModels:
    """Load all ML models. Returns MLModels container.

    Models are read-only after loading — safe to share across adapters.
    """
    models = MLModels()

    # 1. GBT model (Surfer ML)
    try:
        import pickle
        import numpy as np

        model_path = _find_model_path('gbt_model.pkl')
        if model_path:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            class _GBTShim:
                def __init__(self, model_dict, feature_names):
                    self.models = model_dict
                    self.feature_names = feature_names
                def predict(self, X):
                    results = {}
                    for name, mdl in self.models.items():
                        results[name] = mdl.predict(X)
                    return results

            models.gbt_model = _GBTShim(data['models'], data['feature_names'])
            models.gbt_feature_names = data['feature_names']

            # Verify
            test_pred = models.gbt_model.predict(
                np.zeros((1, len(models.gbt_feature_names)), dtype=np.float32))
            logger.info("GBT model loaded: %d features, keys=%s",
                        len(models.gbt_feature_names), list(test_pred.keys()))
        else:
            logger.warning("GBT model not found")
            models.load_errors.append("GBT model not found")
    except Exception as e:
        logger.warning("GBT model load failed: %s", e)
        models.load_errors.append(f"GBT: {e}")

    # 2. EL (Extreme Loser) + ER (Extended Run) + Fast Rev (Momentum Reversal) sub-models
    try:
        from v15.core.surfer_ml import (
            ExtremeLoserDetector, ExtendedRunPredictor, MomentumReversalDetector,
        )

        el_path = _find_model_path('extreme_loser_model.pkl')
        if el_path:
            models.el_model = ExtremeLoserDetector.load(str(el_path))
            logger.info("EL model loaded from %s", el_path)

        er_path = _find_model_path('extended_run_model.pkl')
        if er_path:
            models.er_model = ExtendedRunPredictor.load(str(er_path))
            logger.info("ER model loaded from %s", er_path)

        fr_path = _find_model_path('momentum_reversal_model.pkl')
        if fr_path:
            models.fast_rev_model = MomentumReversalDetector.load(str(fr_path))
            logger.info("Fast reversion model loaded from %s", fr_path)
    except Exception as e:
        logger.warning("EL/ER/FastRev model load failed: %s", e)
        models.load_errors.append(f"EL/ER/FastRev: {e}")

    # 3. Intraday ML model (LightGBM filter)
    try:
        import pickle
        intra_data = None
        intra_path = _find_model_path('intraday_ml_model.pkl')

        if intra_path:
            with open(intra_path, 'rb') as f:
                intra_data = pickle.load(f)
            logger.info("Intraday ML model loaded from %s", intra_path)
        else:
            # Fall back to embedded base64 model (for HF Spaces)
            try:
                import base64
                import io
                from v15.trading.intraday_ml_data import MODEL_B64
                raw = base64.b64decode(MODEL_B64.strip())
                intra_data = pickle.load(io.BytesIO(raw))
                logger.info("Intraday ML model loaded from embedded base64")
            except ImportError:
                logger.warning("Intraday ML model not found")
                models.load_errors.append("Intraday model not found")

        if intra_data:
            models.intraday_model = intra_data['model']
            models.intraday_features = intra_data['feature_names']
            models.intraday_threshold = intra_data.get('threshold', 0.5)
            logger.info("Intraday ML model ready: %d features, threshold=%.2f",
                        len(models.intraday_features), models.intraday_threshold)
    except Exception as e:
        logger.warning("Intraday ML model load failed: %s", e)
        models.load_errors.append(f"Intraday: {e}")

    return models
