"""
Online Learning System for Hierarchical LNN

Handles continuous error monitoring, validation scheduling, and adaptive weight updates.
Key features:
- Error tracking per layer (fast, medium, slow)
- Cross-layer error propagation (fast errors → medium/slow learns)
- Adaptive fusion weight adjustment (reward accurate layers)
- Validation scheduling (checks predictions when validation_time reached)
"""

import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import sys
import yaml
import copy

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from src.ml.database import PredictionDatabase


class OnlineLearner:
    """
    Manages online learning for Hierarchical LNN.

    Monitors predictions, validates when ready, triggers weight updates
    when errors exceed thresholds, and adjusts fusion weights based on
    layer accuracy.
    """

    def __init__(
        self,
        model,
        db_path: str = 'data/predictions.db',
        config_path: str = 'config/hierarchical_config.yaml',
        error_threshold_high: float = None,  # Load from config if None
        error_threshold_medium: float = None,
        error_threshold_low: float = None,
        fast_layer_lr: float = None,
        medium_layer_lr: float = None,
        slow_layer_lr: float = None,
        fusion_weight_adaptation_rate: float = None
    ):
        """
        Initialize online learner with configuration system.

        Args:
            model: HierarchicalLNN model instance
            db_path: Path to predictions database
            config_path: Path to YAML configuration file
            error_threshold_high/medium/low: Error thresholds (override config if provided)
            fast/medium/slow_layer_lr: Learning rates (override config if provided)
            fusion_weight_adaptation_rate: Fusion adaptation rate (override config if provided)
        """
        self.model = model
        self.db = PredictionDatabase(db_path)

        # Load configuration
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file) as f:
                self.config = yaml.safe_load(f)
        else:
            print(f"⚠️ Config not found: {config_path}, using defaults")
            self.config = self._get_default_config()

        # Error thresholds (config or override)
        ol_config = self.config['online_learning']
        self.error_threshold_high = error_threshold_high or ol_config['error_thresholds']['high']
        self.error_threshold_medium = error_threshold_medium or ol_config['error_thresholds']['medium']
        self.error_threshold_low = error_threshold_low or ol_config['error_thresholds']['low']

        # Learning rates (config or override)
        self.lr_fast = fast_layer_lr or ol_config['learning_rates']['fast']
        self.lr_medium = medium_layer_lr or ol_config['learning_rates']['medium']
        self.lr_slow = slow_layer_lr or ol_config['learning_rates']['slow']

        # Fusion weight adaptation (config or override)
        fwa_config = ol_config['fusion_weight_adaptation']
        self.fusion_adaptation_rate = fusion_weight_adaptation_rate or fwa_config['adaptation_rate']

        # Track layer accuracies (moving average)
        self.fast_accuracy_ma = 0.5
        self.medium_accuracy_ma = 0.5
        self.slow_accuracy_ma = 0.5
        self.accuracy_ma_alpha = fwa_config['accuracy_smoothing_alpha']

        # DAILY UPDATE CAPS (prevents catastrophic forgetting)
        self.max_updates_per_day = ol_config['max_updates_per_day']
        self.updates_today = {'fast': 0, 'medium': 0, 'slow': 0}
        self.last_reset_date = datetime.now().date()

        # WEIGHTED SCORING (profit-focused update decisions)
        self.weighted_scoring_config = ol_config.get('weighted_scoring', {})
        self.use_weighted_scoring = self.weighted_scoring_config.get('enabled', True)

        # RE-ANCHORING (prevents long-term drift)
        self.reanchor_config = self.config.get('reanchoring', {})
        if self.reanchor_config.get('enabled', True):
            self.original_weights = copy.deepcopy(model.state_dict())
            self.baseline_val_loss = None  # Set after first validation
            self.last_reanchor_check = datetime.now()
        else:
            self.original_weights = None

    def _get_default_config(self) -> Dict:
        """Return default configuration if YAML not found."""
        return {
            'online_learning': {
                'max_updates_per_day': {'fast': 20, 'medium': 5, 'slow': 2},
                'learning_rates': {'fast': 0.0001, 'medium': 0.00005, 'slow': 0.00001},
                'error_thresholds': {'high': 2.0, 'medium': 1.5, 'low': 1.0},
                'weighted_scoring': {
                    'enabled': True,
                    'no_hit_band_penalty': 3.0,
                    'no_hit_target_penalty': 2.0,
                    'overshoot_penalty': 1.5,
                    'loss_penalty': 2.0,
                    'high_error_penalty': 1.0,
                    'score_thresholds': {
                        'update_all_layers': 5.0,
                        'update_fast_medium': 3.0,
                        'update_fast_only': 1.5
                    }
                },
                'fusion_weight_adaptation': {
                    'enabled': True,
                    'adaptation_rate': 0.01,
                    'accuracy_smoothing_alpha': 0.1
                }
            },
            'reanchoring': {
                'enabled': True,
                'check_interval_days': 7,
                'performance_degradation_threshold': 1.2,
                'drift_threshold': 0.15,
                'blend_ratio': 0.1,
                'min_validations': 50
            },
            'validation': {
                'band_tolerance': 0.02,
                'stop_loss_multiplier': 2.0,
                'max_hold_time': {'fast': 120, 'medium': 480, 'slow': 1440}
            }
        }

    def predict_with_tracking(
        self,
        x: torch.Tensor,
        current_price: float,
        timestamp: datetime,
        features_df=None,
        market_state: Optional[torch.Tensor] = None,
        news_vec: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[Dict[str, Any], int]:
        """
        Make prediction and log it to database for validation.

        Args:
            x: Input features [1, 200, 299]
            current_price: Current TSLA price
            timestamp: Current timestamp
            features_df: Optional features DataFrame (for database logging)
            market_state: Optional market state [1, 12]
            news_vec: Optional news embeddings [1, 768]

        Returns:
            prediction_dict: Prediction dictionary
            prediction_id: Database ID for tracking
        """
        # Make prediction
        pred_dict = self.model.predict(x, market_state, news_vec)

        # Calculate validation time (30 mins for fast layer)
        validation_time = timestamp + timedelta(minutes=30)

        # Log to database
        prediction_data = {
            'timestamp': timestamp,
            'target_timestamp': validation_time,
            'model_timeframe': '1min_hierarchical',
            'model_type': 'hierarchical',
            'predicted_high': pred_dict['predicted_high'],
            'predicted_low': pred_dict['predicted_low'],
            'confidence': pred_dict['confidence'],
            'current_price': current_price,
            'validation_time': validation_time,
            'validated': False,
            'error_triggered_update': False,
            # Layer-specific predictions
            'fast_pred_high': pred_dict['fast_pred_high'],
            'fast_pred_low': pred_dict['fast_pred_low'],
            'fast_pred_conf': pred_dict['fast_pred_conf'],
            'medium_pred_high': pred_dict['medium_pred_high'],
            'medium_pred_low': pred_dict['medium_pred_low'],
            'medium_pred_conf': pred_dict['medium_pred_conf'],
            'slow_pred_high': pred_dict['slow_pred_high'],
            'slow_pred_low': pred_dict['slow_pred_low'],
            'slow_pred_conf': pred_dict['slow_pred_conf'],
            # Fusion weights
            'fusion_weights': str(pred_dict['fusion_weights'])  # JSON string
        }

        prediction_id = self.db.log_prediction(**prediction_data)

        return pred_dict, prediction_id

    def compute_outcome_metrics(
        self,
        pred_record: Dict,
        actual_high: float,
        actual_low: float,
        price_sequence: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute sophisticated outcome metrics for profit-focused validation.

        Args:
            pred_record: Prediction record from database
            actual_high: Actual high (%)
            actual_low: Actual low (%)
            price_sequence: Optional sequence of prices during horizon (for hit_target analysis)

        Returns:
            outcome_metrics: Dict with:
                - error_high/error_low: MSE errors
                - hit_band: Did price enter predicted band?
                - hit_target_before_stop: Reached target before stop?
                - overshoot_ratio: How far outside band?
                - realized_return_pct: Actual P&L following plan
        """
        # Basic errors
        error_high = abs(pred_record['predicted_high'] - actual_high)
        error_low = abs(pred_record['predicted_low'] - actual_low)

        # Predicted band (absolute prices)
        current_price = pred_record['current_price']
        pred_high_price = current_price * (1 + pred_record['predicted_high'] / 100)
        pred_low_price = current_price * (1 + pred_record['predicted_low'] / 100)

        # Actual band (absolute prices)
        actual_high_price = current_price * (1 + actual_high / 100)
        actual_low_price = current_price * (1 + actual_low / 100)

        # Metric 1: Hit Band (did price enter predicted range?)
        # Conservative check: actual range overlaps predicted range
        hit_band = (actual_low_price <= pred_high_price and actual_high_price >= pred_low_price)

        # Metric 2: Overshoot Ratio (how far outside band?)
        pred_range = abs(pred_high_price - pred_low_price)
        if pred_range > 0:
            overshoot_high_amt = max(0, actual_high_price - pred_high_price)
            overshoot_low_amt = max(0, pred_low_price - actual_low_price)
            overshoot_ratio = (overshoot_high_amt + overshoot_low_amt) / pred_range
        else:
            overshoot_ratio = 0.0

        # Metric 3: Hit Target Before Stop (requires price sequence)
        if price_sequence is not None:
            # Calculate stop price (2% below predicted low or 2x ATR)
            stop_price = pred_low_price * 0.98  # Simple 2% stop

            hit_target_before_stop = self._check_sequential_target(
                price_sequence, current_price, pred_high_price, stop_price
            )
        else:
            # Approximation without sequence: did we hit target and not stop?
            hit_target_before_stop = (actual_high_price >= pred_high_price and
                                      actual_low_price >= pred_low_price * 0.98)

        # Metric 4: Realized Return (simulate trade execution)
        if price_sequence is not None:
            realized_return = self._simulate_trade_return(
                price_sequence, current_price, pred_high_price, pred_low_price * 0.98
            )
        else:
            # Approximation: assume hit high if above target, hit low if below
            if actual_high >= pred_record['predicted_high']:
                realized_return = pred_record['predicted_high']  # Hit target
            elif actual_low < pred_record['predicted_low'] * 0.98:
                realized_return = pred_record['predicted_low'] * 0.98 - actual_low  # Hit stop
            else:
                realized_return = actual_high  # Approximate with actual high

        return {
            'error_high': error_high,
            'error_low': error_low,
            'avg_error': (error_high + error_low) / 2,
            'hit_band': hit_band,
            'hit_target_before_stop': hit_target_before_stop,
            'overshoot_ratio': overshoot_ratio,
            'realized_return_pct': realized_return
        }

    def _check_sequential_target(
        self, prices: np.ndarray, entry: float, target: float, stop: float
    ) -> bool:
        """Check if target hit before stop in price sequence."""
        for price in prices:
            if price >= target:
                return True  # Hit target first
            if price <= stop:
                return False  # Hit stop first
        return False

    def _simulate_trade_return(
        self, prices: np.ndarray, entry: float, target: float, stop: float
    ) -> float:
        """Simulate trade and return realized %."""
        for price in prices:
            if price >= target:
                return (target - entry) / entry * 100
            if price <= stop:
                return (stop - entry) / entry * 100
        return (prices[-1] - entry) / entry * 100

    def should_update_weighted(self, outcome_metrics: Dict) -> List[str]:
        """
        Determine which layers to update using weighted scoring (profit-focused).

        Args:
            outcome_metrics: Dict from compute_outcome_metrics()

        Returns:
            layers_to_update: List of layer names to update
        """
        if not self.use_weighted_scoring:
            # Fallback to simple threshold logic
            avg_error = outcome_metrics['avg_error']
            if avg_error > self.error_threshold_high:
                return ['fast', 'medium', 'slow']
            elif avg_error > self.error_threshold_medium:
                return ['fast', 'medium']
            elif avg_error > self.error_threshold_low:
                return ['fast']
            else:
                return []

        # Weighted scoring
        score = 0.0
        ws = self.weighted_scoring_config

        # Heavily penalize: Never touched predicted band
        if not outcome_metrics['hit_band']:
            score += ws.get('no_hit_band_penalty', 3.0)

        # Penalize: Didn't work as trade (target before stop)
        if not outcome_metrics['hit_target_before_stop']:
            score += ws.get('no_hit_target_penalty', 2.0)

        # Penalize: Badly misjudged range (overshoot)
        if outcome_metrics['overshoot_ratio'] > 0.5:
            score += ws.get('overshoot_penalty', 1.5)

        # Penalize: Lost money (realized return)
        if outcome_metrics['realized_return_pct'] < -2.0:
            score += ws.get('loss_penalty', 2.0)

        # Penalize: High MSE error
        if outcome_metrics['avg_error'] > 2.0:
            score += ws.get('high_error_penalty', 1.0)

        # Map score to layers
        thresholds = ws.get('score_thresholds', {})
        if score > thresholds.get('update_all_layers', 5.0):
            return ['fast', 'medium', 'slow']
        elif score > thresholds.get('update_fast_medium', 3.0):
            return ['fast', 'medium']
        elif score > thresholds.get('update_fast_only', 1.5):
            return ['fast']
        else:
            return []  # Don't update - prediction was good enough

    def validate_and_update(
        self,
        prediction_id: int,
        actual_high: float,
        actual_low: float,
        price_sequence: Optional[np.ndarray] = None,
        features_df=None
    ) -> Dict[str, Any]:
        """
        Validate prediction with sophisticated metrics and trigger update if needed.

        Args:
            prediction_id: Database prediction ID
            actual_high: Actual high price (percentage)
            actual_low: Actual low price (percentage)
            price_sequence: Optional tick-by-tick prices during horizon (for sequential analysis)
            features_df: Optional features DataFrame (for re-feeding)

        Returns:
            update_info: Dict with error stats, outcome metrics, and update details
        """
        # Get prediction from database
        pred_record = self.db.get_prediction(prediction_id)

        if pred_record is None:
            return {'error': 'Prediction not found'}

        # Compute sophisticated outcome metrics
        outcome_metrics = self.compute_outcome_metrics(
            pred_record, actual_high, actual_low, price_sequence
        )

        # Update database with actuals
        self.db.update_actuals(
            prediction_id,
            actual_high=actual_high,
            actual_low=actual_low
        )

        # Determine layers to update (profit-focused weighted scoring)
        layers_to_update = self.should_update_weighted(outcome_metrics)
        triggered_update = len(layers_to_update) > 0

        # Perform online update if needed
        if triggered_update and features_df is not None:
            # Re-extract features and perform update
            # (In practice, features should be cached or re-loaded)
            update_result = self._perform_online_update(
                pred_record,
                actual_high,
                actual_low,
                layers_to_update
            )

            # Mark as updated in database
            self.db.execute_query(
                "UPDATE predictions SET error_triggered_update = TRUE WHERE id = ?",
                (prediction_id,)
            )

            # Log update to online_updates table (if exists)
            update_info = {
                'prediction_id': prediction_id,
                'error_high': error_high,
                'error_low': error_low,
                'avg_error': avg_error,
                'triggered_update': True,
                'layers_updated': layers_to_update,
                'learning_rates': {
                    'fast': self.lr_fast if 'fast' in layers_to_update else 0,
                    'medium': self.lr_medium if 'medium' in layers_to_update else 0,
                    'slow': self.lr_slow if 'slow' in layers_to_update else 0
                }
            }
        else:
            update_info = {
                'prediction_id': prediction_id,
                'error_high': error_high,
                'error_low': error_low,
                'avg_error': avg_error,
                'triggered_update': False,
                'layers_updated': []
            }

        # Update layer accuracy tracking
        self._update_layer_accuracies(pred_record, actual_high, actual_low)

        # Adapt fusion weights based on layer performance
        self._adapt_fusion_weights()

        return update_info

    def _perform_online_update(
        self,
        pred_record: Dict,
        actual_high: float,
        actual_low: float,
        layers: List[str]
    ) -> Dict[str, Any]:
        """
        Perform online weight update on specified layers (with daily caps).

        Args:
            pred_record: Prediction record from database
            actual_high: Actual high (percentage)
            actual_low: Actual low (percentage)
            layers: List of layers to update ['fast', 'medium', 'slow']

        Returns:
            result: Update result dict with actually_updated layers
        """
        # Reset daily counters if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            print(f"📅 New day - resetting update counters")
            print(f"   Previous day totals: Fast={self.updates_today['fast']}, Medium={self.updates_today['medium']}, Slow={self.updates_today['slow']}")
            self.updates_today = {'fast': 0, 'medium': 0, 'slow': 0}
            self.last_reset_date = current_date

        # Create target tensor
        y = torch.tensor([[actual_high, actual_low]], dtype=torch.float32).to(self.model.device_type)

        # Note: In production, we'd need to re-load the input features
        # For now, use cached inputs from model
        if self.model.last_fast_input is None:
            return {'status': 'no_cached_input', 'layers': [], 'actually_updated': []}

        x = self.model.last_fast_input
        actually_updated = []

        # Update each layer (with daily cap checking)
        for layer in layers:
            # Check daily cap
            if self.updates_today[layer] >= self.max_updates_per_day[layer]:
                print(f"⚠️ Daily update cap reached for {layer} layer ({self.max_updates_per_day[layer]} updates)")
                continue

            # Get learning rate
            lr = {
                'fast': self.lr_fast,
                'medium': self.lr_medium,
                'slow': self.lr_slow
            }[layer]

            # Perform update
            self.model.update_online(x, y, lr=lr, layer=layer)

            # Increment counter
            self.updates_today[layer] += 1
            actually_updated.append(layer)

            print(f"  ✓ Updated {layer} layer (LR={lr:.6f}, updates today: {self.updates_today[layer]}/{self.max_updates_per_day[layer]})")

        return {
            'status': 'updated' if actually_updated else 'caps_reached',
            'layers': layers,
            'actually_updated': actually_updated,
            'updates_today': dict(self.updates_today)
        }

    def propagate_error_up(
        self,
        error_high: float,
        error_low: float,
        layer_source: str
    ):
        """
        Propagate error from lower layer to higher layers.

        When fast layer has high error, medium/slow should learn too.

        Args:
            error_high: Error in high prediction (%)
            error_low: Error in low prediction (%)
            layer_source: Which layer had the error ('fast', 'medium', 'slow')
        """
        avg_error = (abs(error_high) + abs(error_low)) / 2

        if layer_source == 'fast' and avg_error > self.error_threshold_high:
            # Fast layer error → update medium with weighted error
            if self.model.last_medium_input is not None:
                # Weight the error (50% for medium, 30% for slow)
                weighted_error_medium = avg_error * 0.5
                weighted_error_slow = avg_error * 0.3

                # Create adjusted targets
                y_medium = torch.tensor(
                    [[error_high * 0.5, error_low * 0.5]],
                    dtype=torch.float32
                ).to(self.model.device_type)

                y_slow = torch.tensor(
                    [[error_high * 0.3, error_low * 0.3]],
                    dtype=torch.float32
                ).to(self.model.device_type)

                # Update medium layer
                self.model.update_online(
                    self.model.last_fast_input,
                    y_medium,
                    lr=self.lr_medium * 0.5,
                    layer='medium'
                )

                # Update slow layer
                self.model.update_online(
                    self.model.last_fast_input,
                    y_slow,
                    lr=self.lr_slow * 0.3,
                    layer='slow'
                )

        elif layer_source == 'medium' and avg_error > self.error_threshold_medium:
            # Medium layer error → update slow
            if self.model.last_slow_input is not None:
                y_slow = torch.tensor(
                    [[error_high * 0.5, error_low * 0.5]],
                    dtype=torch.float32
                ).to(self.model.device_type)

                self.model.update_online(
                    self.model.last_fast_input,
                    y_slow,
                    lr=self.lr_slow * 0.5,
                    layer='slow'
                )

    def _update_layer_accuracies(
        self,
        pred_record: Dict,
        actual_high: float,
        actual_low: float
    ):
        """
        Update moving average of layer accuracies.

        Args:
            pred_record: Prediction record with layer predictions
            actual_high: Actual high (%)
            actual_low: Actual low (%)
        """
        # Calculate layer-specific errors
        fast_error = (
            abs(pred_record['fast_pred_high'] - actual_high) +
            abs(pred_record['fast_pred_low'] - actual_low)
        ) / 2

        medium_error = (
            abs(pred_record['medium_pred_high'] - actual_high) +
            abs(pred_record['medium_pred_low'] - actual_low)
        ) / 2

        slow_error = (
            abs(pred_record['slow_pred_high'] - actual_high) +
            abs(pred_record['slow_pred_low'] - actual_low)
        ) / 2

        # Convert errors to accuracy (1 - normalized_error)
        # Normalize by 10% (assume errors > 10% are considered 0 accuracy)
        fast_acc = max(0, 1 - fast_error / 10.0)
        medium_acc = max(0, 1 - medium_error / 10.0)
        slow_acc = max(0, 1 - slow_error / 10.0)

        # Update moving averages
        alpha = self.accuracy_ma_alpha
        self.fast_accuracy_ma = alpha * fast_acc + (1 - alpha) * self.fast_accuracy_ma
        self.medium_accuracy_ma = alpha * medium_acc + (1 - alpha) * self.medium_accuracy_ma
        self.slow_accuracy_ma = alpha * slow_acc + (1 - alpha) * self.slow_accuracy_ma

    def _adapt_fusion_weights(self):
        """
        Adapt fusion weights based on layer accuracy.

        Better-performing layers get higher weights.
        """
        # Get current fusion weights
        current_weights = self.model.fusion_weights.data.cpu().numpy()

        # Target weights based on accuracy
        accuracy_sum = (
            self.fast_accuracy_ma +
            self.medium_accuracy_ma +
            self.slow_accuracy_ma
        )

        if accuracy_sum > 0:
            target_weights = np.array([
                self.fast_accuracy_ma / accuracy_sum,
                self.medium_accuracy_ma / accuracy_sum,
                self.slow_accuracy_ma / accuracy_sum
            ])

            # Smoothly adapt towards target
            new_weights = (
                current_weights * (1 - self.fusion_adaptation_rate) +
                target_weights * self.fusion_adaptation_rate
            )

            # Normalize
            new_weights = new_weights / new_weights.sum()

            # Update model
            self.model.fusion_weights.data = torch.tensor(
                new_weights,
                dtype=torch.float32,
                device=self.model.device_type
            )

    def get_layer_stats(self) -> Dict[str, float]:
        """
        Get current layer accuracy statistics.

        Returns:
            stats: Dict with layer accuracies and weights
        """
        weights = self.model.fusion_weights.data.cpu().numpy()

        return {
            'fast_accuracy': self.fast_accuracy_ma,
            'medium_accuracy': self.medium_accuracy_ma,
            'slow_accuracy': self.slow_accuracy_ma,
            'fast_weight': weights[0],
            'medium_weight': weights[1],
            'slow_weight': weights[2]
        }

    def check_and_reanchor(self, force: bool = False) -> Dict[str, Any]:
        """
        Check for performance degradation and re-anchor to original weights if needed.

        Only re-anchors if:
        - Recent validation loss > baseline * threshold (performance degraded)
        - Weight drift > threshold (weights drifted too far)
        - Enough validations collected (min_validations)

        Args:
            force: Force re-anchoring check regardless of interval

        Returns:
            result: Dict with re-anchoring status and metrics
        """
        if not self.reanchor_config.get('enabled', True):
            return {'status': 'disabled'}

        if self.original_weights is None:
            return {'status': 'no_original_weights'}

        # Check interval
        days_since_check = (datetime.now() - self.last_reanchor_check).days
        check_interval = self.reanchor_config.get('check_interval_days', 7)

        if not force and days_since_check < check_interval:
            return {
                'status': 'waiting',
                'days_until_check': check_interval - days_since_check
            }

        self.last_reanchor_check = datetime.now()

        # Get recent validation loss (last 7 days)
        # Note: Requires database method - for now, placeholder
        try:
            recent_val_loss = self._get_recent_validation_loss(days=7)
        except:
            # If can't get validation loss, skip re-anchoring
            return {'status': 'no_validation_data'}

        # Set baseline on first check
        if self.baseline_val_loss is None:
            self.baseline_val_loss = recent_val_loss
            return {
                'status': 'baseline_set',
                'baseline_val_loss': recent_val_loss
            }

        # Check for performance degradation
        degradation_ratio = recent_val_loss / self.baseline_val_loss
        degradation_threshold = self.reanchor_config.get('performance_degradation_threshold', 1.2)

        print(f"📊 Re-anchoring check:")
        print(f"   Recent val loss: {recent_val_loss:.4f}")
        print(f"   Baseline val loss: {self.baseline_val_loss:.4f}")
        print(f"   Degradation ratio: {degradation_ratio:.2f}x")

        if degradation_ratio > degradation_threshold:
            print(f"⚠️ Performance degraded by {(degradation_ratio - 1) * 100:.1f}%")

            # Calculate weight drift
            drift = self._calculate_weight_drift()
            drift_threshold = self.reanchor_config.get('drift_threshold', 0.15)

            print(f"   Weight drift: {drift:.4f} (threshold: {drift_threshold})")

            if drift > drift_threshold:
                print(f"🔄 Re-anchoring to original weights...")
                self._blend_with_original()

                # Reset baseline (expect improvement)
                self.baseline_val_loss = recent_val_loss * 0.95  # Expect 5% improvement

                return {
                    'status': 're_anchored',
                    'degradation_ratio': degradation_ratio,
                    'drift': drift,
                    'new_baseline': self.baseline_val_loss
                }
            else:
                print(f"✓ Drift within threshold - no re-anchoring needed")
                return {
                    'status': 'degraded_but_low_drift',
                    'degradation_ratio': degradation_ratio,
                    'drift': drift
                }
        else:
            print(f"✅ Performance stable (ratio: {degradation_ratio:.2f})")
            return {
                'status': 'stable',
                'degradation_ratio': degradation_ratio
            }

    def _calculate_weight_drift(self) -> float:
        """
        Calculate L2 distance from original weights (RMS drift).

        Returns:
            drift: Root mean square drift from original weights
        """
        if self.original_weights is None:
            return 0.0

        current = self.model.state_dict()
        drift_sum = 0.0
        total_params = 0

        for key in current.keys():
            if 'weight' in key or 'bias' in key:
                diff = current[key] - self.original_weights[key]
                drift_sum += (diff ** 2).sum().item()
                total_params += diff.numel()

        if total_params == 0:
            return 0.0

        return (drift_sum / total_params) ** 0.5  # RMS drift

    def _blend_with_original(self):
        """
        Blend current weights with original weights.

        Uses blend_ratio from config (default: 10% original, 90% current).
        """
        blend_ratio = self.reanchor_config.get('blend_ratio', 0.1)
        current = self.model.state_dict()

        blended = {
            key: (1 - blend_ratio) * current[key] + blend_ratio * self.original_weights[key]
            for key in current.keys()
        }

        self.model.load_state_dict(blended)
        print(f"   ✓ Blended weights: {blend_ratio * 100:.0f}% original, {(1 - blend_ratio) * 100:.0f}% current")

    def _get_recent_validation_loss(self, days: int = 7) -> float:
        """
        Get average validation loss from recent predictions.

        Args:
            days: Number of days to look back

        Returns:
            avg_loss: Average validation loss (MSE)
        """
        # Query database for recent validated predictions
        # This is a placeholder - would need to implement in PredictionDatabase
        cutoff_date = datetime.now() - timedelta(days=days)

        # Placeholder: return dummy value for now
        # TODO: Implement in PredictionDatabase
        return 3.0  # Dummy value
