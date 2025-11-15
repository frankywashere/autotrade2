"""
Multi-Scale Ensemble System with Meta-LNN Coach.

This module provides the main interface for multi-scale predictions:
1. Loads multiple timeframe-specific LNN sub-models
2. Gets predictions from each sub-model
3. Combines predictions using Meta-LNN coach
4. Optionally incorporates news sentiment (LFM2-based)

Two operational modes:
- backtest_no_news: News disabled (for backtesting)
- live_with_news: News enabled (for live trading)
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from src.ml.model import LNNTradingModel, LSTMTradingModel
from src.ml.meta_models import MetaLNN, MetaLNNWithModalityDropout, MetaFTTransformer, calculate_market_state
from src.ml.news_encoder import NewsEncoder
from src.ml.fetch_news import get_news_window
from src.ml.events import CombinedEventsHandler
from src.ml.hierarchical_model import load_hierarchical_model


class MultiScaleEnsemble:
    """
    Multi-scale ensemble system combining predictions from multiple timeframes.

    Architecture:
    - Sub-models: Separate LNN/LSTM for each timeframe (15min, 1hour, 4hour, daily)
    - Meta-model: MetaLNN coach that learns adaptive combination
    - News encoder: Optional LFM2-based news sentiment (for live mode)

    Modes:
    - backtest_no_news: News disabled (zeros), for backtesting
    - live_with_news: News enabled, fetches and encodes headlines
    """

    def __init__(self,
                 model_paths: Dict[str, str],
                 meta_model_path: str = None,
                 mode: str = 'backtest_no_news',
                 device: str = 'cpu',
                 events_handler: Optional[CombinedEventsHandler] = None):
        """
        Initialize multi-scale ensemble OR hierarchical model.

        Args:
            model_paths: Dict mapping timeframe → model path
                         For ensemble: {'15min': 'models/lnn_15min.pth', '1hour': ...}
                         For hierarchical: {'hierarchical': 'models/hierarchical_lnn.pth'}
            meta_model_path: Path to meta-model checkpoint (ensemble only, optional for hierarchical)
            mode: 'backtest_no_news' or 'live_with_news'
            device: 'cpu', 'cuda', or 'mps'
            events_handler: CombinedEventsHandler instance (for market state features)
        """
        self.mode = mode
        self.device = torch.device(device)
        self.events_handler = events_handler

        # Check if hierarchical mode
        self.is_hierarchical = 'hierarchical' in model_paths

        if self.is_hierarchical:
            # HIERARCHICAL MODE
            print("=" * 70)
            print("HIERARCHICAL LNN MODE")
            print("=" * 70)
            self._init_hierarchical(model_paths['hierarchical'])
            return

        # ENSEMBLE MODE (original behavior)
        # Validate inputs
        if len(model_paths) < 2:
            raise ValueError("Need at least 2 sub-models for ensemble")

        # Load sub-models
        print("Loading sub-models...")
        self.sub_models = {}
        self.timeframes = []

        for timeframe, model_path in model_paths.items():
            print(f"  Loading {timeframe} model from {model_path}...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

                # Extract model config
                model_type = checkpoint.get('model_type', 'LNN')
                input_size = checkpoint.get('input_size')
                hidden_size = checkpoint.get('hidden_size', 128)

                # Create model
                if model_type == 'LNN':
                    model = LNNTradingModel(input_size, hidden_size)
                else:
                    model = LSTMTradingModel(input_size, hidden_size)

                # Load weights
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                model.eval()

                self.sub_models[timeframe] = model
                self.timeframes.append(timeframe)

                print(f"    ✓ Loaded {model_type} model ({input_size} features → {hidden_size} hidden)")

            except Exception as e:
                print(f"    ✗ Error loading {timeframe} model: {e}")
                raise

        print(f"\n  ✓ Loaded {len(self.sub_models)} sub-models: {self.timeframes}")

        # Load meta-model
        print(f"\nLoading meta-model from {meta_model_path}...")
        try:
            meta_checkpoint = torch.load(meta_model_path, map_location=self.device, weights_only=False)

            meta_model_type = meta_checkpoint.get('model_type', 'meta_lnn')
            num_submodels = len(self.sub_models)

            # Create meta-model
            if meta_model_type == 'meta_lnn':
                self.meta_model = MetaLNN(num_submodels=num_submodels)
            elif meta_model_type == 'meta_lnn_dropout':
                self.meta_model = MetaLNNWithModalityDropout(num_submodels=num_submodels)
            else:
                self.meta_model = MetaFTTransformer(num_submodels=num_submodels)

            # Load weights
            self.meta_model.load_state_dict(meta_checkpoint['model_state_dict'])
            self.meta_model = self.meta_model.to(self.device)
            self.meta_model.eval()

            print(f"  ✓ Loaded {meta_model_type} meta-model")

        except Exception as e:
            print(f"  ✗ Error loading meta-model: {e}")
            raise

        # Load news encoder (if live mode)
        if mode == 'live_with_news':
            print("\nLoading news encoder (LFM2)...")
            self.news_encoder = NewsEncoder(mode='live_with_news', device=device)
        else:
            self.news_encoder = NewsEncoder(mode='backtest_no_news', device=device)

        print("\n" + "=" * 70)
        print("✅ ENSEMBLE INITIALIZED")
        print("=" * 70)
        print(f"  Mode: {mode}")
        print(f"  Sub-models: {len(self.sub_models)}")
        print(f"  Timeframes: {', '.join(self.timeframes)}")
        print(f"  Device: {device}")
        print()

    def _init_hierarchical(self, model_path: str):
        """
        Initialize hierarchical LNN model.

        Args:
            model_path: Path to hierarchical model checkpoint
        """
        print(f"Loading hierarchical model from {model_path}...")

        try:
            # Load hierarchical model
            self.hierarchical_model = load_hierarchical_model(model_path, str(self.device))
            self.hierarchical_model.eval()

            print(f"  ✓ Loaded HierarchicalLNN model")
            print(f"  ✓ Input size: 299 features")
            print(f"  ✓ Hidden size: 128")
            print(f"  ✓ Layers: Fast (1-min) → Medium (5-min) → Slow (1-hour)")

        except Exception as e:
            print(f"  ✗ Error loading hierarchical model: {e}")
            raise

        # Load news encoder (if live mode)
        if self.mode == 'live_with_news':
            print("\nLoading news encoder (LFM2)...")
            self.news_encoder = NewsEncoder(mode='live_with_news', device=str(self.device))
        else:
            self.news_encoder = NewsEncoder(mode='backtest_no_news', device=str(self.device))

        print("\n" + "=" * 70)
        print("✅ HIERARCHICAL LNN INITIALIZED")
        print("=" * 70)
        print(f"  Mode: {self.mode}")
        print(f"  Model type: Hierarchical LNN (3-layer)")
        print(f"  Device: {self.device}")
        print()

    def predict(self,
                data: Dict[str, torch.Tensor],
                features_df: pd.DataFrame,
                current_idx: int,
                timestamp: Optional[datetime] = None) -> Dict:
        """
        Make prediction (ensemble or hierarchical).

        Args:
            data: Dict mapping timeframe → input tensor
                  For ensemble: {'15min': tensor([seq_len, features]), '1hour': ...}
                  For hierarchical: {'1min': tensor([200, 299])}
            features_df: Market data DataFrame for calculating market state
            current_idx: Current index in features_df
            timestamp: Current timestamp (for news retrieval)

        Returns:
            Dictionary with:
            - predicted_high, predicted_low, confidence: Final prediction
            - sub_predictions: Dict of sub-model predictions (ensemble) or layer predictions (hierarchical)
        """
        # HIERARCHICAL MODE
        if self.is_hierarchical:
            return self._predict_hierarchical(data, features_df, current_idx, timestamp)

        # ENSEMBLE MODE (original behavior)
        # Get predictions from all sub-models
        sub_predictions = {}
        subpreds_list = []

        for timeframe in self.timeframes:
            if timeframe not in data:
                raise ValueError(f"Missing data for timeframe: {timeframe}")

            # Get sub-model prediction
            model = self.sub_models[timeframe]
            x = data[timeframe].unsqueeze(0).to(self.device)  # Add batch dimension

            with torch.no_grad():
                pred = model.predict(x)

            sub_predictions[timeframe] = {
                'predicted_high': pred['predicted_high'][0],
                'predicted_low': pred['predicted_low'][0],
                'confidence': pred['confidence'][0]
            }

            # Collect for meta-model input
            subpreds_list.append([
                pred['predicted_high'][0],
                pred['predicted_low'][0],
                pred['confidence'][0]
            ])

        # Convert to tensor: [1, num_submodels, 3]
        subpreds = torch.tensor([subpreds_list], dtype=torch.float32).to(self.device)

        # Calculate market state
        market_state = calculate_market_state(
            features_df,
            current_idx,
            self.events_handler
        )
        market_state = market_state.unsqueeze(0).to(self.device)  # [1, 12]

        # Get news embeddings (if live mode)
        if self.mode == 'live_with_news' and timestamp is not None:
            # Fetch news from window
            news_articles = get_news_window(timestamp, lookback_minutes=120)
            headlines = [article['title'] for article in news_articles]

            # Encode news
            news_vec, news_mask = self.news_encoder.encode_headlines(headlines, timestamp)
            news_vec = news_vec.unsqueeze(0).to(self.device)  # [1, 768]
            news_mask = torch.tensor([[news_mask]], dtype=torch.float32).to(self.device)  # [1, 1]
        else:
            # Backtest mode: zeros
            news_vec = torch.zeros(1, 768, dtype=torch.float32).to(self.device)
            news_mask = torch.zeros(1, 1, dtype=torch.float32).to(self.device)

        # Meta-model prediction
        with torch.no_grad():
            pred_high, pred_low, pred_conf, _ = self.meta_model(
                subpreds, market_state, news_vec, news_mask
            )

        # Get current price for percentage → absolute conversion
        current_price = float(features_df['tsla_close'].iloc[current_idx])

        # Return results
        return {
            'predicted_high': float(pred_high[0].cpu()),
            'predicted_low': float(pred_low[0].cpu()),
            'confidence': float(pred_conf[0].cpu()),
            'current_price': current_price,  # Needed for percentage → absolute conversion
            'sub_predictions': sub_predictions,  # For logging and analysis
            'mode': self.mode,
            'news_enabled': self.mode == 'live_with_news' and news_mask[0, 0].item() > 0
        }

    def _predict_hierarchical(self,
                              data: Dict[str, torch.Tensor],
                              features_df: pd.DataFrame,
                              current_idx: int,
                              timestamp: Optional[datetime] = None) -> Dict:
        """
        Make hierarchical prediction.

        Args:
            data: Dict with '1min' key containing [200, 299] tensor
            features_df: Features DataFrame
            current_idx: Current index
            timestamp: Optional timestamp

        Returns:
            Prediction dict with layer-specific predictions
        """
        # Get 1-min data
        if '1min' not in data:
            raise ValueError("Hierarchical mode requires '1min' key in data dict")

        x = data['1min'].unsqueeze(0).to(self.device)  # [1, 200, 299]

        # Calculate market state
        market_state = calculate_market_state(
            features_df,
            current_idx,
            self.events_handler
        )
        market_state = market_state.unsqueeze(0).to(self.device)  # [1, 12]

        # Get news embeddings (if live mode)
        if self.mode == 'live_with_news' and timestamp is not None:
            news_articles = get_news_window(timestamp, lookback_minutes=120)
            headlines = [article['title'] for article in news_articles]
            news_vec, news_mask = self.news_encoder.encode_headlines(headlines, timestamp)
            news_vec = news_vec.unsqueeze(0).to(self.device)  # [1, 768]
            news_mask = torch.tensor([[news_mask]], dtype=torch.float32).to(self.device)  # [1, 1]
        else:
            news_vec = torch.zeros(1, 768, dtype=torch.float32).to(self.device)
            news_mask = torch.zeros(1, 1, dtype=torch.float32).to(self.device)

        # Hierarchical prediction
        with torch.no_grad():
            pred_dict = self.hierarchical_model.predict(
                x, market_state, news_vec, news_mask
            )

        # Get current price
        current_price = float(features_df['tsla_close'].iloc[current_idx])

        # Extract layer predictions for logging
        layer_predictions = {
            'fast': {
                'predicted_high': pred_dict['fast_pred_high'],
                'predicted_low': pred_dict['fast_pred_low'],
                'confidence': pred_dict['fast_pred_conf']
            },
            'medium': {
                'predicted_high': pred_dict['medium_pred_high'],
                'predicted_low': pred_dict['medium_pred_low'],
                'confidence': pred_dict['medium_pred_conf']
            },
            'slow': {
                'predicted_high': pred_dict['slow_pred_high'],
                'predicted_low': pred_dict['slow_pred_low'],
                'confidence': pred_dict['slow_pred_conf']
            }
        }

        # Return in ensemble-compatible format
        return {
            'predicted_high': pred_dict['predicted_high'],
            'predicted_low': pred_dict['predicted_low'],
            'confidence': pred_dict['confidence'],
            'current_price': current_price,
            'sub_predictions': layer_predictions,  # Layer predictions instead of timeframe predictions
            'fusion_weights': pred_dict['fusion_weights'],
            'mode': self.mode,
            'model_type': 'hierarchical',
            'news_enabled': self.mode == 'live_with_news' and news_mask[0, 0].item() > 0
        }

    def predict_batch(self,
                      data_batch: List[Dict[str, torch.Tensor]],
                      features_df: pd.DataFrame,
                      indices: List[int],
                      timestamps: Optional[List[datetime]] = None) -> List[Dict]:
        """
        Make predictions for a batch of samples.

        Args:
            data_batch: List of data dicts (one per sample)
            features_df: Market data DataFrame
            indices: List of indices in features_df
            timestamps: List of timestamps (for news retrieval)

        Returns:
            List of prediction dictionaries
        """
        predictions = []

        for i, (data, idx) in enumerate(zip(data_batch, indices)):
            timestamp = timestamps[i] if timestamps else None
            pred = self.predict(data, features_df, idx, timestamp)
            predictions.append(pred)

        return predictions


def load_ensemble(mode: str = 'backtest_no_news',
                  device: str = 'cpu',
                  models_dir: str = 'models',
                  events_csv: Optional[str] = None) -> MultiScaleEnsemble:
    """
    Convenience function to load ensemble with default paths.

    Args:
        mode: 'backtest_no_news' or 'live_with_news'
        device: 'cpu', 'cuda', or 'mps'
        models_dir: Directory containing model checkpoints
        events_csv: Path to events CSV (optional)

    Returns:
        MultiScaleEnsemble instance
    """
    models_dir = Path(models_dir)

    # Default model paths
    model_paths = {
        '15min': str(models_dir / 'lnn_15min.pth'),
        '1hour': str(models_dir / 'lnn_1hour.pth'),
        '4hour': str(models_dir / 'lnn_4hour.pth'),
        'daily': str(models_dir / 'lnn_daily.pth')
    }

    meta_model_path = str(models_dir / 'meta_lnn.pth')

    # Load events handler
    events_handler = None
    if events_csv:
        events_handler = CombinedEventsHandler(events_csv)

    # Create ensemble
    ensemble = MultiScaleEnsemble(
        model_paths=model_paths,
        meta_model_path=meta_model_path,
        mode=mode,
        device=device,
        events_handler=events_handler
    )

    return ensemble


__all__ = [
    'MultiScaleEnsemble',
    'load_ensemble'
]
