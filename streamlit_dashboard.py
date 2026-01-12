#!/usr/bin/env python3
"""
v7 Channel Prediction Dashboard - Streamlit Version

A web-based dashboard for real-time channel analysis and break predictions.
Uses the v7 HierarchicalCfCModel for multi-timeframe predictions.

Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import sys

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import v7 components
from v7.models import create_model
from v7.models.hierarchical_cfc import HierarchicalCfCModel
from v7.core.channel import detect_channel, Channel, Direction, detect_channels_multi_window, STANDARD_WINDOWS

# Try to import end-to-end window model components
try:
    from v7.models.end_to_end_window_model import EndToEndWindowModel, create_end_to_end_model
    END_TO_END_AVAILABLE = True
except ImportError:
    END_TO_END_AVAILABLE = False
    EndToEndWindowModel = None

# Try to import live data module
try:
    from v7.data.live import fetch_live_data, LiveDataResult
    LIVE_DATA_AVAILABLE = True
except ImportError:
    LIVE_DATA_AVAILABLE = False

# =============================================================================
# Configuration
# =============================================================================

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_DIR = PROJECT_ROOT / "data"

# Import canonical timeframes from v7.core - these must match the model's 11 timeframes
from v7.core.timeframe import TIMEFRAMES, RESAMPLE_RULES

# Display-friendly names for UI (model uses 'daily', UI shows '1d', etc.)
TIMEFRAME_DISPLAY_NAMES = {
    '5min': '5min', '15min': '15min', '30min': '30min',
    '1h': '1h', '2h': '2h', '3h': '3h', '4h': '4h',
    'daily': '1d', 'weekly': '1w', 'monthly': '1M', '3month': '3M'
}

# Architecture inference mapping: (hidden_dim, cfc_layer0_dim) -> cfc_units
# Note: cfc_layer0_dim is implementation-dependent (AutoNCP internal wiring).
# Fallback to hidden_dim * 1.5 works for standard presets but may fail for custom configs.
CFC_UNITS_MAP = {
    # hidden_dim=64 (Quick Start preset, cfc_units=96)
    (64, 20): 96,
    (64, 39): 96,
    # hidden_dim=128 (Standard preset, cfc_units=192)
    (128, 39): 192,
    (128, 77): 256,
    (128, 154): 384,
    # hidden_dim=256 (Full Training preset, cfc_units=384)
    (256, 39): 320,
    (256, 77): 384,
    (256, 154): 512,
}

# =============================================================================
# Model Loading Functions
# =============================================================================

@st.cache_resource
def find_checkpoints() -> List[Dict]:
    """Find all available model checkpoints."""
    checkpoints = []

    if not CHECKPOINTS_DIR.exists():
        return checkpoints

    def extract_metrics(path: Path) -> Dict:
        """Extract training metrics from a checkpoint file."""
        metrics = {
            'best_val_loss': None,
            'best_epoch': None,
            'direction_acc': None,
            'next_channel_acc': None,
            'modified_time': path.stat().st_mtime
        }
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                metrics['best_val_loss'] = checkpoint.get('best_val_metric')
                metrics['best_epoch'] = checkpoint.get('epoch')
                val_metrics_history = checkpoint.get('val_metrics_history')
                if val_metrics_history and len(val_metrics_history) > 0:
                    last_entry = val_metrics_history[-1]
                    if isinstance(last_entry, dict):
                        metrics['direction_acc'] = last_entry.get('direction_acc')
                        metrics['next_channel_acc'] = last_entry.get('next_channel_acc')
                        metrics['duration_mae'] = last_entry.get('duration_mae')
                        metrics['duration_rmse'] = last_entry.get('duration_rmse')
                        # Extract per-TF MAEs if available
                        for tf_name in ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly', '3month']:
                            mae_key = f'duration_mae_{tf_name}'
                            if mae_key in last_entry:
                                metrics[mae_key] = last_entry.get(mae_key)
        except Exception:
            pass
        return metrics

    # Check for root best_model.pt
    root_model = CHECKPOINTS_DIR / "best_model.pt"
    if root_model.exists():
        config = extract_config_from_checkpoint(root_model)
        metrics = extract_metrics(root_model)
        checkpoints.append({
            "name": "best_model (root)",
            "path": root_model,
            "config": config,
            "size_mb": root_model.stat().st_size / (1024 * 1024),
            **metrics
        })

    # Check for walk-forward window models
    for window_dir in sorted(CHECKPOINTS_DIR.glob("window_*")):
        if window_dir.is_dir():
            model_path = window_dir / "best_model.pt"
            if model_path.exists():
                config = extract_config_from_checkpoint(model_path)
                metrics = extract_metrics(model_path)
                checkpoints.append({
                    "name": f"{window_dir.name}/best_model",
                    "path": model_path,
                    "config": config,
                    "size_mb": model_path.stat().st_size / (1024 * 1024),
                    **metrics
                })

    # Also scan runs directory for models
    # Structure: runs/TIMESTAMP_name/windows/window_1/best_model.pt (walk-forward)
    #            runs/TIMESTAMP_name/windows/best_model.pt (standard)
    if RUNS_DIR.exists():
        for run_dir in sorted(RUNS_DIR.iterdir()):
            if run_dir.is_dir() and not run_dir.name.startswith('.'):
                windows_dir = run_dir / "windows"
                if not windows_dir.exists():
                    continue

                # Check for walk-forward window subdirectories: runs/*/windows/window_*/best_model.pt
                for window_dir in sorted(windows_dir.glob("window_*")):
                    if window_dir.is_dir():
                        model_path = window_dir / "best_model.pt"
                        if model_path.exists():
                            config = extract_config_from_checkpoint(model_path)
                            metrics = extract_metrics(model_path)
                            checkpoints.append({
                                "name": f"runs/{run_dir.name}/windows/{window_dir.name}/best_model",
                                "path": model_path,
                                "config": config,
                                "size_mb": model_path.stat().st_size / (1024 * 1024),
                                "source": "run",
                                "run_id": run_dir.name,
                                **metrics
                            })

                # Check for standard (non-walk-forward): runs/*/windows/best_model.pt
                root_model = windows_dir / "best_model.pt"
                if root_model.exists():
                    config = extract_config_from_checkpoint(root_model)
                    metrics = extract_metrics(root_model)
                    checkpoints.append({
                        "name": f"runs/{run_dir.name}/windows/best_model",
                        "path": root_model,
                        "config": config,
                        "size_mb": root_model.stat().st_size / (1024 * 1024),
                        "source": "run",
                        "run_id": run_dir.name,
                        **metrics
                    })

    return checkpoints


@st.cache_data(ttl=60)  # Cache for 60 seconds, auto-refresh
def load_experiments_index() -> List[dict]:
    """Load experiments from runs/experiments_index.json."""
    index_path = RUNS_DIR / "experiments_index.json"
    if not index_path.exists():
        return []
    try:
        with open(index_path) as f:
            return json.load(f)
    except Exception:
        return []


def _load_training_config_json(checkpoint_path: Path) -> Optional[Dict]:
    """Try to load training_config.json from checkpoint directory or parent."""
    # Check in same directory as checkpoint
    config_paths = [
        checkpoint_path.parent / "training_config.json",
        checkpoint_path.parent.parent / "training_config.json",  # For window_X/best_model.pt
    ]
    for config_path in config_paths:
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config_data = json.load(f)
                if 'model' in config_data:
                    return config_data['model']
            except Exception:
                pass
    return None


def extract_config_from_checkpoint(checkpoint_path: Path, checkpoint: Optional[Dict] = None) -> Optional[Dict]:
    """Extract config from checkpoint file.

    Config sources (in priority order):
    1. TrainingConfig.model_kwargs embedded in checkpoint (best - exact match)
    2. training_config.json file in checkpoint directory (reliable - saved at training time)
    3. Heuristic inference from tensor shapes (fallback - may not match custom configs)

    Args:
        checkpoint_path: Path to checkpoint file
        checkpoint: Optional pre-loaded checkpoint dict to avoid redundant loading
    """
    try:
        if checkpoint is None:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        config_dict = {}

        if isinstance(checkpoint, dict):
            # === SOURCE 1: Embedded config in checkpoint ===
            if 'config' in checkpoint:
                config = checkpoint['config']

                # Try TrainingConfig.model_kwargs (the actual attribute name)
                if hasattr(config, 'model_kwargs') and config.model_kwargs:
                    config_dict['model'] = dict(config.model_kwargs)
                    config_dict['model']['_source'] = 'checkpoint_model_kwargs'

                # Try config.model (for compatibility with other formats)
                elif hasattr(config, 'model') and config.model:
                    model_cfg = config.model
                    if hasattr(model_cfg, '__dict__'):
                        config_dict['model'] = dict(model_cfg.__dict__)
                    elif isinstance(model_cfg, dict):
                        config_dict['model'] = dict(model_cfg)
                    if 'model' in config_dict:
                        config_dict['model']['_source'] = 'checkpoint_config_model'

                # Try dict-style access
                elif isinstance(config, dict) and 'model' in config:
                    config_dict['model'] = dict(config['model'])
                    config_dict['model']['_source'] = 'checkpoint_dict'

            # === SOURCE 2: training_config.json file ===
            if 'model' not in config_dict or not config_dict['model']:
                json_config = _load_training_config_json(checkpoint_path)
                if json_config:
                    config_dict['model'] = dict(json_config)
                    config_dict['model']['_source'] = 'training_config_json'

            # === SOURCE 3: Heuristic inference from tensor shapes (last resort) ===
            if 'model' not in config_dict or not config_dict['model']:
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                hidden_dim = 64
                cfc_layer0_dim = None

                for key, tensor in state_dict.items():
                    if 'tf_branches.0.input_proj.weight' in key:
                        hidden_dim = tensor.shape[0]
                    if 'tf_branches.0.cfc.rnn_cell.layer_0.ff1.weight' in key:
                        cfc_layer0_dim = tensor.shape[0]

                cfc_units = CFC_UNITS_MAP.get(
                    (hidden_dim, cfc_layer0_dim),
                    int(hidden_dim * 1.5)
                )

                # Heuristic: detect SE-blocks from state_dict keys
                use_se_blocks = any('tf_branches.0.se_block' in k for k in state_dict.keys())

                config_dict['model'] = {
                    'hidden_dim': hidden_dim,
                    'cfc_units': cfc_units,
                    'num_attention_heads': 4 if hidden_dim <= 128 else 8,
                    'dropout': 0.1,
                    'use_se_blocks': use_se_blocks,
                    'se_reduction_ratio': 4,  # Default value when inferred
                    '_source': 'heuristic_inference',
                    '_inferred': True,
                    '_inference_note': f"Config inferred from tensor shapes (may not match training)"
                }

            # === Post-processing: Extract SE-block params if not already present ===
            # Also add heuristic detection for SE-blocks from state_dict
            if 'model' in config_dict and config_dict['model']:
                model_cfg = config_dict['model']
                state_dict = checkpoint.get('model_state_dict', checkpoint)

                # Extract use_se_blocks and se_reduction_ratio from config if present
                # (they may already be there from SOURCE 1 or 2)
                if 'use_se_blocks' not in model_cfg:
                    # Heuristic fallback: check if SE-block exists in state_dict
                    has_se_block = any('tf_branches.0.se_block' in k for k in state_dict.keys())
                    model_cfg['use_se_blocks'] = has_se_block

                if 'se_reduction_ratio' not in model_cfg:
                    model_cfg['se_reduction_ratio'] = 4  # Default value

                # === NEW CONFIG FIELDS ===
                # TCN options
                if 'use_tcn' not in model_cfg:
                    # Heuristic: check if TCN layers exist in state_dict
                    has_tcn = any('tcn' in k.lower() for k in state_dict.keys())
                    model_cfg['use_tcn'] = has_tcn
                if 'tcn_channels' not in model_cfg:
                    model_cfg['tcn_channels'] = None  # Unknown
                if 'tcn_kernel_size' not in model_cfg:
                    model_cfg['tcn_kernel_size'] = None
                if 'tcn_layers' not in model_cfg:
                    model_cfg['tcn_layers'] = None

                # Multi-resolution options
                if 'use_multi_resolution' not in model_cfg:
                    has_multi_res = any('multi_res' in k.lower() or 'resolution' in k.lower() for k in state_dict.keys())
                    model_cfg['use_multi_resolution'] = has_multi_res
                if 'resolution_levels' not in model_cfg:
                    model_cfg['resolution_levels'] = None

                # Gradient balancing options (training config, not model weights)
                if 'gradient_balancing' not in model_cfg:
                    model_cfg['gradient_balancing'] = None
                if 'gradnorm_alpha' not in model_cfg:
                    model_cfg['gradnorm_alpha'] = None

                # Two-stage training options (training config)
                if 'two_stage_training' not in model_cfg:
                    model_cfg['two_stage_training'] = None
                if 'stage1_epochs' not in model_cfg:
                    model_cfg['stage1_epochs'] = None
                if 'stage1_task' not in model_cfg:
                    model_cfg['stage1_task'] = None

                # Loss types
                if 'duration_loss_type' not in model_cfg:
                    model_cfg['duration_loss_type'] = None
                if 'direction_loss_type' not in model_cfg:
                    model_cfg['direction_loss_type'] = None

                # EndToEndWindowModel-specific parameters
                if 'window_embed_dim' not in model_cfg:
                    # Heuristic: infer from embed_to_features layer shape
                    if 'embed_to_features.weight' in state_dict:
                        model_cfg['window_embed_dim'] = state_dict['embed_to_features.weight'].shape[1]
                    else:
                        model_cfg['window_embed_dim'] = 128  # Default
                if 'temperature' not in model_cfg:
                    model_cfg['temperature'] = 1.0  # Default
                if 'use_gumbel' not in model_cfg:
                    model_cfg['use_gumbel'] = False  # Default
                if 'num_windows' not in model_cfg:
                    model_cfg['num_windows'] = 8  # Default (STANDARD_WINDOWS)
                if 'num_hazard_bins' not in model_cfg:
                    # Heuristic: infer from hazard_head layer if present
                    # Check for per-TF hazard heads first (separate heads architecture)
                    hazard_key = 'hierarchical_model.per_tf_duration_heads.0.hazard_head.weight'
                    if hazard_key in state_dict:
                        model_cfg['num_hazard_bins'] = state_dict[hazard_key].shape[0]
                    else:
                        # Check for shared heads architecture
                        hazard_key_shared = 'hierarchical_model.per_tf_duration_head.hazard_head.weight'
                        if hazard_key_shared in state_dict:
                            model_cfg['num_hazard_bins'] = state_dict[hazard_key_shared].shape[0]
                        else:
                            model_cfg['num_hazard_bins'] = 0  # Default (disabled)

                if 'max_duration' not in model_cfg:
                    if hasattr(config, 'max_duration'):
                        model_cfg['max_duration'] = config.max_duration
                    else:
                        model_cfg['max_duration'] = 100.0

        return config_dict if config_dict else None
    except Exception as e:
        st.warning(f"Error extracting config: {e}")
        return None


@st.cache_resource
def load_model(checkpoint_path: str) -> Optional[torch.nn.Module]:
    """Load model from checkpoint with proper architecture detection.

    Supports both HierarchicalCfCModel (standard) and EndToEndWindowModel (Phase 2b).
    Model type is auto-detected from checkpoint keys.
    """
    try:
        path = Path(checkpoint_path)

        # Load checkpoint once and reuse for both config extraction and model loading
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

        # Detect model type from state_dict keys
        is_end_to_end = any('window_encoder' in k or 'window_selector' in k for k in state_dict.keys())

        # Extract config from the already-loaded checkpoint (pass checkpoint to avoid re-loading)
        config = extract_config_from_checkpoint(path, checkpoint=checkpoint)

        if config and 'model' in config:
            model_config = config['model']
            hidden_dim = model_config.get('hidden_dim', 64)
            cfc_units = model_config.get('cfc_units', 96)
            num_heads = model_config.get('num_attention_heads', 4)
            dropout = model_config.get('dropout', 0.1)
            shared_heads = model_config.get('shared_heads', True)  # Default to shared for backward compat
            use_se_blocks = model_config.get('use_se_blocks', False)
            se_reduction_ratio = model_config.get('se_reduction_ratio', 4)

            # Show config source status
            source = model_config.get('_source', 'unknown')
            if source == 'checkpoint_model_kwargs':
                st.success(f"✓ Config from checkpoint: hidden_dim={hidden_dim}, cfc_units={cfc_units}")
            elif source == 'training_config_json':
                st.info(f"✓ Config from training_config.json: hidden_dim={hidden_dim}, cfc_units={cfc_units}")
            elif source == 'heuristic_inference':
                st.warning(f"⚠️ Config inferred from tensor shapes: hidden_dim={hidden_dim}, cfc_units={cfc_units}")
            if use_se_blocks:
                st.info(f"SE-blocks enabled (reduction_ratio={se_reduction_ratio})")

            # Display new config options if enabled
            use_tcn = model_config.get('use_tcn', False)
            if use_tcn:
                tcn_channels = model_config.get('tcn_channels', 'N/A')
                tcn_kernel = model_config.get('tcn_kernel_size', 'N/A')
                tcn_layers = model_config.get('tcn_layers', 'N/A')
                st.info(f"TCN enabled (channels={tcn_channels}, kernel={tcn_kernel}, layers={tcn_layers})")

            use_multi_res = model_config.get('use_multi_resolution', False)
            if use_multi_res:
                res_levels = model_config.get('resolution_levels', 'N/A')
                st.info(f"Multi-resolution enabled (levels={res_levels})")

            num_hazard_bins = model_config.get('num_hazard_bins', 0)
            if num_hazard_bins > 0:
                st.info(f"Survival loss enabled (hazard_bins={num_hazard_bins})")

            grad_balancing = model_config.get('gradient_balancing', False)
            if grad_balancing:
                gradnorm_alpha = model_config.get('gradnorm_alpha', 'N/A')
                st.info(f"Gradient balancing enabled (alpha={gradnorm_alpha})")

            two_stage = model_config.get('two_stage_training', False)
            if two_stage:
                stage1_epochs = model_config.get('stage1_epochs', 'N/A')
                stage1_task = model_config.get('stage1_task', 'N/A')
                st.info(f"Two-stage training (stage1: {stage1_epochs} epochs, task={stage1_task})")

            duration_loss = model_config.get('duration_loss_type')
            direction_loss = model_config.get('direction_loss_type')
            if duration_loss or direction_loss:
                loss_info = []
                if duration_loss:
                    loss_info.append(f"duration={duration_loss}")
                if direction_loss:
                    loss_info.append(f"direction={direction_loss}")
                st.info(f"Loss types: {', '.join(loss_info)}")
        else:
            hidden_dim, cfc_units, num_heads, dropout, shared_heads = 64, 96, 4, 0.1, True
            use_se_blocks, se_reduction_ratio = False, 4
            st.warning("⚠️ No config found, using defaults: hidden_dim=64, cfc_units=96")

        # Infer shared_heads from state_dict keys (overrides config if separate heads detected)
        has_separate_heads = any('per_tf_duration_heads' in k for k in state_dict.keys())
        if has_separate_heads:
            shared_heads = False
            st.info("Detected separate per-TF heads architecture")

        # Create appropriate model type
        if is_end_to_end and END_TO_END_AVAILABLE:
            st.info("🔄 Detected EndToEndWindowModel (Phase 2b)")
            model = create_end_to_end_model(
                feature_dim=776,
                window_embed_dim=model_config.get('window_embed_dim', 128),
                num_windows=model_config.get('num_windows', 8),
                temperature=model_config.get('temperature', 1.0),
                use_gumbel=model_config.get('use_gumbel', False),
                hidden_dim=hidden_dim,
                cfc_units=cfc_units,
                num_attention_heads=num_heads,
                dropout=dropout,
                shared_heads=shared_heads,
                use_se_blocks=use_se_blocks,
                se_reduction_ratio=se_reduction_ratio,
                use_tcn=model_config.get('use_tcn', False),
                tcn_channels=model_config.get('tcn_channels', 64),
                tcn_kernel_size=model_config.get('tcn_kernel_size', 3),
                tcn_layers=model_config.get('tcn_layers', 2),
                use_multi_resolution=model_config.get('use_multi_resolution', False),
                resolution_levels=model_config.get('resolution_levels', 3),
                num_hazard_bins=model_config.get('num_hazard_bins', 0),
                max_duration=model_config.get('max_duration', 100.0),
                device='cpu'
            )
        else:
            if is_end_to_end:
                st.warning("⚠️ EndToEndWindowModel checkpoint detected but module not available")
            model = create_model(
                hidden_dim=hidden_dim,
                cfc_units=cfc_units,
                num_attention_heads=num_heads,
                dropout=dropout,
                shared_heads=shared_heads,
                use_se_blocks=use_se_blocks,
                se_reduction_ratio=se_reduction_ratio,
                use_tcn=model_config.get('use_tcn', False),
                tcn_channels=model_config.get('tcn_channels', 64),
                tcn_kernel_size=model_config.get('tcn_kernel_size', 3),
                tcn_layers=model_config.get('tcn_layers', 2),
                use_multi_resolution=model_config.get('use_multi_resolution', False),
                resolution_levels=model_config.get('resolution_levels', 3),
                num_hazard_bins=model_config.get('num_hazard_bins', 0),
                max_duration=model_config.get('max_duration', 100.0),
                device='cpu'
            )

        # state_dict already loaded above for shared_heads inference
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            st.warning(f"Checkpoint missing {len(incompatible.missing_keys)} keys")
        if incompatible.unexpected_keys:
            st.warning(f"Checkpoint has {len(incompatible.unexpected_keys)} unexpected keys")
        model.eval()

        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


# =============================================================================
# Data Loading Functions
# =============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_market_data(use_live: bool = True, lookback_days: int = 90) -> Dict:
    """Load market data from live source or CSV files."""
    result = {
        "tsla_df": None,
        "spy_df": None,
        "vix_df": None,
        "status": "UNKNOWN",
        "timestamp": None,
        "error": None
    }

    # Try live data first
    if use_live and LIVE_DATA_AVAILABLE:
        try:
            live_result = fetch_live_data(lookback_days=lookback_days)
            if live_result.tsla_df is not None and len(live_result.tsla_df) > 0:
                result["tsla_df"] = live_result.tsla_df
                result["spy_df"] = live_result.spy_df
                result["vix_df"] = live_result.vix_df
                # Status might be string or enum - handle both
                status = live_result.status
                result["status"] = status.name if hasattr(status, 'name') else str(status)
                result["timestamp"] = live_result.timestamp
                return result
            else:
                result["error"] = "Live data returned empty TSLA dataframe"
        except Exception as e:
            result["error"] = f"Live data failed: {e}"
    elif use_live and not LIVE_DATA_AVAILABLE:
        result["error"] = "Live data module not available (import failed)"

    # Fall back to CSV files (use correct file names matching actual files)
    try:
        # Primary paths (actual file names in data/)
        tsla_path = DATA_DIR / "TSLA_1min.csv"
        spy_path = DATA_DIR / "SPY_1min.csv"
        vix_path = DATA_DIR / "VIX_History.csv"

        # Calculate cutoff date to respect lookback_days (same as live data path)
        cutoff = datetime.now() - timedelta(days=lookback_days)

        # Load TSLA data (CSV has 'timestamp' column)
        tsla_df = pd.read_csv(tsla_path, parse_dates=['timestamp'], index_col='timestamp')
        tsla_df = tsla_df[tsla_df.index >= cutoff]  # Filter by lookback_days

        # Load SPY data
        if spy_path.exists():
            spy_df = pd.read_csv(spy_path, parse_dates=['timestamp'], index_col='timestamp')
            spy_df = spy_df[spy_df.index >= cutoff]  # Filter by lookback_days
        else:
            spy_df = pd.DataFrame()

        # Load VIX data (different format - has DATE column with MM/DD/YYYY format)
        if vix_path.exists():
            vix_df = pd.read_csv(vix_path, parse_dates=['DATE'], index_col='DATE')
            vix_df.columns = [c.lower() for c in vix_df.columns]
            vix_df = vix_df[vix_df.index >= cutoff]  # Filter by lookback_days
        else:
            vix_df = pd.DataFrame()

        # Resample to 5min if needed
        if len(tsla_df) > 0 and tsla_df.index[1] - tsla_df.index[0] < timedelta(minutes=5):
            tsla_df = resample_ohlc(tsla_df, '5min')
            if len(spy_df) > 0:
                spy_df = resample_ohlc(spy_df, '5min')

        result["tsla_df"] = tsla_df
        result["spy_df"] = spy_df
        result["vix_df"] = vix_df
        result["status"] = "HISTORICAL"
        if len(tsla_df) > 0:
            result["timestamp"] = tsla_df.index[-1]
    except Exception as e:
        result["error"] = str(e)

    return result


def resample_ohlc(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample OHLC data to a different timeframe."""
    if df.empty:
        return df

    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }
    if 'volume' in df.columns:
        agg_dict['volume'] = 'sum'

    # Use canonical resample rules from v7.core.timeframe
    resample_rule = RESAMPLE_RULES.get(timeframe, timeframe)
    return df.resample(resample_rule).agg(agg_dict).dropna()


def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to regular trading hours only (9:30 AM - 4:00 PM ET)."""
    if df.empty:
        return df

    try:
        # Use simple time-based filtering (works for tz-naive data)
        hour = df.index.hour
        minute = df.index.minute

        # Trading hours: 9:30 AM to 4:00 PM
        mask = (
            ((hour == 9) & (minute >= 30)) |  # 9:30 AM and later
            ((hour > 9) & (hour < 16)) |      # 10 AM to 3:59 PM
            ((hour == 16) & (minute == 0))    # Exactly 4:00 PM close
        )

        return df[mask]
    except Exception:
        # If filtering fails, return original data
        return df


# =============================================================================
# Channel Detection
# =============================================================================

def detect_all_channels(tsla_df: pd.DataFrame, spy_df: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Detect channels for all timeframes with adaptive window sizes."""
    # Filter to regular trading hours first
    tsla_df = filter_market_hours(tsla_df)
    spy_df = filter_market_hours(spy_df)

    # Adaptive window sizes - shorter windows for higher timeframes
    # Now includes all 11 model timeframes (using canonical names from v7.core.timeframe)
    WINDOW_MAP = {
        '5min': 50,    # 50 bars = 4.2 hours
        '15min': 40,   # 40 bars = 10 hours
        '30min': 30,   # 30 bars = 15 hours
        '1h': 20,      # 20 bars = 20 hours
        '2h': 15,      # 15 bars = 30 hours
        '3h': 12,      # 12 bars = 36 hours
        '4h': 12,      # 12 bars = 48 hours
        'daily': 10,   # 10 bars = 10 days
        'weekly': 10,  # 10 bars = 10 weeks
        'monthly': 10, # 10 bars = 10 months
        '3month': 8,   # 8 bars = 2 years
    }

    tsla_channels = {}
    spy_channels = {}

    for tf in TIMEFRAMES:
        try:
            window = WINDOW_MAP.get(tf, 50)

            if tf == '5min':
                tf_tsla = tsla_df
                tf_spy = spy_df if len(spy_df) > 0 else None
            else:
                tf_tsla = resample_ohlc(tsla_df, tf)
                tf_spy = resample_ohlc(spy_df, tf) if len(spy_df) > 0 else None

            if len(tf_tsla) >= window:
                tsla_channels[tf] = detect_channel(tf_tsla, window=window, min_cycles=0)

            if tf_spy is not None and len(tf_spy) >= window:
                spy_channels[tf] = detect_channel(tf_spy, window=window, min_cycles=0)
        except Exception:
            pass

    return tsla_channels, spy_channels


# =============================================================================
# Predictions
# =============================================================================

def make_predictions(
    tsla_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    vix_df: pd.DataFrame,
    model: torch.nn.Module
) -> Optional[Dict]:
    """Make predictions using the model - returns per-TF predictions.

    Supports both HierarchicalCfCModel (single-window) and EndToEndWindowModel (multi-window).
    """
    if model is None:
        return None

    try:
        # Import feature extraction
        from v7.features.full_features import extract_full_features, features_to_tensor_dict
        from v7.features.feature_ordering import FEATURE_ORDER

        # Check if model is end-to-end (multi-window)
        # Use hasattr() instead of isinstance() for reliability with pickled models
        is_end_to_end = hasattr(model, 'window_encoder') and hasattr(model, 'window_selector')

        if is_end_to_end:
            # Extract features for ALL 8 windows
            per_window_features = []
            window_scores_list = []
            window_valid_list = []

            for window in STANDARD_WINDOWS:  # [10, 20, 30, 40, 50, 60, 70, 80]
                features = extract_full_features(
                    tsla_df=tsla_df,
                    spy_df=spy_df,
                    vix_df=vix_df,
                    window=window,
                    include_history=False
                )

                if features is not None:
                    feature_arrays = features_to_tensor_dict(features)
                    feature_list = [feature_arrays[k] for k in FEATURE_ORDER if k in feature_arrays]
                    feature_array = np.concatenate(feature_list)
                    per_window_features.append(feature_array)

                    # Extract channel quality scores from tsla_window_scores
                    # tsla_window_scores has shape (8, 5) with metrics for all STANDARD_WINDOWS
                    # Metrics order: bounce_count, r_squared, quality_score, alternation_ratio, width_pct
                    if features.tsla_window_scores is not None:
                        window_idx = STANDARD_WINDOWS.index(window)
                        scores = features.tsla_window_scores[window_idx]
                        window_scores_list.append([
                            float(scores[0]),  # bounce_count
                            float(scores[1]),  # r_squared
                            float(scores[2]),  # quality_score
                            float(scores[3]),  # alternation_ratio
                            float(scores[4]),  # width_pct
                        ])
                        window_valid_list.append(True)
                    else:
                        window_scores_list.append([0.0, 0.0, 0.0, 0.0, 0.0])
                        window_valid_list.append(False)
                else:
                    # Use zeros for missing windows
                    per_window_features.append(np.zeros(776, dtype=np.float32))
                    window_scores_list.append([0.0, 0.0, 0.0, 0.0, 0.0])
                    window_valid_list.append(False)

            # Stack into tensors: [1, 8, 776]
            per_window_tensor = torch.from_numpy(np.stack(per_window_features)).float().unsqueeze(0)
            window_scores_tensor = torch.tensor([window_scores_list], dtype=torch.float32)
            window_valid_tensor = torch.tensor([window_valid_list], dtype=torch.bool)

            # Run inference with multi-window input
            with torch.no_grad():
                outputs = model.predict(
                    per_window_tensor,
                    window_scores=window_scores_tensor,
                    window_valid=window_valid_tensor
                )
        else:
            # Standard single-window inference
            features = extract_full_features(
                tsla_df=tsla_df,
                spy_df=spy_df,
                vix_df=vix_df,
                window=50,
                include_history=False  # Faster without history
            )

            if features is None:
                return None

            # Convert FullFeatures to dict of arrays
            feature_arrays = features_to_tensor_dict(features)

            # Concatenate using canonical FEATURE_ORDER
            feature_list = []
            for key in FEATURE_ORDER:
                if key in feature_arrays:
                    feature_list.append(feature_arrays[key])

            # Combine into single array then convert to tensor
            feature_array = np.concatenate(feature_list)
            feature_tensor = torch.from_numpy(feature_array).float().unsqueeze(0)

            # Run inference
            with torch.no_grad():
                outputs = model.predict(feature_tensor)

        # Defensive NaN check - log if detected (indicates upstream data issue)
        import warnings
        if outputs is not None:
            for key in ['per_tf', 'aggregate']:
                if key in outputs:
                    for subkey, value in outputs[key].items():
                        if hasattr(value, 'numpy'):
                            arr = value.numpy() if hasattr(value, 'numpy') else value
                            if not np.isfinite(arr).all():
                                warnings.warn(f"NaN/Inf detected in outputs['{key}']['{subkey}'] - check data sufficiency")

        # Extract per-TF and aggregate predictions
        per_tf = outputs['per_tf']
        agg = outputs['aggregate']
        best_tf_idx = int(outputs['best_tf_idx'][0])

        # TF names for indexing
        TF_NAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', '1d', '1w', '1M', '3M']

        result = {
            # Per-timeframe predictions (all 11 TFs)
            'per_tf': {
                'duration_mean': per_tf['duration_mean'][0].numpy(),  # [11]
                'duration_std': per_tf['duration_std'][0].numpy(),    # [11]
                'direction': per_tf['direction'][0].numpy(),          # [11]
                'direction_probs': per_tf['direction_probs'][0].numpy(),  # [11]
                'next_channel': per_tf['next_channel'][0].numpy(),    # [11]
                'next_channel_probs': per_tf['next_channel_probs'][0].numpy(),  # [11, 3]
                'confidence': per_tf['confidence'][0].numpy(),        # [11]
                'channel_valid': per_tf['channel_valid'],            # [11] bool
            },
            # Best (most confident) timeframe
            'best_tf_idx': best_tf_idx,
            'best_tf_name': TF_NAMES[best_tf_idx],
            # Aggregate (for reference)
            'aggregate': {
                'duration_mean': float(agg['duration_mean'][0, 0]),
                'duration_std': float(agg['duration_std'][0, 0]),
                'direction': int(agg['direction'][0, 0]),
                'direction_probs': agg['direction_probs'][0].numpy(),
                'next_channel': int(agg['next_channel'][0, 0]),
                'next_channel_probs': agg['next_channel_probs'][0].numpy(),
                'confidence': float(agg['confidence'][0, 0]),
                # v9.0.0: Trigger TF predictions
                'trigger_tf': int(agg['trigger_tf'][0, 0]),
                'trigger_tf_probs': agg['trigger_tf_probs'][0].numpy()
            }
        }

        # Add window selection info if available (Phase 2b)
        if is_end_to_end and 'window_selection' in outputs:
            ws = outputs['window_selection']
            result['window_selection'] = {
                'probs': ws['probs'][0].numpy(),
                'selected_idx': int(ws['selected_idx'][0]),
                'confidence': float(ws['confidence'][0])
            }

        return result
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="v7 Channel Predictions",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📊 v7 Channel Prediction Dashboard")

    # Initialize session state
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'model_name' not in st.session_state:
        st.session_state.model_name = None

    # Sidebar - Model Selection
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model selection
        st.subheader("Model Selection")
        checkpoints = find_checkpoints()

        if checkpoints:
            checkpoint_names = [cp['name'] for cp in checkpoints]
            selected_idx = st.selectbox(
                "Select Model",
                range(len(checkpoint_names)),
                format_func=lambda x: checkpoint_names[x]
            )

            selected_cp = checkpoints[selected_idx]

            # Show model info
            st.caption(f"Size: {selected_cp['size_mb']:.1f} MB")
            if selected_cp.get('config', {}).get('model'):
                cfg = selected_cp['config']['model']
                st.caption(f"hidden_dim: {cfg.get('hidden_dim', '?')}")
                st.caption(f"cfc_units: {cfg.get('cfc_units', '?')}")

                # Display new config options in sidebar
                # SE-blocks
                if cfg.get('use_se_blocks'):
                    st.caption(f"SE-blocks: Yes (r={cfg.get('se_reduction_ratio', 4)})")

                # TCN
                if cfg.get('use_tcn'):
                    tcn_info = f"TCN: Yes"
                    if cfg.get('tcn_channels'):
                        tcn_info += f" (ch={cfg.get('tcn_channels')})"
                    st.caption(tcn_info)

                # Multi-resolution
                if cfg.get('use_multi_resolution'):
                    res_levels = cfg.get('resolution_levels', '?')
                    st.caption(f"Multi-res: {res_levels} levels")

                # Gradient balancing
                if cfg.get('gradient_balancing'):
                    alpha = cfg.get('gradnorm_alpha', '?')
                    st.caption(f"Grad bal: alpha={alpha}")

                # Two-stage training
                if cfg.get('two_stage_training'):
                    stage1 = cfg.get('stage1_task', '?')
                    st.caption(f"Two-stage: {stage1}")

                # Loss types (only show if non-default)
                dur_loss = cfg.get('duration_loss_type')
                dir_loss = cfg.get('direction_loss_type')
                if dur_loss or dir_loss:
                    loss_parts = []
                    if dur_loss:
                        loss_parts.append(f"dur={dur_loss}")
                    if dir_loss:
                        loss_parts.append(f"dir={dir_loss}")
                    st.caption(f"Loss: {', '.join(loss_parts)}")

            # Show training metrics if available
            val_loss = selected_cp.get('val_loss')
            if val_loss is not None:
                st.caption(f"Val Loss: {val_loss:.4f}")

            best_epoch = selected_cp.get('best_epoch')
            if best_epoch is not None:
                st.caption(f"Best Epoch: {best_epoch}")

            dir_acc = selected_cp.get('direction_accuracy')
            if dir_acc is not None:
                st.caption(f"Dir Acc: {dir_acc*100:.1f}%")

            next_ch_acc = selected_cp.get('next_channel_accuracy')
            if next_ch_acc is not None:
                st.caption(f"Next Ch Acc: {next_ch_acc*100:.1f}%")

            # Visual indicator for model quality based on direction accuracy
            if dir_acc is not None:
                if dir_acc >= 0.6:
                    st.markdown(":green[Quality: Good]")
                elif dir_acc >= 0.5:
                    st.markdown(":orange[Quality: Fair]")
                else:
                    st.markdown(":red[Quality: Poor]")
                st.progress(min(dir_acc, 1.0))

            # Load button
            if st.button("Load Model", type="primary"):
                with st.spinner("Loading model..."):
                    model = load_model(str(selected_cp['path']))
                    if model is not None:
                        st.session_state.model = model
                        st.session_state.model_name = selected_cp['name']
                        st.success(f"Loaded: {selected_cp['name']}")
                    else:
                        st.error("Failed to load model")
        else:
            st.warning("No checkpoints found in checkpoints/")

        # Show current model
        if st.session_state.model is not None:
            st.success(f"✓ Active: {st.session_state.model_name}")
        else:
            st.warning("No model loaded")

        st.divider()

        # Training Runs section
        st.subheader("Training Runs")
        experiments = load_experiments_index()

        if experiments:
            # Create display names with timestamps
            run_display_names = []
            for exp in experiments:
                name = exp.get('name') or exp.get('run_id', 'Unknown')
                timestamp = exp.get('timestamp', '')
                if timestamp:
                    # Format timestamp for display (show date and time)
                    try:
                        ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        ts_str = ts.strftime('%m/%d %H:%M')
                    except Exception:
                        ts_str = timestamp[:16] if len(timestamp) > 16 else timestamp
                    run_display_names.append(f"{name} ({ts_str})")
                else:
                    run_display_names.append(name)

            selected_run_idx = st.selectbox(
                "Select Run",
                range(len(run_display_names)),
                format_func=lambda x: run_display_names[x],
                key="run_selector"
            )

            selected_exp = experiments[selected_run_idx]

            # Show selected run info
            st.caption(f"Timestamp: {selected_exp.get('timestamp', 'N/A')}")

            # Status
            status = selected_exp.get('status', 'unknown')
            if status == 'completed':
                st.caption(":green[Status: Completed]")
            elif status == 'running':
                st.caption(":orange[Status: Running]")
            else:
                st.caption(f"Status: {status}")

            # Best metrics
            best_val_loss = selected_exp.get('best_val_loss')
            if best_val_loss is not None:
                st.caption(f"Best Val Loss: {best_val_loss:.4f}")

            dir_acc = selected_exp.get('best_direction_acc')
            if dir_acc is not None:
                st.caption(f"Dir Accuracy: {dir_acc*100:.1f}%")

            # Key settings
            settings = selected_exp.get('settings', {})
            if settings:
                se_blocks = settings.get('use_se_blocks', False)
                hidden_dim = settings.get('hidden_dim', 'N/A')
                lr = settings.get('lr', settings.get('learning_rate', 'N/A'))
                batch_size = settings.get('batch_size', 'N/A')

                st.caption(f"SE-blocks: {'Yes' if se_blocks else 'No'}")
                st.caption(f"hidden_dim: {hidden_dim}")
                if lr != 'N/A':
                    st.caption(f"LR: {lr}")
                st.caption(f"Batch: {batch_size}")

            # Load Run's Best Model button
            # run_id contains the directory name (e.g., "20250109_143022_test")
            run_dir = selected_exp.get('run_id', '')
            if run_dir:
                run_path = RUNS_DIR / run_dir
                windows_path = run_path / "windows"
                # Check for best_model.pt in windows/ or windows/window_*/ subdirectories
                best_model_path = None
                best_model_name = ""
                if windows_path.exists():
                    # First check standard training: runs/*/windows/best_model.pt
                    if (windows_path / "best_model.pt").exists():
                        best_model_path = windows_path / "best_model.pt"
                        best_model_name = f"runs/{run_dir}/windows/best_model"
                    else:
                        # Check walk-forward window directories: runs/*/windows/window_*/best_model.pt
                        for window_dir in sorted(windows_path.glob("window_*")):
                            model_path = window_dir / "best_model.pt"
                            if model_path.exists():
                                best_model_path = model_path
                                best_model_name = f"runs/{run_dir}/windows/{window_dir.name}/best_model"
                                break

                if best_model_path and st.button("Load Run's Best Model", key="load_run_model"):
                    with st.spinner("Loading model from run..."):
                        model = load_model(str(best_model_path))
                        if model is not None:
                            st.session_state.model = model
                            st.session_state.model_name = best_model_name
                            st.success(f"Loaded: {st.session_state.model_name}")
                        else:
                            st.error("Failed to load model")
        else:
            st.info("No runs found. Train a model to see runs here.")

        st.divider()

        # Data settings
        st.subheader("Data Settings")

        # Show live data module status
        if LIVE_DATA_AVAILABLE:
            st.caption("✓ Live data module available")
        else:
            st.caption("✗ Live data module not available")

        use_live = st.checkbox("Use Live Data", value=LIVE_DATA_AVAILABLE, disabled=not LIVE_DATA_AVAILABLE)
        lookback_days = st.slider("Lookback Days", 420, 730, 500, help="Model trained with 420-day warmup. Values below 420 produce unreliable predictions.")

        # Data sufficiency warning
        if lookback_days < 420:
            st.warning("⚠️ Lookback below training minimum (420 days). Weekly/monthly timeframe predictions will be unreliable.")

        col_refresh1, col_refresh2 = st.columns(2)
        with col_refresh1:
            if st.button("Refresh Data"):
                st.cache_data.clear()
                st.rerun()
        with col_refresh2:
            if st.button("Refresh Models"):
                # Clear resource cache to rescan checkpoints directory
                st.cache_resource.clear()
                st.rerun()

    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Live Predictions", "📊 Channel Analysis", "ℹ️ Info"])

    # Load data
    data = load_market_data(use_live=use_live, lookback_days=lookback_days)
    st.info(f"📊 Using {lookback_days} days of historical data. Training requires 420+ days for all timeframes.")

    # Tab 1: Live Predictions
    with tab1:
        st.header("Live Predictions")

        if data.get("error"):
            st.error(f"Data Error: {data['error']}")

        if data["tsla_df"] is None or len(data["tsla_df"]) == 0:
            st.warning("No market data available")
        elif st.session_state.model is None:
            st.warning("⚠️ No model loaded. Select a model from the sidebar and click 'Load Model'.")
        else:
            # Make predictions
            predictions = make_predictions(
                data["tsla_df"],
                data["spy_df"],
                data["vix_df"],
                st.session_state.model
            )

            if predictions is None:
                st.error("Failed to generate predictions")
            else:
                # Get most confident timeframe predictions
                best_tf_idx = predictions['best_tf_idx']
                best_tf_name = predictions['best_tf_name']
                per_tf = predictions['per_tf']

                # Extract best TF predictions
                best_conf = per_tf['confidence'][best_tf_idx]
                best_dur = per_tf['duration_mean'][best_tf_idx]
                best_dur_std = per_tf['duration_std'][best_tf_idx]
                best_dir_prob = per_tf['direction_probs'][best_tf_idx]
                best_next_ch = per_tf['next_channel'][best_tf_idx]
                best_next_probs = per_tf['next_channel_probs'][best_tf_idx]

                # Display predictions for most confident timeframe
                st.info(f"**Most Confident Timeframe:** {best_tf_name}")

                # Add channel validity warning
                if 'channel_valid' in per_tf and per_tf['channel_valid'][best_tf_idx] < 0.5:
                    st.warning("⚠️ No valid channel structure detected - prediction based on cross-timeframe context")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Confidence", f"{best_conf:.1%}")

                with col2:
                    direction = "UP ↑" if best_dir_prob > 0.5 else "DOWN ↓"
                    st.metric("Direction", direction, f"{abs(best_dir_prob - 0.5)*200:.0f}%")

                with col3:
                    # Show uncertainty only if available and > 0
                    if best_dur_std and best_dur_std > 0.01:
                        st.metric(f"Duration ({best_tf_name})", f"{best_dur:.1f} bars", f"±{best_dur_std:.1f}")
                    else:
                        st.metric(f"Duration ({best_tf_name})", f"{best_dur:.1f} bars")

                with col4:
                    next_labels = ["DOWN", "SAME", "UP"]
                    st.metric("Next Channel", next_labels[best_next_ch], f"{best_next_probs[best_next_ch]:.0%}")

                # Duration error metrics (if available from training)
                # Get metrics from the selected checkpoint
                checkpoints = find_checkpoints()
                checkpoint_metrics = {}
                for cp in checkpoints:
                    if cp['name'] == st.session_state.model_name:
                        checkpoint_metrics = cp
                        break

                if checkpoint_metrics.get('duration_mae') is not None:
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration MAE", f"{checkpoint_metrics['duration_mae']:.2f} bars",
                                  help="Mean absolute error from validation")
                    with col2:
                        st.metric("Duration RMSE", f"{checkpoint_metrics.get('duration_rmse', 0):.2f} bars",
                                  help="Root mean squared error from validation")
                    with col3:
                        # Show uncertainty only if available and > 0
                        if best_dur_std and best_dur_std > 0.01:
                            st.metric("Pred Uncertainty", f"{best_dur_std:.2f} bars",
                                      help="Model's estimated prediction uncertainty")
                        else:
                            st.metric("Pred Uncertainty", "Not available",
                                      help="Model's estimated prediction uncertainty")

                    # Per-TF MAE breakdown
                    with st.expander("Per-Timeframe Duration MAE"):
                        tf_maes = {
                            '5min': checkpoint_metrics.get('duration_mae_5min'),
                            '15min': checkpoint_metrics.get('duration_mae_15min'),
                            '30min': checkpoint_metrics.get('duration_mae_30min'),
                            '1h': checkpoint_metrics.get('duration_mae_1h'),
                            '2h': checkpoint_metrics.get('duration_mae_2h'),
                            '3h': checkpoint_metrics.get('duration_mae_3h'),
                            '4h': checkpoint_metrics.get('duration_mae_4h'),
                            'daily': checkpoint_metrics.get('duration_mae_daily'),
                            'weekly': checkpoint_metrics.get('duration_mae_weekly'),
                            'monthly': checkpoint_metrics.get('duration_mae_monthly'),
                            '3month': checkpoint_metrics.get('duration_mae_3month'),
                        }

                        # Display as table
                        if any(v is not None for v in tf_maes.values()):
                            mae_df = pd.DataFrame({
                                'Timeframe': list(tf_maes.keys()),
                                'MAE (bars)': [f"{v:.2f}" if v is not None else 'N/A' for v in tf_maes.values()]
                            })
                            st.dataframe(mae_df, hide_index=True)
                        else:
                            st.caption("Per-TF MAEs not available in this checkpoint")

                # Signal interpretation
                st.divider()

                if best_conf > 0.7:
                    if best_dir_prob > 0.6:
                        st.success("🟢 **STRONG LONG SIGNAL** - High confidence bullish break expected")
                    elif best_dir_prob < 0.4:
                        st.error("🔴 **STRONG SHORT SIGNAL** - High confidence bearish break expected")
                    else:
                        st.info("🟡 **NEUTRAL** - Direction unclear despite high confidence")
                elif best_conf > 0.5:
                    st.warning("⚠️ **MODERATE CONFIDENCE** - Consider waiting for stronger signal")
                else:
                    st.info("ℹ️ **LOW CONFIDENCE** - No clear trading signal")

                # Channel Prediction Visualization
                st.divider()
                st.subheader("Channel Prediction")

                try:
                    # Detect channel for the best timeframe
                    tsla_channels, _ = detect_all_channels(data["tsla_df"], data["spy_df"])

                    if best_tf_name in tsla_channels and tsla_channels[best_tf_name].valid:
                        channel = tsla_channels[best_tf_name]
                        predicted_direction = "UP" if best_dir_prob > 0.5 else "DOWN"

                        # Create prediction chart
                        pred_chart = create_prediction_chart(
                            df=data["tsla_df"],
                            channel=channel,
                            timeframe=best_tf_name,
                            symbol="TSLA",
                            predicted_duration=best_dur,
                            predicted_direction=predicted_direction,
                            direction_prob=best_dir_prob if predicted_direction == "UP" else (1 - best_dir_prob),
                            confidence=best_conf
                        )

                        if pred_chart:
                            st.plotly_chart(pred_chart, use_container_width=True)

                            # Show price level summary
                            current_price = data["tsla_df"]['close'].iloc[-1]
                            current_upper = channel.upper_line[-1]
                            current_lower = channel.lower_line[-1]
                            position = channel.position_at()

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Current Price", f"${current_price:.2f}")
                            with col2:
                                st.metric("Upper Bound", f"${current_upper:.2f}", f"{((current_upper/current_price)-1)*100:+.1f}%")
                            with col3:
                                st.metric("Lower Bound", f"${current_lower:.2f}", f"{((current_lower/current_price)-1)*100:+.1f}%")
                            with col4:
                                pos_label = "Near Upper" if position > 0.7 else "Near Lower" if position < 0.3 else "Mid-Channel"
                                st.metric("Position", pos_label, f"{position:.0%}")
                        else:
                            st.info("Chart visualization not available (Plotly not installed)")
                    else:
                        st.info(f"No valid channel detected for {best_tf_name} timeframe")
                except Exception as e:
                    st.warning(f"Could not generate channel visualization: {e}")

                # v9.0.0: Display trigger TF prediction if available
                agg = predictions.get('aggregate', {})
                trigger_tf = agg.get('trigger_tf', 0)
                trigger_tf_probs = agg.get('trigger_tf_probs', None)

                if trigger_tf_probs is not None:
                    TRIGGER_TF_NAMES = [
                        'NO_TRIGGER',
                        '15min↑', '15min↓', '30min↑', '30min↓',
                        '1h↑', '1h↓', '2h↑', '2h↓',
                        '3h↑', '3h↓', '4h↑', '4h↓',
                        'daily↑', 'daily↓', 'weekly↑', 'weekly↓',
                        'monthly↑', 'monthly↓', '3month↑', '3month↓'
                    ]
                    trigger_name = TRIGGER_TF_NAMES[trigger_tf] if trigger_tf < len(TRIGGER_TF_NAMES) else f'Class_{trigger_tf}'
                    trigger_conf = trigger_tf_probs[trigger_tf] if trigger_tf < len(trigger_tf_probs) else 0.0

                    st.divider()
                    st.subheader("Trigger TF Prediction (v9.0.0)")
                    col1, col2 = st.columns(2)
                    with col1:
                        if trigger_tf == 0:
                            st.info(f"**{trigger_name}** (prob: {trigger_conf:.0%})")
                        else:
                            st.warning(f"**{trigger_name}** boundary may cause break (prob: {trigger_conf:.0%})")

                # Window Selection info (Phase 2b - EndToEndWindowModel only)
                if 'window_selection' in predictions:
                    window_sel = predictions['window_selection']
                    st.divider()
                    st.subheader("Window Selection (Phase 2b)")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        selected_idx = window_sel['selected_idx']
                        selected_window = STANDARD_WINDOWS[selected_idx] if selected_idx < len(STANDARD_WINDOWS) else 0
                        st.metric("Selected Window", f"{selected_window} bars")

                    with col2:
                        st.metric("Selection Confidence", f"{window_sel['confidence']:.1%}")

                    with col3:
                        # Show top-2 windows by probability
                        probs = window_sel['probs']
                        sorted_indices = np.argsort(probs)[::-1][:2]
                        top_windows = [f"{STANDARD_WINDOWS[i]}({probs[i]:.0%})" for i in sorted_indices]
                        st.metric("Top Windows", " / ".join(top_windows))

                    # Window probability distribution
                    with st.expander("Window Probability Distribution"):
                        window_df = pd.DataFrame({
                            "Window": [f"{w} bars" for w in STANDARD_WINDOWS],
                            "Probability": [f"{p:.1%}" for p in probs],
                            "Raw": probs
                        })
                        st.dataframe(window_df, hide_index=True)

                # Per-timeframe breakdown table
                st.divider()
                st.subheader("All Timeframe Predictions")

                TF_NAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', '1d', '1w', '1M', '3M']
                tf_rows = []
                for i, tf_name in enumerate(TF_NAMES):
                    is_best = (i == best_tf_idx)

                    # Check channel validity
                    is_valid = per_tf.get('channel_valid', [1]*11)[i] >= 0.5
                    tf_display = f"**{tf_name}**" if is_best else tf_name
                    if not is_valid:
                        tf_display += " ⚠️"  # Warning icon for invalid channels

                    # Format duration with uncertainty only if > 0
                    dur_mean = per_tf['duration_mean'][i]
                    dur_std = per_tf['duration_std'][i]
                    if dur_std and dur_std > 0.01:
                        duration_str = f"{dur_mean:.1f}±{dur_std:.1f}"
                    else:
                        duration_str = f"{dur_mean:.1f}"

                    tf_rows.append({
                        "TF": tf_display,
                        "Confidence": f"{per_tf['confidence'][i]:.1%}",
                        "Duration": duration_str,
                        "Direction": "UP ↑" if per_tf['direction_probs'][i] > 0.5 else "DOWN ↓",
                        "Dir Prob": f"{per_tf['direction_probs'][i]:.0%}",
                        "Next Ch": ["DN", "SAME", "UP"][per_tf['next_channel'][i]],
                    })

                tf_df = pd.DataFrame(tf_rows)
                st.dataframe(tf_df, width='stretch', hide_index=True)

                # Per-timeframe validity warning
                invalid_tfs = [TF_NAMES[i] for i in range(len(TF_NAMES)) if per_tf.get('channel_valid', [1]*11)[i] < 0.5]
                if invalid_tfs:
                    st.warning(f"⚠️ Insufficient data for valid channel detection in: {', '.join(invalid_tfs)}. Predictions for these timeframes are unreliable. Consider using 420+ days of lookback data.")

        # Price info
        if data["tsla_df"] is not None and len(data["tsla_df"]) > 0:
            st.divider()
            st.subheader("Current Prices")

            col1, col2, col3 = st.columns(3)

            with col1:
                tsla_price = data["tsla_df"]['close'].iloc[-1]
                tsla_change = (tsla_price / data["tsla_df"]['close'].iloc[-2] - 1) * 100 if len(data["tsla_df"]) > 1 else 0
                st.metric("TSLA", f"${tsla_price:.2f}", f"{tsla_change:+.2f}%")

            with col2:
                if data["spy_df"] is not None and len(data["spy_df"]) > 0:
                    spy_price = data["spy_df"]['close'].iloc[-1]
                    spy_change = (spy_price / data["spy_df"]['close'].iloc[-2] - 1) * 100 if len(data["spy_df"]) > 1 else 0
                    st.metric("SPY", f"${spy_price:.2f}", f"{spy_change:+.2f}%")

            with col3:
                if data["vix_df"] is not None and len(data["vix_df"]) > 0:
                    vix_value = data["vix_df"]['close'].iloc[-1]
                    st.metric("VIX", f"{vix_value:.2f}")

            st.caption(f"Data Status: {data['status']} | Last Update: {data.get('timestamp', 'Unknown')}")

    # Tab 2: Channel Analysis
    with tab2:
        st.header("Channel Analysis")

        if data["tsla_df"] is None or len(data["tsla_df"]) == 0:
            st.warning("No market data available for channel analysis")
        else:
            tsla_channels, spy_channels = detect_all_channels(
                data["tsla_df"],
                data["spy_df"] if data["spy_df"] is not None else pd.DataFrame()
            )

            # Channel tables
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("TSLA Channels")
                display_channel_table(tsla_channels)

            with col2:
                st.subheader("SPY Channels")
                display_channel_table(spy_channels)

            # Channel visualizations
            st.divider()
            st.subheader("Channel Visualizations")

            # Select timeframe to visualize
            vis_col1, vis_col2 = st.columns([1, 3])
            with vis_col1:
                tf_options = [tf for tf in TIMEFRAMES if tf in tsla_channels and tsla_channels[tf].valid]
                if tf_options:
                    selected_tf = st.selectbox("Timeframe", tf_options, key="viz_tf")
                else:
                    st.warning("No valid channels to visualize")
                    selected_tf = None

            if selected_tf and selected_tf in tsla_channels:
                # Create charts for selected timeframe
                chart_col1, chart_col2 = st.columns(2)

                with chart_col1:
                    display_tf = TIMEFRAME_DISPLAY_NAMES.get(selected_tf, selected_tf)
                    st.write(f"**TSLA {display_tf}**")
                    tsla_chart = create_channel_chart(
                        data["tsla_df"],
                        tsla_channels[selected_tf],
                        selected_tf,
                        "TSLA"
                    )
                    if tsla_chart is not None:
                        st.plotly_chart(tsla_chart, key=f"tsla_{selected_tf}")
                    else:
                        st.warning("Plotly required for visualization. Install: `pip install plotly`")

                with chart_col2:
                    if selected_tf in spy_channels:
                        display_tf = TIMEFRAME_DISPLAY_NAMES.get(selected_tf, selected_tf)
                        st.write(f"**SPY {display_tf}**")
                        spy_chart = create_channel_chart(
                            data["spy_df"],
                            spy_channels[selected_tf],
                            selected_tf,
                            "SPY"
                        )
                        if spy_chart is not None:
                            st.plotly_chart(spy_chart, key=f"spy_{selected_tf}")
                        else:
                            st.warning("Plotly required for visualization. Install: `pip install plotly`")

    # Tab 3: Info
    with tab3:
        st.header("Dashboard Information")

        st.subheader("Model Architecture")
        st.write("""
        The v7 HierarchicalCfCModel uses:
        - **11 Timeframe Branches** (5min → 15min → 30min → 1h → 2h → 3h → 4h → daily → weekly → monthly → 3month)
        - **Closed-form Continuous-time (CfC)** neural networks
        - **Cross-timeframe Attention** for information sharing
        - **4 Prediction Heads**: Duration, Direction, Next Channel, Confidence
        """)

        st.subheader("Feature Set")
        st.write("""
        - **761 input features** per prediction
        - Per-timeframe features (56 features x 11 TFs = 616):
          - TSLA channel features (geometry, bounces, RSI): 35 per TF
          - SPY channel features for cross-asset analysis: 11 per TF
          - Cross-asset containment features: 10 per TF
        - Shared features (145 total):
          - VIX regime and volatility: 6 features
          - TSLA historical patterns: 25 features
          - SPY historical patterns: 25 features
          - Cross-asset alignment: 3 features
          - Event awareness: 46 features
          - Window quality scores: 40 features (8 windows x 5 metrics)
        """)

        st.subheader("Model Comparison")

        # Build comparison data from all checkpoints
        checkpoints = find_checkpoints()
        if checkpoints:
            comparison_rows = []
            for cp in checkpoints:
                # Get model config for new fields
                model_cfg = cp.get('config', {}).get('model', {})

                row = {
                    'Model Name': cp['name'],
                    'Val Loss': None,
                    'Best Epoch': None,
                    'Dir Acc %': None,
                    'Next Ch Acc %': None,
                    'Duration MAE': cp.get('duration_mae'),
                    'SE': 'Yes' if model_cfg.get('use_se_blocks') else 'No',
                    'TCN': 'Yes' if model_cfg.get('use_tcn') else 'No',
                    'Multi-Res': 'Yes' if model_cfg.get('use_multi_resolution') else 'No',
                    'Grad Bal': 'Yes' if model_cfg.get('gradient_balancing') else 'No',
                    'Two-Stage': 'Yes' if model_cfg.get('two_stage_training') else 'No',
                    'Size MB': cp['size_mb']
                }

                # Try to extract metrics from checkpoint
                try:
                    checkpoint = torch.load(cp['path'], map_location='cpu', weights_only=False)
                    if isinstance(checkpoint, dict):
                        # Get best_val_metric (this is the best validation loss)
                        if 'best_val_metric' in checkpoint:
                            row['Val Loss'] = checkpoint['best_val_metric']

                        # Get epoch
                        if 'epoch' in checkpoint:
                            row['Best Epoch'] = checkpoint['epoch']

                        # Get metrics from val_metrics_history
                        val_history = checkpoint.get('val_metrics_history', [])
                        if val_history:
                            # Find the epoch with best (lowest) total loss
                            best_idx = 0
                            best_loss = float('inf')
                            for i, metrics in enumerate(val_history):
                                total = metrics.get('total', float('inf'))
                                if total < best_loss:
                                    best_loss = total
                                    best_idx = i

                            best_metrics = val_history[best_idx]
                            row['Best Epoch'] = best_idx + 1  # epochs are 1-indexed

                            if row['Val Loss'] is None:
                                row['Val Loss'] = best_metrics.get('total')

                            if 'direction_acc' in best_metrics:
                                row['Dir Acc %'] = best_metrics['direction_acc'] * 100

                            if 'next_channel_acc' in best_metrics:
                                row['Next Ch Acc %'] = best_metrics['next_channel_acc'] * 100
                except Exception:
                    pass  # Keep None values for missing metrics

                comparison_rows.append(row)

            if comparison_rows:
                # Create DataFrame
                df = pd.DataFrame(comparison_rows)

                # Sort by Val Loss (ascending) - best model at top
                df = df.sort_values('Val Loss', ascending=True, na_position='last')

                # Add star prefix to best model name (first row after sorting)
                if len(df) > 0 and pd.notna(df.iloc[0]['Val Loss']):
                    df.iloc[0, df.columns.get_loc('Model Name')] = '\u2605 ' + str(df.iloc[0]['Model Name'])

                # Format numeric columns and handle None values
                def format_value(val, decimals=4, is_percentage=False):
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        return '\u2014'  # em-dash
                    if is_percentage:
                        return f'{val:.1f}'
                    return f'{val:.{decimals}f}'

                # Create display DataFrame with formatted values
                display_df = pd.DataFrame({
                    'Model Name': df['Model Name'],
                    'Val Loss': df['Val Loss'].apply(lambda x: format_value(x, 4)),
                    'Best Epoch': df['Best Epoch'].apply(lambda x: format_value(x, 0) if x is None or pd.isna(x) else str(int(x))),
                    'Dir Acc %': df['Dir Acc %'].apply(lambda x: format_value(x, 1, True)),
                    'Next Ch Acc %': df['Next Ch Acc %'].apply(lambda x: format_value(x, 1, True)),
                    'Duration MAE': df['Duration MAE'].apply(lambda x: format_value(x, 2)),
                    'SE': df['SE'],
                    'TCN': df['TCN'],
                    'Multi-Res': df['Multi-Res'],
                    'Grad Bal': df['Grad Bal'],
                    'Two-Stage': df['Two-Stage'],
                    'Size MB': df['Size MB'].apply(lambda x: format_value(x, 1))
                })

                st.dataframe(display_df, width='stretch', hide_index=True)

                # Show expandable details for each checkpoint
                st.subheader("Checkpoint Details")
                for cp in checkpoints:
                    with st.expander(cp['name']):
                        st.write(f"Path: {cp['path']}")
                        st.write(f"Size: {cp['size_mb']:.1f} MB")
                        if cp.get('config'):
                            st.json(cp['config'])
        else:
            st.info("No checkpoints found in checkpoints/")

        # Training Runs Comparison section
        st.subheader("Training Runs Comparison")
        experiments = load_experiments_index()

        if experiments:
            # Build comparison DataFrame
            runs_rows = []
            for exp in experiments:
                settings = exp.get('settings', {})
                runs_rows.append({
                    'Run Name': exp.get('name') or exp.get('run_id', 'Unknown'),
                    'Timestamp': exp.get('timestamp', 'N/A'),
                    'Status': exp.get('status', 'unknown'),
                    'Val Loss': exp.get('best_val_loss'),
                    'Dir Acc%': exp.get('best_direction_acc', 0) * 100 if exp.get('best_direction_acc') else None,
                    'SE': 'Yes' if settings.get('use_se_blocks', False) else 'No',
                    'TCN': 'Yes' if settings.get('use_tcn', False) else 'No',
                    'Multi-Res': 'Yes' if settings.get('use_multi_resolution', False) else 'No',
                    'Grad Bal': 'Yes' if settings.get('gradient_balancing', False) else 'No',
                    'Two-Stage': 'Yes' if settings.get('two_stage_training', False) else 'No',
                    'hidden_dim': settings.get('hidden_dim', 'N/A'),
                    'LR': settings.get('lr', settings.get('learning_rate', 'N/A')),
                    'Batch': settings.get('batch_size', 'N/A'),
                })

            runs_df = pd.DataFrame(runs_rows)

            # Sort by Val Loss ascending (best at top)
            runs_df = runs_df.sort_values('Val Loss', ascending=True, na_position='last')

            # Mark best run with star
            if len(runs_df) > 0 and pd.notna(runs_df.iloc[0]['Val Loss']):
                runs_df.iloc[0, runs_df.columns.get_loc('Run Name')] = '\u2b50 ' + str(runs_df.iloc[0]['Run Name'])

            # Format display DataFrame
            def format_run_value(val, decimals=4, is_percentage=False):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return '\u2014'  # em-dash
                if is_percentage:
                    return f'{val:.1f}'
                if isinstance(val, float):
                    return f'{val:.{decimals}f}'
                return str(val)

            display_runs_df = pd.DataFrame({
                'Run Name': runs_df['Run Name'],
                'Timestamp': runs_df['Timestamp'].apply(lambda x: x[:19] if isinstance(x, str) and len(x) > 19 else x),
                'Status': runs_df['Status'],
                'Val Loss': runs_df['Val Loss'].apply(lambda x: format_run_value(x, 4)),
                'Dir Acc%': runs_df['Dir Acc%'].apply(lambda x: format_run_value(x, 1, True)),
                'SE': runs_df['SE'],
                'TCN': runs_df['TCN'],
                'Multi-Res': runs_df['Multi-Res'],
                'Grad Bal': runs_df['Grad Bal'],
                'Two-Stage': runs_df['Two-Stage'],
                'hidden_dim': runs_df['hidden_dim'],
                'LR': runs_df['LR'],
                'Batch': runs_df['Batch'],
            })

            st.dataframe(display_runs_df, width='stretch', hide_index=True)

            # Expanders for full settings
            st.subheader("Run Details")
            for exp in experiments:
                run_name = exp.get('name') or exp.get('run_id', 'Unknown')
                with st.expander(run_name):
                    st.write(f"**Run Directory:** {exp.get('path', 'N/A')}")
                    st.write(f"**Timestamp:** {exp.get('timestamp', 'N/A')}")
                    st.write(f"**Status:** {exp.get('status', 'N/A')}")

                    best_val_loss = exp.get('best_val_loss')
                    if best_val_loss is not None:
                        st.write(f"**Best Val Loss:** {best_val_loss:.4f}")

                    best_dir_acc = exp.get('best_direction_acc')
                    if best_dir_acc is not None:
                        st.write(f"**Best Direction Accuracy:** {best_dir_acc*100:.1f}%")

                    settings = exp.get('settings', {})
                    if settings:
                        st.write("**Full Settings:**")
                        st.json(settings)
        else:
            st.info("No runs found in experiments_index.json. Train a model to see runs here.")


def create_channel_chart(
    df: pd.DataFrame,
    channel: Channel,
    timeframe: str,
    symbol: str
) -> Optional['go.Figure']:
    """Create interactive candlestick chart with channel overlay."""
    if not PLOTLY_AVAILABLE:
        return None

    # Filter to regular trading hours only
    df = filter_market_hours(df)

    # Resample data to match timeframe
    if timeframe != '5min':
        df_resampled = resample_ohlc(df, timeframe)
    else:
        df_resampled = df

    # Get last window bars
    window = channel.window
    df_window = df_resampled.iloc[-window:].copy()

    # Use bar indices instead of datetime to avoid gaps (kinked lines during market closures)
    bar_indices = list(range(len(df_window)))
    timestamps = df_window.index.tolist()

    # Create candlestick with bar indices
    fig = go.Figure(data=[go.Candlestick(
        x=bar_indices,
        open=df_window['open'],
        high=df_window['high'],
        low=df_window['low'],
        close=df_window['close'],
        name=symbol
    )])

    # Add channel lines using bar indices (ensures straight lines with no gaps)
    fig.add_trace(go.Scatter(
        x=bar_indices,
        y=channel.center_line,
        mode='lines',
        name='Center',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=bar_indices,
        y=channel.upper_line,
        mode='lines',
        name='Upper',
        line=dict(color='red', width=1, dash='dash'),
        fill=None
    ))

    fig.add_trace(go.Scatter(
        x=bar_indices,
        y=channel.lower_line,
        mode='lines',
        name='Lower',
        line=dict(color='green', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(0, 100, 200, 0.1)'
    ))

    # Format tick labels to show dates at regular intervals
    tick_step = max(1, len(bar_indices) // 6)  # ~6 ticks
    tickvals = bar_indices[::tick_step]
    ticktext = [ts.strftime('%m/%d %H:%M') if hasattr(ts, 'strftime') else str(ts)
                for ts in timestamps[::tick_step]]

    # Update layout with bar index x-axis and formatted tick labels
    fig.update_layout(
        title=f"{symbol} {timeframe} - {channel.direction.name} Channel",
        yaxis_title="Price ($)",
        xaxis_title="Bar Index (Time)",
        height=500,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        legend=dict(x=0.01, y=0.99),
        xaxis=dict(
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=-45
        )
    )

    return fig


def create_prediction_chart(
    df: pd.DataFrame,
    channel: Channel,
    timeframe: str,
    symbol: str,
    predicted_duration: float,
    predicted_direction: str,  # "UP" or "DOWN"
    direction_prob: float,
    confidence: float
) -> Optional['go.Figure']:
    """
    Create channel chart with forward projection based on model predictions.

    Shows:
    - Current channel with price data
    - Projected channel extension (predicted duration bars forward)
    - Current and projected price levels
    - Break direction indicator
    """
    if not PLOTLY_AVAILABLE:
        return None

    # Filter to regular trading hours only
    df = filter_market_hours(df)

    # Resample data to match timeframe
    if timeframe != '5min':
        df_resampled = resample_ohlc(df, timeframe)
    else:
        df_resampled = df

    # Get last window bars
    window = channel.window
    df_window = df_resampled.iloc[-window:].copy()

    # Bar indices for current data
    bar_indices = list(range(len(df_window)))

    # Calculate projection bars (at least 5, cap at reasonable amount)
    proj_bars = max(5, min(int(predicted_duration), 50))
    proj_indices = list(range(len(df_window), len(df_window) + proj_bars))
    all_indices = bar_indices + proj_indices

    # Project channel forward using slope
    x_proj = np.arange(window, window + proj_bars)
    center_proj = channel.slope * x_proj + channel.intercept
    upper_proj = center_proj + 2 * channel.std_dev
    lower_proj = center_proj - 2 * channel.std_dev

    # Combine current and projected lines
    center_all = np.concatenate([channel.center_line, center_proj])
    upper_all = np.concatenate([channel.upper_line, upper_proj])
    lower_all = np.concatenate([channel.lower_line, lower_proj])

    # Create figure
    fig = go.Figure()

    # Add candlestick for current data
    fig.add_trace(go.Candlestick(
        x=bar_indices,
        open=df_window['open'],
        high=df_window['high'],
        low=df_window['low'],
        close=df_window['close'],
        name=symbol,
        increasing_line_color='green',
        decreasing_line_color='red'
    ))

    # Add current channel lines (solid)
    fig.add_trace(go.Scatter(
        x=bar_indices,
        y=channel.center_line,
        mode='lines',
        name='Center',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=bar_indices,
        y=channel.upper_line,
        mode='lines',
        name='Upper',
        line=dict(color='red', width=1.5),
        fill=None
    ))

    fig.add_trace(go.Scatter(
        x=bar_indices,
        y=channel.lower_line,
        mode='lines',
        name='Lower',
        line=dict(color='green', width=1.5),
        fill='tonexty',
        fillcolor='rgba(0, 100, 200, 0.1)'
    ))

    # Add projected channel lines (dashed, with different fill color)
    proj_color = 'rgba(255, 100, 100, 0.3)' if predicted_direction == "DOWN" else 'rgba(100, 255, 100, 0.3)'

    fig.add_trace(go.Scatter(
        x=proj_indices,
        y=center_proj,
        mode='lines',
        name='Projected Center',
        line=dict(color='blue', width=2, dash='dot'),
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=proj_indices,
        y=upper_proj,
        mode='lines',
        name='Projected Upper',
        line=dict(color='red', width=1.5, dash='dash'),
        fill=None,
        showlegend=False
    ))

    fig.add_trace(go.Scatter(
        x=proj_indices,
        y=lower_proj,
        mode='lines',
        name='Projected Lower',
        line=dict(color='green', width=1.5, dash='dash'),
        fill='tonexty',
        fillcolor=proj_color,
        showlegend=False
    ))

    # Add vertical line at "now"
    fig.add_vline(
        x=len(bar_indices) - 0.5,
        line_dash="dash",
        line_color="gray",
        annotation_text="NOW",
        annotation_position="top"
    )

    # Current price and channel levels
    current_price = df_window['close'].iloc[-1]
    current_upper = channel.upper_line[-1]
    current_lower = channel.lower_line[-1]

    # Projected levels at predicted duration
    proj_idx = min(proj_bars - 1, int(predicted_duration) - 1) if predicted_duration > 0 else 0
    projected_upper = upper_proj[proj_idx] if proj_idx < len(upper_proj) else upper_proj[-1]
    projected_lower = lower_proj[proj_idx] if proj_idx < len(lower_proj) else lower_proj[-1]

    # Add price level annotations on the right side
    annotations = [
        # Current levels
        dict(x=len(bar_indices) - 1, y=current_upper, text=f"Upper: ${current_upper:.2f}",
             showarrow=False, xanchor='right', font=dict(size=10, color='red')),
        dict(x=len(bar_indices) - 1, y=current_lower, text=f"Lower: ${current_lower:.2f}",
             showarrow=False, xanchor='right', font=dict(size=10, color='green')),
        # Projected levels
        dict(x=len(bar_indices) + proj_bars - 1, y=projected_upper,
             text=f"Target High: ${projected_upper:.2f}",
             showarrow=True, arrowhead=2, ax=40, ay=0, font=dict(size=10, color='darkred')),
        dict(x=len(bar_indices) + proj_bars - 1, y=projected_lower,
             text=f"Target Low: ${projected_lower:.2f}",
             showarrow=True, arrowhead=2, ax=40, ay=0, font=dict(size=10, color='darkgreen')),
    ]

    # Add break direction arrow
    break_y = current_lower if predicted_direction == "DOWN" else current_upper
    break_color = "red" if predicted_direction == "DOWN" else "green"
    arrow_y_offset = -30 if predicted_direction == "DOWN" else 30

    fig.add_annotation(
        x=len(bar_indices) + proj_bars // 2,
        y=break_y,
        text=f"Break {predicted_direction} ({direction_prob:.0%})",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor=break_color,
        ax=0,
        ay=arrow_y_offset,
        font=dict(size=12, color=break_color, weight='bold'),
        bgcolor='white',
        bordercolor=break_color,
        borderwidth=1
    )

    # Update layout
    dir_emoji = "🔴" if predicted_direction == "DOWN" else "🟢"
    title = f"{dir_emoji} {symbol} {timeframe} | Predicted: {predicted_direction} in ~{predicted_duration:.0f} bars | Confidence: {confidence:.0%}"

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        yaxis_title="Price ($)",
        xaxis_title=f"Bars ({timeframe})",
        height=450,
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        template='plotly_white',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
        annotations=annotations,
        margin=dict(r=120)  # Extra right margin for annotations
    )

    return fig


def display_channel_table(channels: Dict[str, Channel]):
    """Display channel data as a table."""
    if not channels:
        st.info("No channels detected")
        return

    rows = []
    for tf in TIMEFRAMES:
        if tf in channels:
            ch = channels[tf]
            display_tf = TIMEFRAME_DISPLAY_NAMES.get(tf, tf)
            if ch.valid:
                dir_map = {Direction.BULL: "🟢 UP", Direction.SIDEWAYS: "🟡 SIDE", Direction.BEAR: "🔴 DOWN"}
                pos = ch.position_at()
                rows.append({
                    "Timeframe": display_tf,
                    "Direction": dir_map.get(ch.direction, "?"),
                    "Position": f"{pos:.2f}",
                    "Width %": f"{ch.width_pct:.1f}%",
                    "Bounces": int(ch.bounce_count),  # Ensure int type
                    "R²": f"{ch.r_squared:.2f}"
                })
            else:
                rows.append({
                    "Timeframe": display_tf,
                    "Direction": "—",
                    "Position": "—",
                    "Width %": "—",
                    "Bounces": 0,  # Use 0 instead of "—" to maintain int type
                    "R²": "—"
                })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, width='stretch', hide_index=True)
    else:
        st.info("No valid channels")


if __name__ == "__main__":
    main()
