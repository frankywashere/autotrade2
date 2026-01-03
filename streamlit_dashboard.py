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
from v7.core.channel import detect_channel, Channel, Direction

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

    return checkpoints


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

                config_dict['model'] = {
                    'hidden_dim': hidden_dim,
                    'cfc_units': cfc_units,
                    'num_attention_heads': 4 if hidden_dim <= 128 else 8,
                    'dropout': 0.1,
                    '_source': 'heuristic_inference',
                    '_inferred': True,
                    '_inference_note': f"Config inferred from tensor shapes (may not match training)"
                }

        return config_dict if config_dict else None
    except Exception as e:
        st.warning(f"Error extracting config: {e}")
        return None


@st.cache_resource
def load_model(checkpoint_path: str) -> Optional[HierarchicalCfCModel]:
    """Load model from checkpoint with proper architecture."""
    try:
        path = Path(checkpoint_path)

        # Load checkpoint once and reuse for both config extraction and model loading
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint

        # Extract config from the already-loaded checkpoint (pass checkpoint to avoid re-loading)
        config = extract_config_from_checkpoint(path, checkpoint=checkpoint)

        if config and 'model' in config:
            model_config = config['model']
            hidden_dim = model_config.get('hidden_dim', 64)
            cfc_units = model_config.get('cfc_units', 96)
            num_heads = model_config.get('num_attention_heads', 4)
            dropout = model_config.get('dropout', 0.1)
            shared_heads = model_config.get('shared_heads', True)  # Default to shared for backward compat

            # Show config source status
            source = model_config.get('_source', 'unknown')
            if source == 'checkpoint_model_kwargs':
                st.success(f"✓ Config from checkpoint: hidden_dim={hidden_dim}, cfc_units={cfc_units}")
            elif source == 'training_config_json':
                st.info(f"✓ Config from training_config.json: hidden_dim={hidden_dim}, cfc_units={cfc_units}")
            elif source == 'heuristic_inference':
                st.warning(f"⚠️ Config inferred from tensor shapes: hidden_dim={hidden_dim}, cfc_units={cfc_units}")
        else:
            hidden_dim, cfc_units, num_heads, dropout, shared_heads = 64, 96, 4, 0.1, True
            st.warning("⚠️ No config found, using defaults: hidden_dim=64, cfc_units=96")

        # Infer shared_heads from state_dict keys (overrides config if separate heads detected)
        has_separate_heads = any('per_tf_duration_heads' in k for k in state_dict.keys())
        if has_separate_heads:
            shared_heads = False
            st.info("Detected separate per-TF heads architecture")

        model = create_model(
            hidden_dim=hidden_dim,
            cfc_units=cfc_units,
            num_attention_heads=num_heads,
            dropout=dropout,
            shared_heads=shared_heads,
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
    model: HierarchicalCfCModel
) -> Optional[Dict]:
    """Make predictions using the model - returns per-TF predictions."""
    if model is None:
        return None

    try:
        # Import feature extraction
        from v7.features.full_features import extract_full_features, features_to_tensor_dict
        from v7.features.feature_ordering import FEATURE_ORDER

        # Extract features (pass vix_df, not vix_value)
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

        # Extract per-TF and aggregate predictions
        per_tf = outputs['per_tf']
        agg = outputs['aggregate']
        best_tf_idx = int(outputs['best_tf_idx'][0])

        # TF names for indexing
        TF_NAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', '1d', '1w', '1M', '3M']

        return {
            # Per-timeframe predictions (all 11 TFs)
            'per_tf': {
                'duration_mean': per_tf['duration_mean'][0].numpy(),  # [11]
                'duration_std': per_tf['duration_std'][0].numpy(),    # [11]
                'direction': per_tf['direction'][0].numpy(),          # [11]
                'direction_probs': per_tf['direction_probs'][0].numpy(),  # [11]
                'next_channel': per_tf['next_channel'][0].numpy(),    # [11]
                'next_channel_probs': per_tf['next_channel_probs'][0].numpy(),  # [11, 3]
                'confidence': per_tf['confidence'][0].numpy(),        # [11]
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
                'confidence': float(agg['confidence'][0, 0])
            }
        }
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

        # Data settings
        st.subheader("Data Settings")

        # Show live data module status
        if LIVE_DATA_AVAILABLE:
            st.caption("✓ Live data module available")
        else:
            st.caption("✗ Live data module not available")

        use_live = st.checkbox("Use Live Data", value=LIVE_DATA_AVAILABLE, disabled=not LIVE_DATA_AVAILABLE)
        lookback_days = st.slider("Lookback Days", 30, 180, 90)

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

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Confidence", f"{best_conf:.1%}")

                with col2:
                    direction = "UP ↑" if best_dir_prob > 0.5 else "DOWN ↓"
                    st.metric("Direction", direction, f"{abs(best_dir_prob - 0.5)*200:.0f}%")

                with col3:
                    st.metric(f"Duration ({best_tf_name})", f"{best_dur:.1f} bars", f"±{best_dur_std:.1f}")

                with col4:
                    next_labels = ["DOWN", "SAME", "UP"]
                    st.metric("Next Channel", next_labels[best_next_ch], f"{best_next_probs[best_next_ch]:.0%}")

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

                # Per-timeframe breakdown table
                st.divider()
                st.subheader("All Timeframe Predictions")

                TF_NAMES = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', '1d', '1w', '1M', '3M']
                tf_rows = []
                for i, tf_name in enumerate(TF_NAMES):
                    is_best = (i == best_tf_idx)
                    tf_rows.append({
                        "TF": f"**{tf_name}**" if is_best else tf_name,
                        "Confidence": f"{per_tf['confidence'][i]:.1%}",
                        "Duration": f"{per_tf['duration_mean'][i]:.1f}±{per_tf['duration_std'][i]:.1f}",
                        "Direction": "UP ↑" if per_tf['direction_probs'][i] > 0.5 else "DOWN ↓",
                        "Dir Prob": f"{per_tf['direction_probs'][i]:.0%}",
                        "Next Ch": ["DN", "SAME", "UP"][per_tf['next_channel'][i]],
                    })

                tf_df = pd.DataFrame(tf_rows)
                st.dataframe(tf_df, width='stretch', hide_index=True)

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
        - **644 input features** per prediction
        - TSLA channel features (geometry, bounces, RSI)
        - SPY channel features for cross-asset analysis
        - VIX for volatility context
        - Historical bounce patterns
        """)

        st.subheader("Model Comparison")

        # Build comparison data from all checkpoints
        checkpoints = find_checkpoints()
        if checkpoints:
            comparison_rows = []
            for cp in checkpoints:
                row = {
                    'Model Name': cp['name'],
                    'Val Loss': None,
                    'Best Epoch': None,
                    'Dir Acc %': None,
                    'Next Ch Acc %': None,
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
                    'Size MB': df['Size MB'].apply(lambda x: format_value(x, 1))
                })

                st.dataframe(display_df, use_container_width=True, hide_index=True)

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
