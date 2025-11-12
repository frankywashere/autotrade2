"""
Arrow-key based interactive parameter selection system for ML training scripts.
Uses InquirerPy for intuitive navigation and inline editing.
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
    from InquirerPy.separator import Separator
    from InquirerPy.validator import PathValidator, NumberValidator
    INQUIRERPY_AVAILABLE = True
except ImportError:
    INQUIRERPY_AVAILABLE = False
    print("Warning: InquirerPy not installed. Install with: pip install inquirerpy")

import config
from src.ml.device_manager import DeviceManager
from src.ml.interactive_params import calculate_optimal_batch_size


class ArrowKeyParameterSelector:
    """
    Arrow-key based parameter selection using InquirerPy.
    Navigate with arrow keys, press Enter to edit parameters.
    """

    def __init__(self, mode: str = 'standard'):
        """
        Initialize the arrow-key parameter selector.

        Args:
            mode: 'standard' for train_model.py, 'lazy' for train_model_lazy.py
        """
        if not INQUIRERPY_AVAILABLE:
            raise ImportError(
                "InquirerPy is required for arrow-key navigation.\n"
                "Install with: pip install inquirerpy"
            )

        self.mode = mode
        self.device_manager = DeviceManager()
        self.params = self._get_default_params()
        self.modified_params = set()

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters from config."""
        return {
            # Data parameters
            'input_timeframe': '1min',  # NEW: Timeframe for multi-scale training
            'spy_data': 'data/SPY_1min.csv',
            'tsla_data': 'data/TSLA_1min.csv',
            'tsla_events': 'data/tsla_events_REAL.csv',
            'macro_api_key': None,

            # Training period
            'start_year': config.ML_TRAIN_START_YEAR,
            'end_year': config.ML_TRAIN_END_YEAR,

            # Model architecture
            'model_type': config.ML_MODEL_TYPE,
            'hidden_size': config.LNN_HIDDEN_SIZE,
            'num_layers': config.LNN_NUM_LAYERS,
            'sequence_length': config.ML_SEQUENCE_LENGTH,

            # Training hyperparameters
            'epochs': config.ML_EPOCHS,
            'pretrain_epochs': 10,
            'batch_size': config.ML_BATCH_SIZE or 16,
            'learning_rate': config.LNN_LEARNING_RATE,
            'validation_split': config.ML_VALIDATION_SPLIT,

            # Features
            'use_channel_features': config.USE_CHANNEL_FEATURES,
            'use_rsi_features': config.USE_RSI_FEATURES,
            'use_correlation_features': config.USE_CORRELATION_FEATURES,
            'use_event_features': config.USE_EVENT_FEATURES,
            'prediction_horizon_mode': 'uniform_bars',  # NEW: uniform_bars or uniform_time
            'prediction_horizon_hours': config.PREDICTION_HORIZON_HOURS,

            # Device
            'device': None,  # Will be auto-detected
            'auto_device': False,

            # GPU optimization
            'num_workers': 0,  # CPU threads for data loading
            'pin_memory': False,  # Pin memory for GPU transfers

            # System info (populated during detection)
            '_system_ram_gb': 8.0,
            '_available_ram_gb': 4.0,
        }

    def run(self) -> Dict[str, Any]:
        """Run the arrow-key based parameter selection process."""
        self._display_welcome()
        self._display_system_info()

        # Auto-detect device
        self._auto_select_device()

        # Main menu loop
        while True:
            # Build menu choices
            choices = self._build_menu_choices()

            # Show interactive menu
            try:
                action = inquirer.select(
                    message="Navigate with arrow keys, Enter to edit:",
                    choices=choices,
                    default=None,
                    pointer="❯",
                    instruction=" ",  # Hide default instruction
                    long_instruction="[↑↓] Navigate  [Enter] Edit  [d] Done  [r] Reset  [q] Quit",
                    vi_mode=False,  # Disable vim keys for simplicity
                    show_cursor=True,
                ).execute()
            except KeyboardInterrupt:
                print("\n\nCancelled by user.")
                sys.exit(0)

            # Handle action
            if action == 'done':
                if self._validate_and_confirm():
                    return self.params
            elif action == 'reset':
                self._reset_all()
            elif action == 'quit':
                print("\nCancelled.")
                sys.exit(0)
            elif action and action != 'separator':
                # Edit parameter
                self._edit_parameter(action)

    def _display_welcome(self):
        """Display welcome message."""
        print("\n" + "=" * 80)
        print("🎛️  ARROW-KEY PARAMETER SELECTION")
        print("=" * 80)
        print(f"\nMode: {'Memory-efficient (lazy)' if self.mode == 'lazy' else 'Standard'} training")
        print("Navigate with arrow keys, press Enter to edit parameters.")
        print("Press Ctrl+C at any time to cancel.\n")

    def _display_system_info(self):
        """Display system information and capabilities."""
        print("Detecting system capabilities...\n")

        hardware_info = self.device_manager.detect_hardware()

        print("System Information:")
        print(f"  Platform: {hardware_info['platform']}")
        print(f"  Processor: {hardware_info['processor']}")
        print(f"  PyTorch: {hardware_info['pytorch_version']}")

        # RAM info
        try:
            total_ram = self.device_manager.get_system_ram_gb()
            available_ram = self.device_manager.get_available_ram_gb()
            print(f"  RAM: {total_ram:.1f} GB total, {available_ram:.1f} GB available")
            self.params['_system_ram_gb'] = total_ram
            self.params['_available_ram_gb'] = available_ram
        except:
            print("  RAM: Unable to detect")

        # GPU info
        if hardware_info.get('cuda_available'):
            print(f"  CUDA GPU: {hardware_info.get('cuda_device_name', 'Available')}")
        if hardware_info.get('mps_available'):
            print(f"  Apple Metal: Available ({hardware_info.get('apple_chip', 'Apple Silicon')})")

        print("\nPress Enter to continue...")
        input()

    def _build_menu_choices(self) -> List:
        """Build menu choices with categories and separators."""
        choices = []

        # Parameter catalog
        catalog = self._get_parameter_catalog()

        for category_name, params in catalog.items():
            # Category separator
            choices.append(Separator(f"\n{category_name}"))
            choices.append(Separator("─" * 60))

            # Parameters in category
            for param_key, info in params:
                value = self._format_display_value(param_key, self.params.get(param_key))
                modified = "*" if param_key in self.modified_params else " "

                # Format parameter line
                display_name = f"{modified} {info['name']}"
                display_line = f"{display_name:<28} : {value:<25}"

                choices.append(Choice(value=param_key, name=display_line))

        # Add action options
        choices.append(Separator("\n" + "─" * 60))
        choices.append(Choice(value='done', name='✓ Done - Start Training'))
        choices.append(Choice(value='reset', name='↺ Reset All to Defaults'))
        choices.append(Choice(value='quit', name='✗ Quit'))

        return choices

    def _get_parameter_catalog(self) -> Dict[str, List[Tuple[str, Dict]]]:
        """Get categorized parameter catalog."""
        return {
            "📁 DATA FILES": [
                ('input_timeframe', {'name': 'Input timeframe', 'type': 'timeframe'}),
                ('spy_data', {'name': 'SPY data file', 'type': 'file'}),
                ('tsla_data', {'name': 'TSLA data file', 'type': 'file'}),
                ('tsla_events', {'name': 'TSLA events file', 'type': 'file'}),
                ('macro_api_key', {'name': 'Macro API key', 'type': 'api_key'}),
            ],
            "📅 TRAINING PERIOD": [
                ('start_year', {'name': 'Start year', 'type': 'year'}),
                ('end_year', {'name': 'End year', 'type': 'year'}),
            ],
            "🧠 MODEL ARCHITECTURE": [
                ('model_type', {'name': 'Model type', 'type': 'model'}),
                ('hidden_size', {'name': 'Hidden size', 'type': 'number'}),
                ('num_layers', {'name': 'Number of layers', 'type': 'number'}),
                ('sequence_length', {'name': 'Sequence length', 'type': 'number'}),
            ],
            "⚙️ TRAINING PARAMETERS": [
                ('epochs', {'name': 'Training epochs', 'type': 'number'}),
                ('pretrain_epochs', {'name': 'Pretrain epochs', 'type': 'number'}),
                ('batch_size', {'name': 'Batch size', 'type': 'batch_size'}),
                ('learning_rate', {'name': 'Learning rate', 'type': 'float'}),
                ('validation_split', {'name': 'Validation split', 'type': 'float'}),
            ],
            "📊 FEATURE FLAGS": [
                ('use_channel_features', {'name': 'Channel features', 'type': 'boolean'}),
                ('use_rsi_features', {'name': 'RSI features', 'type': 'boolean'}),
                ('use_correlation_features', {'name': 'Correlation features', 'type': 'boolean'}),
                ('use_event_features', {'name': 'Event features', 'type': 'boolean'}),
                ('prediction_horizon_mode', {'name': 'Prediction mode', 'type': 'prediction_mode'}),
                ('prediction_horizon_hours', {'name': 'Prediction horizon (hrs)', 'type': 'number'}),
            ],
            "🖥️ COMPUTE DEVICE": [
                ('device', {'name': 'Device', 'type': 'device'}),
            ],
            "🚀 GPU OPTIMIZATION": [
                ('num_workers', {'name': 'Data loading workers', 'type': 'number'}),
                ('pin_memory', {'name': 'Pin memory', 'type': 'boolean'}),
            ]
        }

    def _format_display_value(self, param_key: str, value: Any) -> str:
        """Format parameter value for display."""
        if param_key == 'macro_api_key':
            return 'Not set (local data available)' if not value else 'Set'
        elif param_key == 'prediction_horizon_mode':
            if value == 'uniform_bars':
                return 'Uniform bars'
            elif value == 'uniform_time':
                return 'Uniform time (24h)'
            else:
                return str(value)
        elif param_key == 'device':
            if self.params.get('auto_device'):
                return f"{value} (auto-detected)"
            return value or 'Auto-detect'
        elif isinstance(value, bool):
            return 'Yes' if value else 'No'
        elif param_key == 'learning_rate':
            return f"{value:.6f}" if value else "0.001"
        elif param_key == 'validation_split':
            return f"{value:.2f}" if value else "0.1"
        else:
            return str(value)

    def _edit_parameter(self, param_key: str):
        """Edit a single parameter based on its type."""
        # Find parameter info
        param_info = None
        for category, params in self._get_parameter_catalog().items():
            for key, info in params:
                if key == param_key:
                    param_info = info
                    break

        if not param_info:
            return

        param_type = param_info.get('type', 'text')
        param_name = param_info['name']

        print(f"\n{'=' * 60}")
        print(f"Editing: {param_name}")
        print(f"Current value: {self._format_display_value(param_key, self.params.get(param_key))}")

        try:
            if param_type == 'boolean':
                self._edit_boolean(param_key, param_name)
            elif param_type == 'timeframe':
                self._edit_timeframe(param_key, param_name)
            elif param_type == 'device':
                self._edit_device(param_key, param_name)
            elif param_type == 'prediction_mode':
                self._edit_prediction_mode(param_key, param_name)
            elif param_type == 'batch_size':
                self._edit_batch_size(param_key, param_name)
            elif param_type == 'year':
                self._edit_year(param_key, param_name)
            elif param_type == 'number':
                self._edit_number(param_key, param_name)
            elif param_type == 'float':
                self._edit_float(param_key, param_name)
            elif param_type == 'model':
                self._edit_model_type(param_key, param_name)
            elif param_type == 'file':
                self._edit_file_path(param_key, param_name)
            elif param_type == 'api_key':
                self._edit_api_key(param_key, param_name)
            else:
                self._edit_text(param_key, param_name)
        except KeyboardInterrupt:
            print("\nEdit cancelled.")

    def _edit_boolean(self, param_key: str, param_name: str):
        """Edit boolean parameter."""
        # Add hints for certain parameters
        if param_key == 'pin_memory':
            print("\nHint: Pin memory for faster GPU→VRAM transfers (~10% speedup)")
            print("      Recommended: True for GPU, False for CPU/MPS")

        result = inquirer.confirm(
            message=f"Enable {param_name}?",
            default=self.params.get(param_key, False),
        ).execute()

        if result != self.params.get(param_key):
            self.params[param_key] = result
            self.modified_params.add(param_key)
            print(f"✓ {param_name} set to: {'Yes' if result else 'No'}")

    def _edit_timeframe(self, param_key: str, param_name: str):
        """Edit timeframe selection for multi-scale training."""
        timeframes = ['1min', '5min', '15min', '30min', '1hour', '2hour', '3hour', '4hour', 'daily', 'weekly', 'monthly', '3month']

        print("\nHint: Select which timeframe CSV to train on (multi-scale training)")
        print("      Each timeframe model specializes in different temporal patterns")

        choices = []
        for tf in timeframes:
            if tf == self.params.get('input_timeframe'):
                choices.append(Choice(value=tf, name=f"{tf} (current)"))
            else:
                choices.append(Choice(value=tf, name=tf))

        result = inquirer.select(
            message="Select input timeframe:",
            choices=choices,
            default=self.params.get('input_timeframe', '1min'),
        ).execute()

        if result and result != self.params.get('input_timeframe'):
            self.params['input_timeframe'] = result
            self.modified_params.add('input_timeframe')

            # Auto-update data file paths
            self.params['spy_data'] = f'data/SPY_{result}.csv'
            self.params['tsla_data'] = f'data/TSLA_{result}.csv'

            print(f"✓ Timeframe set to: {result}")
            print(f"  Auto-updated data paths for {result} CSVs")

    def _edit_device(self, param_key: str, param_name: str):
        """Edit device selection."""
        devices = self.device_manager.get_available_devices()
        choices = []

        # Add available devices
        for device_name, info in devices.items():
            if info['available']:
                if device_name == self.params.get('device'):
                    desc = f"{device_name} (current)"
                else:
                    desc = device_name
                choices.append(Choice(value=device_name, name=desc))

        # Add auto option
        choices.append(Choice(value='auto', name='Auto-detect best device'))

        result = inquirer.select(
            message="Select compute device:",
            choices=choices,
            default=self.params.get('device', 'auto'),
        ).execute()

        if result == 'auto':
            self._auto_select_device()
        else:
            self.params['device'] = result
            self.params['auto_device'] = False
            self.modified_params.add('device')
            print(f"✓ Device set to: {result}")

    def _edit_prediction_mode(self, param_key: str, param_name: str):
        """Edit prediction horizon mode selection."""
        modes = [
            Choice(value='uniform_bars', name='uniform_bars - Same bar count for all models'),
            Choice(value='uniform_time', name='uniform_time - Same time window (24h) for all models')
        ]

        print("\n" + "─" * 60)
        print("Prediction Horizon Mode:")
        print("\n1. Uniform Bars (current behavior):")
        print("   All models use same bar count (e.g., 24 bars)")
        print("   - 15min: 24 bars = 6 hours ahead")
        print("   - 1hour: 24 bars = 24 hours ahead")
        print("   - 4hour: 24 bars = 96 hours ahead")
        print("   - daily: 24 bars = 24 days ahead")
        print("\n2. Uniform Time:")
        print("   All models predict same time window (24 hours)")
        print("   - 15min: 96 bars")
        print("   - 1hour: 24 bars")
        print("   - 4hour: 6 bars")
        print("   - daily: 1 bar")
        print("\nPress ESC to cancel")

        result = inquirer.select(
            message="Select prediction horizon mode:",
            choices=modes,
            default=self.params.get('prediction_horizon_mode', 'uniform_bars')
        ).execute()

        if result:
            self.params['prediction_horizon_mode'] = result
            self.modified_params.add('prediction_horizon_mode')
            print(f"✓ Prediction mode set to: {result}")

    def _edit_batch_size(self, param_key: str, param_name: str):
        """Edit batch size with RAM-based suggestions."""
        suggestions = self._get_batch_size_suggestions()

        print(f"\n💡 RAM-based suggestions ({self.params.get('_available_ram_gb', 4.0):.1f} GB available):")

        choices = [
            Choice(
                value=suggestions['conservative'],
                name=f"Conservative: {suggestions['conservative']} (safest, slower)"
            ),
            Choice(
                value=suggestions['balanced'],
                name=f"Balanced: {suggestions['balanced']} (recommended)"
            ),
            Choice(
                value=suggestions['aggressive'],
                name=f"Aggressive: {suggestions['aggressive']} (fastest, may OOM)"
            ),
            Choice(value='custom', name="Custom value..."),
        ]

        result = inquirer.select(
            message="Select batch size:",
            choices=choices,
            default=suggestions['balanced'] if 'batch_size' not in self.modified_params
                    else self.params.get('batch_size', 16),
        ).execute()

        if result == 'custom':
            def validate_batch_size(val_str):
                """Validate batch size input."""
                try:
                    val = int(val_str) if val_str else 16
                    return 8 <= val <= 8192
                except (ValueError, TypeError):
                    return False

            result = inquirer.number(
                message="Enter custom batch size (8-8192):",
                min_allowed=8,
                max_allowed=8192,
                default=self.params.get('batch_size', 16),
                validate=validate_batch_size,
                invalid_message="Batch size must be between 8 and 8192",
            ).execute()

        if result and result != self.params.get('batch_size'):
            self.params['batch_size'] = int(result)
            self.modified_params.add('batch_size')
            print(f"✓ Batch size set to: {result}")
            if int(result) > 1024:
                print(f"⚠️  Warning: Large batch size ({result}) - ensure your GPU has enough VRAM!")

    def _edit_year(self, param_key: str, param_name: str):
        """Edit year parameter."""
        if param_key == 'start_year':
            min_year, max_year = 2010, 2023
            hint = "Earlier years = more data but slower training"
        else:  # end_year
            min_year = self.params.get('start_year', 2010)
            max_year = 2024
            hint = "Reserve 2024 for validation/backtesting"

        print(f"Hint: {hint}")

        def validate_year(val_str):
            """Validate year input."""
            try:
                val = int(val_str) if val_str else 2018
                return min_year <= val <= max_year
            except (ValueError, TypeError):
                return False

        result = inquirer.number(
            message=f"Enter {param_name.lower()} ({min_year}-{max_year}):",
            min_allowed=min_year,
            max_allowed=max_year,
            default=self.params.get(param_key, 2018),
            validate=validate_year,
            invalid_message=f"Year must be between {min_year} and {max_year}",
        ).execute()

        if result and result != self.params.get(param_key):
            self.params[param_key] = int(result)
            self.modified_params.add(param_key)
            print(f"✓ {param_name} set to: {result}")

    def _edit_number(self, param_key: str, param_name: str):
        """Edit integer parameter."""
        # Set reasonable ranges for different parameters
        ranges = {
            'hidden_size': (32, 1024, "Complexity: 64=simple, 512=complex patterns"),
            'num_layers': (1, 10, "Network depth"),
            'sequence_length': (20, 5000, "Bars of history per prediction (depends on timeframe)"),
            'epochs': (1, 1000, "Full passes through data"),
            'pretrain_epochs': (0, 100, "Self-supervised pretraining epochs"),
            'prediction_horizon_hours': (1, 168, "Hours ahead to predict"),
            'num_workers': (0, 8, "CPU threads for data loading (0=main only, 2=recommended for GPU)"),
        }

        min_val, max_val, hint = ranges.get(param_key, (1, 1000, ""))
        if hint:
            print(f"Hint: {hint}")

        def validate_number(val_str):
            """Validate number input."""
            try:
                val = int(val_str) if val_str else min_val
                return min_val <= val <= max_val
            except (ValueError, TypeError):
                return False

        result = inquirer.number(
            message=f"Enter {param_name.lower()} ({min_val}-{max_val}):",
            min_allowed=min_val,
            max_allowed=max_val,
            default=self.params.get(param_key, min_val),
            validate=validate_number,
            invalid_message=f"Value must be between {min_val} and {max_val}",
        ).execute()

        if result and result != self.params.get(param_key):
            self.params[param_key] = int(result)
            self.modified_params.add(param_key)
            print(f"✓ {param_name} set to: {result}")

    def _edit_float(self, param_key: str, param_name: str):
        """Edit float parameter."""
        if param_key == 'learning_rate':
            print("Hint: Common values: 0.001 (default), 0.0001 (conservative)")
            result = inquirer.text(
                message="Enter learning rate (e.g., 0.001):",
                default=str(self.params.get(param_key, 0.001)),
                validate=lambda val: self._validate_float(val, 0.0, 1.0),
                invalid_message="Must be a number between 0 and 1",
            ).execute()

            if result:
                float_val = float(result)
                if float_val != self.params.get(param_key):
                    self.params[param_key] = float_val
                    self.modified_params.add(param_key)
                    print(f"✓ {param_name} set to: {float_val}")

        elif param_key == 'validation_split':
            print("Hint: Fraction of data for validation (0.1 = 10%)")
            result = inquirer.text(
                message="Enter validation split (0.0-0.5):",
                default=str(self.params.get(param_key, 0.1)),
                validate=lambda val: self._validate_float(val, 0.0, 0.5),
                invalid_message="Must be between 0.0 and 0.5",
            ).execute()

            if result:
                float_val = float(result)
                if float_val != self.params.get(param_key):
                    self.params[param_key] = float_val
                    self.modified_params.add(param_key)
                    print(f"✓ {param_name} set to: {float_val}")

    def _validate_float(self, value: str, min_val: float, max_val: float) -> bool:
        """Validate float input."""
        try:
            val = float(value)
            return min_val <= val <= max_val
        except:
            return False

    def _edit_model_type(self, param_key: str, param_name: str):
        """Edit model type selection."""
        choices = [
            Choice(value='LNN', name='LNN - Liquid Neural Network (recommended)'),
            Choice(value='LSTM', name='LSTM - Long Short-Term Memory'),
        ]

        result = inquirer.select(
            message="Select model type:",
            choices=choices,
            default=self.params.get(param_key, 'LNN'),
        ).execute()

        if result != self.params.get(param_key):
            self.params[param_key] = result
            self.modified_params.add(param_key)
            print(f"✓ Model type set to: {result}")

    def _edit_file_path(self, param_key: str, param_name: str):
        """Edit file path parameter."""
        current = self.params.get(param_key, '')

        def validate_path(val):
            """Validate file path input."""
            return val.strip() != "" if val else False

        result = inquirer.text(
            message=f"Enter path for {param_name}:",
            default=current,
            validate=validate_path,
            invalid_message="Path cannot be empty",
        ).execute()

        if result and result != current:
            self.params[param_key] = result.strip()
            self.modified_params.add(param_key)

            # Check if file exists
            if not Path(result.strip()).exists():
                print(f"⚠️  Warning: File does not exist: {result}")
            else:
                print(f"✓ {param_name} set to: {result}")

    def _edit_api_key(self, param_key: str, param_name: str):
        """Edit API key parameter."""
        print("\n📝 Note: Local macro economic data is available in data/ directory.")
        print("You can use the local data without an API key.")

        # Ensure default is always a string (None would cause InquirerPy to crash)
        current_value = self.params.get(param_key)
        default_value = current_value if current_value is not None else ''

        result = inquirer.text(
            message="Enter API key (or press Enter to skip):",
            default=default_value,
        ).execute()

        if result != self.params.get(param_key):
            self.params[param_key] = result if result else None
            self.modified_params.add(param_key)
            if result:
                print(f"✓ API key set")
            else:
                print(f"✓ API key cleared (will use local data)")

    def _edit_text(self, param_key: str, param_name: str):
        """Edit generic text parameter."""
        result = inquirer.text(
            message=f"Enter {param_name}:",
            default=str(self.params.get(param_key, '')),
        ).execute()

        if result != str(self.params.get(param_key)):
            self.params[param_key] = result
            self.modified_params.add(param_key)
            print(f"✓ {param_name} set to: {result}")

    def _get_batch_size_suggestions(self) -> Dict[str, int]:
        """Get batch size suggestions based on available RAM."""
        device_type = self.params.get('device', 'cpu')
        available_ram = self.params.get('_available_ram_gb', 4.0)

        return calculate_optimal_batch_size(
            device_type=device_type,
            available_ram_gb=available_ram,
            sequence_length=self.params.get('sequence_length', 84),
            num_features=50,  # Approximate
            hidden_size=self.params.get('hidden_size', 128)
        )

    def _auto_select_device(self):
        """Auto-select the best available device."""
        devices = self.device_manager.get_available_devices()

        # Priority order: cuda > mps > cpu
        if devices.get('cuda', {}).get('available'):
            self.params['device'] = 'cuda'
        elif devices.get('mps', {}).get('available'):
            self.params['device'] = 'mps'
        else:
            self.params['device'] = 'cpu'

        self.params['auto_device'] = True

    def _reset_all(self):
        """Reset all parameters to defaults."""
        print("\n✓ All parameters reset to defaults")
        self.params = self._get_default_params()
        self.modified_params = set()
        self._auto_select_device()

    def _validate_and_confirm(self) -> bool:
        """Validate configuration and get confirmation."""
        print("\n" + "=" * 80)
        print("📋 CONFIGURATION SUMMARY")
        print("=" * 80)

        # Check for warnings
        warnings = []

        # Check data files exist
        for param in ['spy_data', 'tsla_data', 'tsla_events']:
            path = self.params[param]
            if path and not Path(path).exists():
                warnings.append(f"⚠️  File not found: {path}")

        # Check year range
        if self.params['start_year'] >= self.params['end_year']:
            warnings.append(f"⚠️  Invalid year range: {self.params['start_year']}-{self.params['end_year']}")

        # Display key parameters
        print("\nKey Settings:")
        print(f"  Training Period : {self.params['start_year']}-{self.params['end_year']}")
        print(f"  Model Type      : {self.params['model_type']}")
        print(f"  Hidden Size     : {self.params['hidden_size']}")
        print(f"  Batch Size      : {self.params['batch_size']}")
        print(f"  Device          : {self.params['device']}")
        print(f"  Epochs          : {self.params['epochs']}")

        if self.modified_params:
            print(f"\nModified Parameters ({len(self.modified_params)}):")
            for param in sorted(self.modified_params):
                print(f"  - {param}")

        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  {warning}")

        # Get confirmation
        print("\n" + "─" * 80)
        confirm = inquirer.confirm(
            message="Start training with these settings?",
            default=True,
        ).execute()

        return confirm


def create_argparse_from_params(params: Dict[str, Any], args) -> Any:
    """Convert parameters dict back to argparse namespace."""
    # Update args with selected parameters
    for key, value in params.items():
        if not key.startswith('_'):  # Skip internal params
            setattr(args, key, value)
    return args


# Export for compatibility
__all__ = ['ArrowKeyParameterSelector', 'create_argparse_from_params']