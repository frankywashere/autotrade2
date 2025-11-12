"""
Interactive parameter selection system for ML training scripts.
Provides RAM-based batch size calculation and guided parameter configuration.
"""

import os
import sys
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import config
from src.ml.device_manager import DeviceManager


def calculate_optimal_batch_size(
    device_type: str,
    available_ram_gb: float,
    sequence_length: int = 84,
    num_features: int = 50,
    hidden_size: int = 128
) -> Dict[str, int]:
    """
    Calculate optimal batch size based on system resources.

    Returns dict with conservative, balanced, and aggressive options.
    """
    # Estimate memory per sample (in MB)
    # Formula: sequence_length × features × 4 bytes × gradient factor
    bytes_per_sample = sequence_length * num_features * 4 * 3  # 3x for model + gradients + optimizer
    mb_per_sample = bytes_per_sample / (1024 * 1024)

    # Add overhead for larger models
    if hidden_size > 256:
        mb_per_sample *= 1.5

    # Safety factors based on device type
    safety_factors = {
        'cpu': 0.3,   # Use 30% of available RAM
        'mps': 0.4,   # Use 40% (unified memory)
        'cuda': 0.6,  # Use 60% (dedicated VRAM)
    }
    safety_factor = safety_factors.get(device_type, 0.3)

    # Calculate max safe batch size
    usable_ram_mb = available_ram_gb * 1024 * safety_factor
    max_batch_size = int(usable_ram_mb / mb_per_sample)

    # Create tiered suggestions
    suggestions = {
        'conservative': max(8, min(max_batch_size // 2, 128)),
        'balanced': max(16, min(int(max_batch_size * 0.75), 256)),
        'aggressive': max(32, min(max_batch_size, 512))
    }

    # Round to nearest power of 2 for efficiency
    for key in suggestions:
        val = suggestions[key]
        suggestions[key] = 2 ** int(math.log2(val)) if val > 0 else 8

    suggestions['memory_per_sample_mb'] = mb_per_sample
    suggestions['max_safe'] = max_batch_size

    return suggestions


class InteractiveParameterSelector:
    """
    Interactive parameter selection system for training scripts.
    """

    def __init__(self, mode: str = 'standard'):
        """
        Initialize the parameter selector.

        Args:
            mode: 'standard' for train_model.py, 'lazy' for train_model_lazy.py
        """
        self.mode = mode
        self.device_manager = DeviceManager()
        self.params = self._get_default_params()
        self.modified_params = set()  # Track which params were changed

    def _get_default_params(self) -> Dict[str, Any]:
        """Get default parameters from config and standard defaults."""
        return {
            # Data parameters
            'input_timeframe': '1min',  # NEW: Timeframe for training
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
            'prediction_horizon_hours': config.PREDICTION_HORIZON_HOURS,

            # Device
            'device': None,  # Will be set during selection
            'auto_device': False,

            # GPU optimization
            'num_workers': 0,  # CPU threads for data loading
            'pin_memory': False,  # Pin memory for GPU transfers
        }

    def run(self) -> Dict[str, Any]:
        """Run the interactive parameter selection process."""
        self._display_welcome()
        self._display_system_info()

        # Ask if user wants to customize
        choice = self._initial_choice()
        if choice == 'defaults':
            # Just select device and use defaults
            self._auto_select_device()
            return self.params
        elif choice == 'arrow':
            # Use arrow-key navigation (new)
            return self._run_arrow_key_mode()
        elif choice == 'menu':
            # Use number-based menu selection
            return self._run_menu_based()
        else:
            # Cancelled
            sys.exit(0)

    def _display_welcome(self):
        """Display welcome message."""
        print("\n" + "=" * 80)
        print("🎛️  INTERACTIVE PARAMETER SELECTION")
        print("=" * 80)
        print(f"\nMode: {'Memory-efficient (lazy)' if self.mode == 'lazy' else 'Standard'} training")
        print("\nThis wizard will help you configure all training parameters.")
        print("Press Ctrl+C at any time to cancel.\n")

    def _display_system_info(self):
        """Display system information and capabilities."""
        print("Detecting system capabilities...\n")

        hardware_info = self.device_manager.detect_hardware()

        print("System Information:")
        print(f"  Platform: {hardware_info['platform']}")
        print(f"  Processor: {hardware_info['processor']}")
        print(f"  Python: {hardware_info['python_version']}")
        print(f"  PyTorch: {hardware_info['pytorch_version']}")

        # RAM info
        try:
            import psutil
            mem = psutil.virtual_memory()
            total_ram_gb = mem.total / 1024**3
            available_ram_gb = mem.available / 1024**3
            print(f"  RAM: {total_ram_gb:.1f} GB total, {available_ram_gb:.1f} GB available")
            self.params['_system_ram_gb'] = total_ram_gb
            self.params['_available_ram_gb'] = available_ram_gb
        except:
            print("  RAM: Unable to detect")
            self.params['_system_ram_gb'] = 8.0  # Assume 8GB
            self.params['_available_ram_gb'] = 4.0

        # GPU info
        if hardware_info.get('cuda_available'):
            print(f"  CUDA GPU: {hardware_info.get('cuda_device_name', 'Available')}")
        if hardware_info.get('mps_available'):
            print(f"  Apple Metal: Available ({hardware_info.get('apple_chip', 'Apple Silicon')})")

        print("\nPress Enter to continue...")
        input()

    def _initial_choice(self) -> str:
        """Ask if user wants to customize parameters."""
        print("\n" + "=" * 80)
        print("📋 CONFIGURATION OPTIONS")
        print("=" * 80)

        # Check if InquirerPy is available
        try:
            import InquirerPy
            arrow_available = True
        except ImportError:
            arrow_available = False

        print("\nCurrent configuration uses defaults from config.py")
        print("\nDo you want to:")
        print("  1. Use default configuration with auto-detected device (quick start)")

        if arrow_available:
            print("  2. Customize parameters with arrow-key navigation (recommended)")
            print("  3. Customize parameters with number-based menu")
            print("  4. Cancel and exit")
            valid_choices = ['1', '2', '3', '4']
        else:
            print("  2. Customize parameters (number-based menu)")
            print("  3. Cancel and exit")
            print("\n  Note: Install 'inquirerpy' for arrow-key navigation: pip install inquirerpy")
            valid_choices = ['1', '2', '3']

        while True:
            choice_prompt = f"\nChoice [1-{len(valid_choices)}]: "
            choice = input(choice_prompt).strip()

            if arrow_available:
                if choice == '1':
                    print("\n✓ Using default configuration with auto-detected device")
                    return 'defaults'
                elif choice == '2':
                    print("\n✓ Entering arrow-key parameter selection")
                    return 'arrow'
                elif choice == '3':
                    print("\n✓ Entering number-based parameter menu")
                    return 'menu'
                elif choice == '4':
                    print("\nCancelled.")
                    return 'cancel'
            else:
                if choice == '1':
                    print("\n✓ Using default configuration with auto-detected device")
                    return 'defaults'
                elif choice == '2':
                    print("\n✓ Entering parameter customization menu")
                    return 'menu'
                elif choice == '3':
                    print("\nCancelled.")
                    return 'cancel'

            print(f"Please enter {', '.join(valid_choices[:-1])} or {valid_choices[-1]}")

    def _run_arrow_key_mode(self) -> Dict[str, Any]:
        """Run the arrow-key based parameter selection."""
        try:
            from src.ml.interactive_params_arrow import ArrowKeyParameterSelector

            # Create arrow-key selector with same mode
            arrow_selector = ArrowKeyParameterSelector(mode=self.mode)

            # Copy over detected system info
            arrow_selector.params['_system_ram_gb'] = self.params.get('_system_ram_gb', 8.0)
            arrow_selector.params['_available_ram_gb'] = self.params.get('_available_ram_gb', 4.0)

            # Run arrow-key selection
            return arrow_selector.run()
        except ImportError as e:
            print(f"\n⚠️  Error loading arrow-key interface: {e}")
            print("Falling back to number-based menu...")
            return self._run_menu_based()

    def _run_menu_based(self) -> Dict[str, Any]:
        """Run the menu-based parameter selection system."""
        while True:
            # Display all parameters in a menu
            self._display_parameter_menu()

            # Get user selection
            print("\nEnter parameter numbers to modify (e.g., '1,5,8' or '5-10' or 'all')")
            print("Or enter 'd' for done, 'r' to reset all, 'q' to quit")
            selection = input("\nSelection: ").strip().lower()

            if selection == 'd':
                # Validate and confirm
                if self._validate_and_confirm():
                    return self.params
            elif selection == 'q':
                print("\nCancelled.")
                sys.exit(0)
            elif selection == 'r':
                print("\n✓ Reset all parameters to defaults")
                self.params = self._get_default_params()
                self.modified_params = set()
            elif selection == 'all':
                # Modify all parameters
                param_indices = list(range(1, 24))
                self._modify_selected_params(param_indices)
            else:
                # Parse and modify selected parameters
                param_indices = self._parse_selection(selection, 23)
                if param_indices:
                    self._modify_selected_params(param_indices)
                else:
                    print("Invalid selection. Please try again.")

    def _display_parameter_menu(self):
        """Display all parameters in a categorized menu format."""
        print("\n" + "=" * 80)
        print("📊 PARAMETER CONFIGURATION MENU")
        print("=" * 80)

        # Create parameter catalog with categories
        catalog = self._get_parameter_catalog()

        param_num = 1
        for category, params in catalog.items():
            print(f"\n{category}")
            print("-" * len(category))

            for param_key, info in params:
                current_value = self.params.get(param_key, info.get('default', 'None'))

                # Special formatting for certain values
                if param_key == 'macro_api_key':
                    display_value = 'Not set (local data available)' if not current_value else 'Set'
                elif param_key == 'device':
                    display_value = current_value or 'Auto-detect'
                elif isinstance(current_value, bool):
                    display_value = 'Yes' if current_value else 'No'
                else:
                    display_value = str(current_value)

                # Mark modified parameters
                modified_marker = " *" if param_key in self.modified_params else ""

                # Format the line
                print(f"  {param_num:2d}. {info['name']:<25} : {display_value:<20}{modified_marker}")
                if 'hint' in info:
                    print(f"      └─ {info['hint']}")

                param_num += 1

        print("\n" + "-" * 80)
        print("* = Modified from default")

        # Show batch size suggestions if not modified
        if 'batch_size' not in self.modified_params:
            self._show_batch_size_suggestions()

    def _get_parameter_catalog(self) -> Dict[str, List[Tuple[str, Dict]]]:
        """Get categorized parameter catalog."""
        return {
            "📁 DATA FILES": [
                ('input_timeframe', {
                    'name': 'Input timeframe',
                    'type': 'choice',
                    'hint': 'Timeframe for multi-scale training',
                    'default': '1min'
                }),
                ('spy_data', {
                    'name': 'SPY data file',
                    'hint': 'Historical SPY data (auto-set by timeframe)',
                    'default': 'data/SPY_1min.csv'
                }),
                ('tsla_data', {
                    'name': 'TSLA data file',
                    'hint': 'Historical TSLA data (auto-set by timeframe)',
                    'default': 'data/TSLA_1min.csv'
                }),
                ('tsla_events', {
                    'name': 'TSLA events file',
                    'hint': 'Earnings, deliveries, major events',
                    'default': 'data/tsla_events_REAL.csv'
                }),
                ('macro_api_key', {
                    'name': 'Macro API key',
                    'hint': 'Optional - local macro data exists in data/',
                    'default': None
                }),
            ],
            "📅 TRAINING PERIOD": [
                ('start_year', {
                    'name': 'Start year',
                    'hint': 'Earlier = more data, slower training',
                    'default': config.ML_TRAIN_START_YEAR
                }),
                ('end_year', {
                    'name': 'End year',
                    'hint': 'Reserve 2024 for validation',
                    'default': config.ML_TRAIN_END_YEAR
                }),
            ],
            "🧠 MODEL ARCHITECTURE": [
                ('model_type', {
                    'name': 'Model type',
                    'hint': 'CfC = Liquid Neural Network',
                    'default': config.ML_MODEL_TYPE
                }),
                ('hidden_size', {
                    'name': 'Hidden size',
                    'hint': 'Complexity: 64=simple, 512=complex patterns',
                    'default': config.LNN_HIDDEN_SIZE
                }),
                ('num_layers', {
                    'name': 'Number of layers',
                    'hint': 'Depth of network',
                    'default': config.LNN_NUM_LAYERS
                }),
                ('sequence_length', {
                    'name': 'Sequence length',
                    'hint': 'Minutes of history per prediction',
                    'default': config.ML_SEQUENCE_LENGTH
                }),
            ],
            "⚙️ TRAINING PARAMETERS": [
                ('epochs', {
                    'name': 'Training epochs',
                    'hint': 'Full passes through data',
                    'default': config.ML_EPOCHS
                }),
                ('pretrain_epochs', {
                    'name': 'Pretrain epochs',
                    'hint': 'Self-supervised pretraining',
                    'default': 10
                }),
                ('batch_size', {
                    'name': 'Batch size',
                    'hint': 'Samples per iteration (RAM-aware)',
                    'default': config.ML_BATCH_SIZE or 16
                }),
                ('learning_rate', {
                    'name': 'Learning rate',
                    'hint': 'Step size for optimization',
                    'default': config.LNN_LEARNING_RATE
                }),
                ('validation_split', {
                    'name': 'Validation split',
                    'hint': 'Fraction for validation',
                    'default': config.ML_VALIDATION_SPLIT
                }),
            ],
            "📊 FEATURE FLAGS": [
                ('use_channel_features', {
                    'name': 'Channel features',
                    'hint': 'Price channels & breakouts',
                    'default': config.USE_CHANNEL_FEATURES
                }),
                ('use_rsi_features', {
                    'name': 'RSI features',
                    'hint': 'Momentum indicators',
                    'default': config.USE_RSI_FEATURES
                }),
                ('use_correlation_features', {
                    'name': 'Correlation features',
                    'hint': 'SPY/TSLA correlation',
                    'default': config.USE_CORRELATION_FEATURES
                }),
                ('use_event_features', {
                    'name': 'Event features',
                    'hint': 'Earnings, macro events',
                    'default': config.USE_EVENT_FEATURES
                }),
                ('prediction_horizon_hours', {
                    'name': 'Prediction horizon',
                    'hint': 'BARS ahead to predict (15min: 24 bars=6hrs, 1hour: 24 bars=24hrs, daily: 24 bars=24 days)',
                    'default': config.PREDICTION_HORIZON_HOURS
                }),
            ],
            "🖥️ COMPUTE DEVICE": [
                ('device', {
                    'name': 'Device',
                    'hint': 'cpu/cuda/mps or auto-detect',
                    'default': None
                }),
            ],
            "🚀 GPU OPTIMIZATION": [
                ('num_workers', {
                    'name': 'Data loading workers',
                    'hint': 'CPU threads (0=main only, 2=recommended for GPU)',
                    'default': 0
                }),
                ('pin_memory', {
                    'name': 'Pin memory',
                    'hint': 'Faster GPU transfers (~10% speedup)',
                    'default': False
                }),
            ]
        }

    def _parse_selection(self, selection: str, max_num: int) -> List[int]:
        """Parse user selection string like '1,5,8' or '5-10'."""
        indices = []
        try:
            # Handle comma-separated and ranges
            parts = selection.split(',')
            for part in parts:
                part = part.strip()
                if '-' in part:
                    # Handle range like '5-10'
                    start, end = part.split('-')
                    start, end = int(start.strip()), int(end.strip())
                    indices.extend(range(start, end + 1))
                else:
                    # Single number
                    indices.append(int(part))

            # Validate indices
            indices = [i for i in indices if 1 <= i <= max_num]
            return sorted(set(indices))
        except:
            return []

    def _modify_selected_params(self, param_indices: List[int]):
        """Modify the selected parameters."""
        catalog = self._get_parameter_catalog()

        # Create a flat list of parameters
        flat_params = []
        for category, params in catalog.items():
            flat_params.extend(params)

        for idx in param_indices:
            if 1 <= idx <= len(flat_params):
                param_key, info = flat_params[idx - 1]
                self._modify_single_param(param_key, info)

    def _modify_single_param(self, param_key: str, info: Dict):
        """Modify a single parameter."""
        print(f"\n{'=' * 60}")
        print(f"Modifying: {info['name']}")
        print(f"Current value: {self.params.get(param_key, info.get('default', 'None'))}")
        if 'hint' in info:
            print(f"Hint: {info['hint']}")

        # Special handling for different parameter types
        if param_key == 'input_timeframe':
            # Timeframe selection (new for multi-scale)
            timeframes = ['1min', '5min', '15min', '30min', '1hour', '2hour', '3hour', '4hour', 'daily', 'weekly', 'monthly', '3month']
            print("\nAvailable timeframes:")
            for i, tf in enumerate(timeframes, 1):
                current = " (current)" if tf == self.params.get('input_timeframe') else ""
                print(f"  {i}. {tf}{current}")

            value = input(f"Enter timeframe number [1-{len(timeframes)}] or name: ").strip()

            # Handle numeric or name input
            if value.isdigit():
                idx = int(value) - 1
                if 0 <= idx < len(timeframes):
                    selected_tf = timeframes[idx]
                else:
                    print("Invalid selection. Keeping current value.")
                    return
            elif value in timeframes:
                selected_tf = value
            else:
                print("Invalid selection. Keeping current value.")
                return

            # Update timeframe
            self.params['input_timeframe'] = selected_tf
            self.modified_params.add('input_timeframe')

            # Auto-update data file paths
            self.params['spy_data'] = f'data/SPY_{selected_tf}.csv'
            self.params['tsla_data'] = f'data/TSLA_{selected_tf}.csv'
            print(f"✓ Timeframe set to: {selected_tf}")
            print(f"  Auto-updated: spy_data = {self.params['spy_data']}")
            print(f"  Auto-updated: tsla_data = {self.params['tsla_data']}")

        elif param_key == 'macro_api_key':
            print("\nNote: Local macro economic data is available in data/ directory.")
            print("You can use the local data without an API key.")
            value = input("Enter API key (or press Enter to skip): ").strip()
            if value:
                self.params[param_key] = value
                self.modified_params.add(param_key)

        elif param_key == 'device':
            devices = self.device_manager.get_available_devices()
            print("\nAvailable devices:")
            device_options = []
            for name, info in devices.items():
                if info['available']:
                    device_options.append(name)
                    print(f"  - {name}")
            print("  - auto (auto-detect best)")

            value = input(f"Enter device [{'/'.join(device_options)}/auto]: ").strip().lower()
            if value == 'auto' or value == '':
                self._auto_select_device()
            elif value in device_options:
                self.params['device'] = value
                self.modified_params.add('device')

        elif param_key in ['use_channel_features', 'use_rsi_features',
                           'use_correlation_features', 'use_event_features', 'pin_memory']:
            # Boolean parameters
            value = input("Enable? [y/n]: ").strip().lower()
            if value in ['y', 'yes', '1', 'true']:
                self.params[param_key] = True
                self.modified_params.add(param_key)
            elif value in ['n', 'no', '0', 'false']:
                self.params[param_key] = False
                self.modified_params.add(param_key)

        elif param_key in ['start_year', 'end_year']:
            # Year parameters
            min_year = 2010 if param_key == 'start_year' else self.params.get('start_year', 2010)
            max_year = 2023 if param_key == 'start_year' else 2024

            value = input(f"Enter year [{min_year}-{max_year}]: ").strip()
            try:
                year = int(value)
                if min_year <= year <= max_year:
                    self.params[param_key] = year
                    self.modified_params.add(param_key)
                else:
                    print(f"Invalid year. Must be between {min_year} and {max_year}.")
            except ValueError:
                print("Invalid input. Keeping current value.")

        elif param_key == 'batch_size':
            # Show batch size suggestions
            self._show_batch_size_suggestions()
            value = input("Enter batch size (or 'auto' for suggestion): ").strip()
            if value.lower() == 'auto':
                suggestions = self._get_batch_size_suggestions()
                self.params[param_key] = suggestions['balanced']
                self.modified_params.add(param_key)
                print(f"✓ Set batch size to {suggestions['balanced']} (balanced)")
            else:
                try:
                    batch_size = int(value)
                    if 8 <= batch_size <= 8192:
                        self.params[param_key] = batch_size
                        self.modified_params.add(param_key)
                        if batch_size > 1024:
                            print(f"⚠️  Warning: Large batch size ({batch_size}) - ensure your GPU has enough VRAM!")
                    else:
                        print("Invalid batch size (must be 8-8192). Keeping current value.")
                except ValueError:
                    print("Invalid input. Keeping current value.")

        elif param_key in ['hidden_size', 'num_layers', 'sequence_length',
                          'epochs', 'pretrain_epochs', 'prediction_horizon_hours', 'num_workers']:
            # Integer parameters
            value = input(f"Enter value: ").strip()
            try:
                int_value = int(value)
                # num_workers can be 0, others must be > 0
                min_val = 0 if param_key == 'num_workers' else 1
                if int_value >= min_val:
                    self.params[param_key] = int_value
                    self.modified_params.add(param_key)
                else:
                    print(f"Value must be >= {min_val}. Keeping current value.")
            except ValueError:
                print("Invalid input. Keeping current value.")

        elif param_key == 'learning_rate':
            # Float parameter
            value = input("Enter learning rate (e.g., 0.001): ").strip()
            try:
                lr = float(value)
                if 0 < lr < 1:
                    self.params[param_key] = lr
                    self.modified_params.add(param_key)
                else:
                    print("Learning rate must be between 0 and 1.")
            except ValueError:
                print("Invalid input. Keeping current value.")

        elif param_key == 'validation_split':
            # Float parameter (0-1)
            value = input("Enter validation split (0.0-0.5): ").strip()
            try:
                split = float(value)
                if 0 <= split <= 0.5:
                    self.params[param_key] = split
                    self.modified_params.add(param_key)
                else:
                    print("Validation split must be between 0 and 0.5.")
            except ValueError:
                print("Invalid input. Keeping current value.")

        else:
            # String parameters (file paths)
            value = input("Enter value: ").strip()
            if value:
                self.params[param_key] = value
                self.modified_params.add(param_key)

    def _show_batch_size_suggestions(self):
        """Show batch size suggestions based on RAM."""
        suggestions = self._get_batch_size_suggestions()

        print(f"\n💡 Batch size suggestions (based on {self.params.get('_available_ram_gb', 4.0):.1f} GB available RAM):")
        print(f"   Conservative : {suggestions['conservative']} (safe, slower)")
        print(f"   Balanced     : {suggestions['balanced']} (recommended)")
        print(f"   Aggressive   : {suggestions['aggressive']} (fast, may OOM)")

    def _get_batch_size_suggestions(self) -> Dict[str, int]:
        """Get batch size suggestions."""
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
            print("✓ Auto-selected: CUDA GPU")
        elif devices.get('mps', {}).get('available'):
            self.params['device'] = 'mps'
            print("✓ Auto-selected: Apple Metal (MPS)")
        else:
            self.params['device'] = 'cpu'
            print("✓ Auto-selected: CPU")

        self.params['auto_device'] = True

    def _select_data_params(self):
        """Select data file parameters."""
        print("\n" + "=" * 80)
        print("📁 DATA PARAMETERS")
        print("=" * 80)
        print("\nSpecify the data files to use for training.\n")

        # SPY data
        print(f"1. SPY 1-minute data")
        print(f"   Current: {self.params['spy_data']}")
        print(f"   Description: Historical SPY data for correlation analysis")
        if self._ask_modify("SPY data path"):
            new_path = input("   New path: ").strip()
            if new_path:
                self.params['spy_data'] = new_path
                self.modified_params.add('spy_data')

        # TSLA data
        print(f"\n2. TSLA 1-minute data")
        print(f"   Current: {self.params['tsla_data']}")
        print(f"   Description: Historical TSLA data for prediction")
        if self._ask_modify("TSLA data path"):
            new_path = input("   New path: ").strip()
            if new_path:
                self.params['tsla_data'] = new_path
                self.modified_params.add('tsla_data')

        # Events file
        print(f"\n3. TSLA events file")
        print(f"   Current: {self.params['tsla_events']}")
        print(f"   Description: Earnings, deliveries, and major TSLA events")
        if self._ask_modify("Events file path"):
            new_path = input("   New path: ").strip()
            if new_path:
                self.params['tsla_events'] = new_path
                self.modified_params.add('tsla_events')

        # Macro API key (optional)
        print(f"\n4. Macro events API key")
        print(f"   Current: {'Set' if self.params['macro_api_key'] else 'None (optional)'}")
        print(f"   Description: Optional FRED/economic calendar API key")
        if self._ask_modify("Macro API key"):
            new_key = input("   API key (or Enter to skip): ").strip()
            if new_key:
                self.params['macro_api_key'] = new_key
                self.modified_params.add('macro_api_key')

    def _select_training_period(self):
        """Select training period parameters."""
        print("\n" + "=" * 80)
        print("📅 TRAINING PERIOD")
        print("=" * 80)
        print("\nSpecify the date range for training data.\n")

        # Start year
        print(f"1. Start year")
        print(f"   Current: {self.params['start_year']}")
        print(f"   Range: 2010-2023")
        print(f"   Description: Earlier years = more data but slower training")
        if self._ask_modify("Start year"):
            while True:
                try:
                    year = int(input("   New year [2010-2023]: ").strip())
                    if 2010 <= year <= 2023:
                        self.params['start_year'] = year
                        self.modified_params.add('start_year')
                        break
                    else:
                        print("   Please enter a year between 2010 and 2023")
                except ValueError:
                    print("   Please enter a valid year")

        # End year
        print(f"\n2. End year")
        print(f"   Current: {self.params['end_year']}")
        print(f"   Range: {self.params['start_year']}-2024")
        print(f"   Description: Reserve 2024 for backtesting validation")
        if self._ask_modify("End year"):
            while True:
                try:
                    year = int(input(f"   New year [{self.params['start_year']}-2024]: ").strip())
                    if self.params['start_year'] <= year <= 2024:
                        self.params['end_year'] = year
                        self.modified_params.add('end_year')
                        break
                    else:
                        print(f"   Please enter a year between {self.params['start_year']} and 2024")
                except ValueError:
                    print("   Please enter a valid year")

        years_of_data = self.params['end_year'] - self.params['start_year'] + 1
        print(f"\n✓ Training on {years_of_data} years of data")

    def _select_model_architecture(self):
        """Select model architecture parameters."""
        print("\n" + "=" * 80)
        print("🧠 MODEL ARCHITECTURE")
        print("=" * 80)
        print("\nConfigure the neural network architecture.\n")

        # Model type
        print(f"1. Model type")
        print(f"   Current: {self.params['model_type']}")
        print(f"   Options: LNN (Liquid Neural Network), LSTM (Classic RNN)")
        print(f"   Description: LNN is better for time series, LSTM is more traditional")
        if self._ask_modify("Model type"):
            print("   Select: 1=LNN, 2=LSTM")
            choice = input("   Choice [1-2]: ").strip()
            if choice == '2':
                self.params['model_type'] = 'LSTM'
                self.modified_params.add('model_type')
            else:
                self.params['model_type'] = 'LNN'
                if choice != '1':
                    print("   Defaulting to LNN")

        # Hidden size
        print(f"\n2. Hidden layer size")
        print(f"   Current: {self.params['hidden_size']}")
        print(f"   Range: 32-512 (powers of 2 recommended)")
        print(f"   Description: Larger = more capacity but slower. 128 is balanced.")
        if self._ask_modify("Hidden size"):
            while True:
                try:
                    size = int(input("   New size [32-512]: ").strip())
                    if 32 <= size <= 512:
                        self.params['hidden_size'] = size
                        self.modified_params.add('hidden_size')
                        break
                    else:
                        print("   Please enter a value between 32 and 512")
                except ValueError:
                    print("   Please enter a valid number")

        # Number of layers
        print(f"\n3. Number of layers")
        print(f"   Current: {self.params['num_layers']}")
        print(f"   Range: 1-4")
        print(f"   Description: Deeper networks learn complex patterns but may overfit")
        if self._ask_modify("Number of layers"):
            while True:
                try:
                    layers = int(input("   New value [1-4]: ").strip())
                    if 1 <= layers <= 4:
                        self.params['num_layers'] = layers
                        self.modified_params.add('num_layers')
                        break
                    else:
                        print("   Please enter a value between 1 and 4")
                except ValueError:
                    print("   Please enter a valid number")

        # Sequence length
        print(f"\n4. Sequence length (timesteps)")
        print(f"   Current: {self.params['sequence_length']} hours")
        print(f"   Range: 24-336 (1 day to 2 weeks)")
        print(f"   Description: How much history to consider. Longer = more memory.")
        if self._ask_modify("Sequence length"):
            while True:
                try:
                    length = int(input("   New value [24-336]: ").strip())
                    if 24 <= length <= 336:
                        self.params['sequence_length'] = length
                        self.modified_params.add('sequence_length')
                        break
                    else:
                        print("   Please enter a value between 24 and 336")
                except ValueError:
                    print("   Please enter a valid number")

    def _select_training_params(self):
        """Select training hyperparameters including batch size."""
        print("\n" + "=" * 80)
        print("🎯 TRAINING HYPERPARAMETERS")
        print("=" * 80)
        print("\nConfigure training parameters.\n")

        # Calculate batch size suggestions first
        available_ram = self.params.get('_available_ram_gb', 4.0)
        device_type = 'cpu'  # Will be updated based on device selection

        batch_suggestions = calculate_optimal_batch_size(
            device_type=device_type,
            available_ram_gb=available_ram,
            sequence_length=self.params['sequence_length'],
            num_features=50,  # Approximate
            hidden_size=self.params['hidden_size']
        )

        # Training epochs
        print(f"1. Training epochs")
        print(f"   Current: {self.params['epochs']}")
        print(f"   Range: 10-200")
        print(f"   Description: More epochs = better learning but risk of overfitting")
        if self._ask_modify("Training epochs"):
            while True:
                try:
                    epochs = int(input("   New value [10-200]: ").strip())
                    if 10 <= epochs <= 200:
                        self.params['epochs'] = epochs
                        self.modified_params.add('epochs')
                        break
                    else:
                        print("   Please enter a value between 10 and 200")
                except ValueError:
                    print("   Please enter a valid number")

        # Pretraining epochs
        print(f"\n2. Pretraining epochs")
        print(f"   Current: {self.params['pretrain_epochs']}")
        print(f"   Range: 0-50 (0 to skip)")
        print(f"   Description: Self-supervised learning helps model understand patterns")
        if self._ask_modify("Pretraining epochs"):
            while True:
                try:
                    epochs = int(input("   New value [0-50]: ").strip())
                    if 0 <= epochs <= 50:
                        self.params['pretrain_epochs'] = epochs
                        self.modified_params.add('pretrain_epochs')
                        break
                    else:
                        print("   Please enter a value between 0 and 50")
                except ValueError:
                    print("   Please enter a valid number")

        # Batch size with auto-calculation
        print(f"\n3. Batch size")
        print(f"   Current: {self.params['batch_size']}")
        print(f"\n   🤖 AUTO-CALCULATED SUGGESTIONS (based on {available_ram:.1f} GB available RAM):")
        print(f"      Conservative: {batch_suggestions['conservative']} (safest)")
        print(f"      Balanced: {batch_suggestions['balanced']} (recommended)")
        print(f"      Aggressive: {batch_suggestions['aggressive']} (fastest)")
        print(f"\n   Memory per sample: ~{batch_suggestions['memory_per_sample_mb']:.1f} MB")
        print(f"   Description: Larger = faster training but more memory")

        print(f"\n   Options:")
        print(f"   1. Use balanced ({batch_suggestions['balanced']})")
        print(f"   2. Use conservative ({batch_suggestions['conservative']})")
        print(f"   3. Use aggressive ({batch_suggestions['aggressive']})")
        print(f"   4. Enter custom value")
        print(f"   5. Keep current ({self.params['batch_size']})")

        choice = input("   Choice [1-5]: ").strip()
        if choice == '1':
            self.params['batch_size'] = batch_suggestions['balanced']
            self.modified_params.add('batch_size')
            print(f"   ✓ Batch size set to {batch_suggestions['balanced']}")
        elif choice == '2':
            self.params['batch_size'] = batch_suggestions['conservative']
            self.modified_params.add('batch_size')
            print(f"   ✓ Batch size set to {batch_suggestions['conservative']}")
        elif choice == '3':
            self.params['batch_size'] = batch_suggestions['aggressive']
            self.modified_params.add('batch_size')
            print(f"   ✓ Batch size set to {batch_suggestions['aggressive']}")
        elif choice == '4':
            while True:
                try:
                    batch_size = int(input("   Custom batch size [8-8192]: ").strip())
                    if 8 <= batch_size <= 8192:
                        self.params['batch_size'] = batch_size
                        self.modified_params.add('batch_size')
                        print(f"   ✓ Batch size set to {batch_size}")
                        if batch_size > 1024:
                            print(f"   ⚠️  Warning: Large batch size ({batch_size}) - ensure your GPU has enough VRAM!")
                        break
                    else:
                        print("   Please enter a value between 8 and 8192")
                except ValueError:
                    print("   Please enter a valid number")
        else:
            print(f"   ✓ Keeping batch size at {self.params['batch_size']}")

        # Learning rate
        print(f"\n4. Learning rate")
        print(f"   Current: {self.params['learning_rate']}")
        print(f"   Range: 0.00001-0.01")
        print(f"   Description: Higher = faster learning but less stable. 0.001 is standard.")
        if self._ask_modify("Learning rate"):
            while True:
                try:
                    lr = float(input("   New value [0.00001-0.01]: ").strip())
                    if 0.00001 <= lr <= 0.01:
                        self.params['learning_rate'] = lr
                        self.modified_params.add('learning_rate')
                        break
                    else:
                        print("   Please enter a value between 0.00001 and 0.01")
                except ValueError:
                    print("   Please enter a valid number")

        # Validation split
        print(f"\n5. Validation split")
        print(f"   Current: {self.params['validation_split']} ({int(self.params['validation_split']*100)}%)")
        print(f"   Range: 0.05-0.3 (5%-30%)")
        print(f"   Description: Portion of data reserved for validation")
        if self._ask_modify("Validation split"):
            while True:
                try:
                    split = float(input("   New value [0.05-0.3]: ").strip())
                    if 0.05 <= split <= 0.3:
                        self.params['validation_split'] = split
                        self.modified_params.add('validation_split')
                        break
                    else:
                        print("   Please enter a value between 0.05 and 0.3")
                except ValueError:
                    print("   Please enter a valid number")

    def _select_feature_params(self):
        """Select feature parameters."""
        print("\n" + "=" * 80)
        print("📊 FEATURE PARAMETERS")
        print("=" * 80)
        print("\nSelect which features to include in the model.\n")

        # Channel features
        print(f"1. Linear regression channels")
        print(f"   Current: {'Enabled' if self.params['use_channel_features'] else 'Disabled'}")
        print(f"   Description: Trend detection using regression channels")
        if self._ask_modify("Channel features"):
            self.params['use_channel_features'] = self._ask_yes_no("   Enable channel features?")
            self.modified_params.add('use_channel_features')

        # RSI features
        print(f"\n2. RSI indicators")
        print(f"   Current: {'Enabled' if self.params['use_rsi_features'] else 'Disabled'}")
        print(f"   Description: Momentum indicators (oversold/overbought)")
        if self._ask_modify("RSI features"):
            self.params['use_rsi_features'] = self._ask_yes_no("   Enable RSI features?")
            self.modified_params.add('use_rsi_features')

        # Correlation features
        print(f"\n3. SPY-TSLA correlation")
        print(f"   Current: {'Enabled' if self.params['use_correlation_features'] else 'Disabled'}")
        print(f"   Description: Correlation between SPY and TSLA movements")
        if self._ask_modify("Correlation features"):
            self.params['use_correlation_features'] = self._ask_yes_no("   Enable correlation features?")
            self.modified_params.add('use_correlation_features')

        # Event features
        print(f"\n4. Event features")
        print(f"   Current: {'Enabled' if self.params['use_event_features'] else 'Disabled'}")
        print(f"   Description: Earnings, deliveries, and macro events")
        if self._ask_modify("Event features"):
            self.params['use_event_features'] = self._ask_yes_no("   Enable event features?")
            self.modified_params.add('use_event_features')

        # Prediction horizon
        print(f"\n5. Prediction horizon")
        print(f"   Current: {self.params['prediction_horizon_hours']} hours")
        print(f"   Range: 1-72 hours")
        print(f"   Description: How far ahead to predict prices")
        if self._ask_modify("Prediction horizon"):
            while True:
                try:
                    hours = int(input("   New value [1-72]: ").strip())
                    if 1 <= hours <= 72:
                        self.params['prediction_horizon_hours'] = hours
                        self.modified_params.add('prediction_horizon_hours')
                        break
                    else:
                        print("   Please enter a value between 1 and 72")
                except ValueError:
                    print("   Please enter a valid number")

    def _select_device_params(self):
        """Select device parameters."""
        print("\n" + "=" * 80)
        print("🖥️  DEVICE SELECTION")
        print("=" * 80)

        # Use the existing device manager's interactive selection
        device = self.device_manager.select_device_interactive()
        self.params['device'] = device.type
        self.params['auto_device'] = False
        self.modified_params.add('device')

    def _validate_and_confirm(self) -> bool:
        """Validate parameters and get final confirmation."""
        print("\n" + "=" * 80)
        print("✅ PARAMETER REVIEW & CONFIRMATION")
        print("=" * 80)

        # Validate file paths
        validation_passed = True
        warnings = []

        # Check data files exist
        for param in ['spy_data', 'tsla_data', 'tsla_events']:
            path = self.params[param]
            if path and not Path(path).exists():
                warnings.append(f"⚠️  File not found: {path}")
                validation_passed = False

        # Check year range
        if self.params['start_year'] >= self.params['end_year']:
            warnings.append(f"⚠️  Invalid year range: {self.params['start_year']}-{self.params['end_year']}")
            validation_passed = False

        # Estimate memory usage
        batch_memory_mb = (
            self.params['batch_size'] *
            self.params['sequence_length'] *
            50 * 4 * 3 / (1024 * 1024)
        )
        total_memory_gb = (1500 + batch_memory_mb) / 1024  # Base overhead + batch

        available_ram = self.params.get('_available_ram_gb', 4.0)
        if total_memory_gb > available_ram * 0.8:
            warnings.append(f"⚠️  High memory usage: {total_memory_gb:.1f} GB needed, {available_ram:.1f} GB available")

        # Display configuration summary
        print("\nYour Configuration:")
        print("┌" + "─" * 30 + "┬" + "─" * 20 + "┬" + "─" * 15 + "┐")
        print("│ Parameter                    │ Value              │ Status        │")
        print("├" + "─" * 30 + "┼" + "─" * 20 + "┼" + "─" * 15 + "┤")

        # Display key parameters with modification status
        params_to_show = [
            ('Model type', 'model_type'),
            ('Hidden size', 'hidden_size'),
            ('Sequence length', 'sequence_length'),
            ('Training epochs', 'epochs'),
            ('Pretrain epochs', 'pretrain_epochs'),
            ('Batch size', 'batch_size'),
            ('Learning rate', 'learning_rate'),
            ('Device', 'device'),
            ('Training years', None),  # Special case
        ]

        for display_name, param_key in params_to_show:
            if param_key is None:
                # Special case for year range
                value = f"{self.params['start_year']}-{self.params['end_year']}"
                status = "Modified" if 'start_year' in self.modified_params else "Default"
            else:
                value = str(self.params[param_key])
                status = "Modified" if param_key in self.modified_params else "Default"

            # Truncate long values
            if len(value) > 18:
                value = value[:15] + "..."

            print(f"│ {display_name:<28} │ {value:<18} │ {status:<13} │")

        print("└" + "─" * 30 + "┴" + "─" * 20 + "┴" + "─" * 15 + "┘")

        # Estimate training time
        years_of_data = self.params['end_year'] - self.params['start_year'] + 1
        estimated_minutes = (
            years_of_data * 10 +  # Base time per year
            self.params['epochs'] * 0.5 +  # Time per epoch
            self.params['pretrain_epochs'] * 0.2  # Pretrain time
        )

        if self.params['device'] == 'mps' or self.params['device'] == 'cuda':
            estimated_minutes /= 3  # GPU acceleration

        print(f"\nEstimated training time: {int(estimated_minutes)}-{int(estimated_minutes*1.5)} minutes")
        print(f"Peak memory usage: ~{total_memory_gb:.1f} GB")

        # Display warnings
        if warnings:
            print("\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  {warning}")

        # Validation status
        print("\nValidation checks:")
        if validation_passed:
            print("  ✓ All checks passed")
        else:
            print("  ✗ Some checks failed - review warnings above")

        # Get confirmation
        print("\nDo you want to proceed with this configuration?")
        print("  1. Yes, start training")
        print("  2. No, go back and modify")
        print("  3. Cancel")

        while True:
            choice = input("\nChoice [1-3]: ").strip()
            if choice == '1':
                return True
            elif choice == '2':
                return False
            elif choice == '3':
                print("\nCancelled.")
                sys.exit(0)
            else:
                print("Please enter 1, 2, or 3")

    def _ask_modify(self, param_name: str) -> bool:
        """Ask if user wants to modify a parameter."""
        response = input(f"   Modify {param_name}? [y/N]: ").strip().lower()
        return response == 'y'

    def _ask_yes_no(self, question: str) -> bool:
        """Ask a yes/no question."""
        while True:
            response = input(f"{question} [y/n]: ").strip().lower()
            if response == 'y':
                return True
            elif response == 'n':
                return False
            else:
                print("   Please enter 'y' or 'n'")


def create_argparse_from_params(params: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Convert interactive parameters to argparse namespace.

    Args:
        params: Dictionary of parameters from interactive selection
        args: Original argparse namespace to update

    Returns:
        Updated argparse namespace
    """
    # Map interactive params to argparse attributes
    mapping = {
        'input_timeframe': 'input_timeframe',  # Multi-scale training parameter
        'spy_data': 'spy_data',
        'tsla_data': 'tsla_data',
        'tsla_events': 'tsla_events',
        'macro_api_key': 'macro_api_key',
        'start_year': 'start_year',
        'end_year': 'end_year',
        'model_type': 'model_type',
        'hidden_size': 'hidden_size',
        'sequence_length': 'sequence_length',  # Multi-scale parameter
        'prediction_horizon_hours': 'prediction_horizon',  # Note: Actually BARS not hours!
        'epochs': 'epochs',
        'pretrain_epochs': 'pretrain_epochs',
        'batch_size': 'batch_size',
        'learning_rate': 'lr',  # Note: different name in argparse
        'validation_split': None,  # Not in argparse, handled in config
        'device': 'device',
        'auto_device': 'auto_device',
        'num_workers': 'num_workers',  # GPU optimization parameter
        'pin_memory': 'pin_memory',  # GPU optimization parameter
    }

    # Update args with selected parameters
    for param_key, arg_key in mapping.items():
        if arg_key and param_key in params:
            setattr(args, arg_key, params[param_key])

    # Handle special cases
    if params.get('device'):
        args.device = params['device']
        args.auto_device = False

    # Store additional params that aren't in argparse
    args.interactive_params = params

    return args


if __name__ == '__main__':
    # Test the interactive selector
    selector = InteractiveParameterSelector()
    params = selector.run()
    print("\nFinal parameters:")
    for key, value in params.items():
        if not key.startswith('_'):  # Skip internal params
            print(f"  {key}: {value}")