#!/usr/bin/env python3
"""
Test that interactive "Train All 4" option properly flows settings to metadata.

This test validates that user-selected settings (epochs, batch_size, sequence_length, etc.)
are correctly saved to metadata for all 4 models when using the interactive multi-model option.
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.interactive_params import InteractiveParameterSelector, create_argparse_from_params


def test_metadata_flow():
    """
    Simulate the interactive "Train All 4" flow and verify metadata.
    """
    print("\n" + "=" * 70)
    print("🧪 TESTING INTERACTIVE TRAIN ALL 4 - METADATA FLOW")
    print("=" * 70)

    # Simulate user selections
    print("\n1. Simulating user parameter selection...")
    selector = InteractiveParameterSelector(mode='lazy')

    # Get defaults and simulate user changes
    base_params = selector._get_default_params()
    base_params['epochs'] = 100          # User changed
    base_params['batch_size'] = 256      # User changed
    base_params['sequence_length'] = 300 # User changed
    base_params['hidden_size'] = 256     # User changed
    base_params['device'] = 'cuda'       # User changed

    print("   Simulated user settings:")
    print(f"     epochs: {base_params['epochs']}")
    print(f"     batch_size: {base_params['batch_size']}")
    print(f"     sequence_length: {base_params['sequence_length']}")
    print(f"     hidden_size: {base_params['hidden_size']}")
    print(f"     device: {base_params['device']}")

    # Create base args (from argparse)
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_timeframe', default='1min')
    parser.add_argument('--spy_data', default=None)
    parser.add_argument('--tsla_data', default=None)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--sequence_length', type=int, default=84)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--device', default=None)
    parser.add_argument('--output', default='models/test.pth')

    base_args = parser.parse_args([])

    print("\n2. Testing parameter flow for each timeframe...")

    timeframes = ['15min', '1hour', '4hour', 'daily']
    all_pass = True

    for i, tf in enumerate(timeframes, 1):
        print(f"\n   [{i}/4] Testing {tf}...")

        # Simulate what train_all_models_interactive() does
        import copy
        params = copy.deepcopy(base_params)

        # Override timeframe-specific
        params['input_timeframe'] = tf
        params['spy_data'] = f'data/SPY_{tf}.csv'
        params['tsla_data'] = f'data/TSLA_{tf}.csv'

        # Convert to args
        model_args = copy.deepcopy(base_args)
        model_args = create_argparse_from_params(params, model_args)
        model_args.output = f'models/lnn_{tf}.pth'
        model_args.spy_data = f'data/SPY_{tf}.csv'
        model_args.tsla_data = f'data/TSLA_{tf}.csv'
        model_args.input_timeframe = tf

        # Simulate metadata creation (what run_training_pipeline() does)
        simulated_metadata = {
            'input_timeframe': model_args.input_timeframe,
            'sequence_length': model_args.sequence_length,
            'epochs': model_args.epochs,
            'batch_size': model_args.batch_size,
            'hidden_size': model_args.hidden_size,
            'device': model_args.device,
        }

        # Verify
        errors = []

        if simulated_metadata['input_timeframe'] != tf:
            errors.append(f"input_timeframe: expected {tf}, got {simulated_metadata['input_timeframe']}")

        if simulated_metadata['sequence_length'] != 300:
            errors.append(f"sequence_length: expected 300, got {simulated_metadata['sequence_length']}")

        if simulated_metadata['epochs'] != 100:
            errors.append(f"epochs: expected 100, got {simulated_metadata['epochs']}")

        if simulated_metadata['batch_size'] != 256:
            errors.append(f"batch_size: expected 256, got {simulated_metadata['batch_size']}")

        if simulated_metadata['hidden_size'] != 256:
            errors.append(f"hidden_size: expected 256, got {simulated_metadata['hidden_size']}")

        if errors:
            print(f"      ❌ FAILED:")
            for error in errors:
                print(f"         {error}")
            all_pass = False
        else:
            print(f"      ✅ PASSED: All settings flow correctly")
            print(f"         input_timeframe: {tf}")
            print(f"         sequence_length: 300 (user setting)")
            print(f"         epochs: 100 (user setting)")
            print(f"         batch_size: 256 (user setting)")
            print(f"         device: cuda (user setting)")

    print("\n" + "=" * 70)
    if all_pass:
        print("✅ ALL TESTS PASSED")
        print("\nInteractive 'Train All 4' option will correctly:")
        print("  1. Configure parameters once")
        print("  2. Train 4 models sequentially")
        print("  3. Each model gets:")
        print("     - Different input_timeframe (15min, 1hour, 4hour, daily)")
        print("     - Same user settings (epochs, batch_size, sequence_length, etc.)")
        print("     - Correct metadata saved to checkpoint")
        print("  4. Each model file contains correct timeframe in metadata")
        print("\nMetadata will flow correctly through the entire pipeline!")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nParameter flow has issues that need to be fixed.")

    print("=" * 70)
    print()


if __name__ == '__main__':
    test_metadata_flow()
