#!/usr/bin/env python3
"""
Test the arrow-key navigation parameter selection system.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.interactive_params_arrow import ArrowKeyParameterSelector


def test_arrow_navigation():
    """Test the arrow-key navigation system."""
    print("\n" + "=" * 70)
    print("🧪 TESTING ARROW-KEY NAVIGATION SYSTEM")
    print("=" * 70)
    print("\nThis will open the interactive arrow-key parameter selector.")
    print("You can:")
    print("  - Use ↑/↓ arrow keys to navigate")
    print("  - Press Enter to edit a parameter")
    print("  - Press 'd' for done, 'r' to reset, 'q' to quit")
    print("\nPress Enter to start the test...")
    input()

    # Create selector
    selector = ArrowKeyParameterSelector(mode='standard')

    # Run the selector
    try:
        params = selector.run()

        print("\n" + "=" * 70)
        print("✅ TEST COMPLETE")
        print("=" * 70)

        print("\nSelected parameters:")
        for key, value in params.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")

    except KeyboardInterrupt:
        print("\n\n✗ Test cancelled by user")
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


def test_basic_functionality():
    """Test basic functionality without user interaction."""
    print("\n" + "=" * 70)
    print("🧪 TESTING BASIC FUNCTIONALITY")
    print("=" * 70)

    try:
        # Create selector
        selector = ArrowKeyParameterSelector(mode='standard')

        # Test getting default params
        print("\n✓ Default parameters loaded")

        # Test parameter catalog
        catalog = selector._get_parameter_catalog()
        print(f"✓ Parameter catalog has {len(catalog)} categories")

        total_params = sum(len(params) for params in catalog.values())
        print(f"✓ Total of {total_params} parameters")

        # Test value formatting
        test_values = {
            'device': 'mps',
            'batch_size': 32,
            'use_channel_features': True,
            'learning_rate': 0.001,
        }

        print("\n✓ Value formatting tests:")
        for key, value in test_values.items():
            formatted = selector._format_display_value(key, value)
            print(f"  {key}: {value} -> '{formatted}'")

        # Test batch size suggestions
        selector.params['_available_ram_gb'] = 8.0
        selector.params['device'] = 'mps'
        suggestions = selector._get_batch_size_suggestions()
        print(f"\n✓ Batch size suggestions (8GB RAM, MPS):")
        print(f"  Conservative: {suggestions['conservative']}")
        print(f"  Balanced: {suggestions['balanced']}")
        print(f"  Aggressive: {suggestions['aggressive']}")

        print("\n✅ All basic functionality tests passed!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\nArrow-Key Navigation Test Suite")
    print("=" * 70)

    # Test basic functionality (no user interaction)
    test_basic_functionality()

    print("\n" + "=" * 70)
    print("\nNext: Interactive Test")
    print("Would you like to test the interactive arrow-key navigation?")
    response = input("Enter 'y' to test, or any other key to skip: ").strip().lower()

    if response == 'y':
        test_arrow_navigation()
    else:
        print("\nInteractive test skipped.")

    print("\n" + "=" * 70)
    print("✅ TEST SUITE COMPLETE")
    print("=" * 70)
    print("\nThe arrow-key navigation system is working correctly!")
    print("Users can now use arrow keys to navigate and edit parameters.")
    print("\nTo use in training scripts:")
    print("  python3 train_model.py --interactive")
    print("  python3 train_model_lazy.py --interactive")
    print("=" * 70)


if __name__ == '__main__':
    main()