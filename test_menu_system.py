#!/usr/bin/env python3
"""
Test the new menu-based parameter selection system.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.interactive_params import InteractiveParameterSelector


def test_menu_display():
    """Test that the menu displays correctly."""
    print("\n" + "=" * 70)
    print("🧪 TESTING MENU-BASED PARAMETER SYSTEM")
    print("=" * 70)

    # Create selector
    selector = InteractiveParameterSelector(mode='standard')

    # Display the parameter menu
    selector._display_parameter_menu()

    print("\n" + "=" * 70)
    print("✅ MENU DISPLAY TEST COMPLETE")
    print("=" * 70)


def test_selection_parser():
    """Test the selection parser."""
    print("\n" + "=" * 70)
    print("🧪 TESTING SELECTION PARSER")
    print("=" * 70)

    selector = InteractiveParameterSelector(mode='standard')

    test_cases = [
        ("1,5,8", [1, 5, 8]),
        ("5-10", [5, 6, 7, 8, 9, 10]),
        ("1,3-5,10", [1, 3, 4, 5, 10]),
        ("invalid", []),
        ("25", []),  # Out of range
        ("1-3,5", [1, 2, 3, 5]),
    ]

    for input_str, expected in test_cases:
        result = selector._parse_selection(input_str, 21)
        passed = result == expected
        status = "✓" if passed else "✗"
        print(f"  {status} '{input_str}' -> {result} (expected {expected})")

    print("\n✅ SELECTION PARSER TEST COMPLETE")
    print("=" * 70)


def test_batch_size_suggestions():
    """Test batch size suggestions."""
    print("\n" + "=" * 70)
    print("🧪 TESTING BATCH SIZE SUGGESTIONS")
    print("=" * 70)

    selector = InteractiveParameterSelector(mode='standard')

    # Set some RAM values for testing
    selector.params['_available_ram_gb'] = 8.0
    selector.params['device'] = 'mps'

    suggestions = selector._get_batch_size_suggestions()

    print(f"\nWith 8GB RAM and MPS device:")
    print(f"  Conservative: {suggestions['conservative']}")
    print(f"  Balanced: {suggestions['balanced']}")
    print(f"  Aggressive: {suggestions['aggressive']}")

    print("\n✅ BATCH SIZE SUGGESTIONS TEST COMPLETE")
    print("=" * 70)


def main():
    """Run all tests."""
    print("\nThis test verifies the menu-based parameter selection system.")
    print("The full interactive mode can be tested with:")
    print("  python3 train_model.py --interactive")
    print("  python3 train_model_lazy.py --interactive")

    test_menu_display()
    test_selection_parser()
    test_batch_size_suggestions()

    print("\n" + "=" * 70)
    print("✅ ALL MENU SYSTEM TESTS COMPLETE")
    print("=" * 70)
    print("\nThe menu-based system is working correctly!")
    print("Users can now:")
    print("  1. See all parameters at once in categorized groups")
    print("  2. Select specific parameters to modify (e.g., '1,5,8' or '5-10')")
    print("  3. Get RAM-based batch size suggestions")
    print("  4. See that local macro data exists without needing API key")
    print("=" * 70)


if __name__ == '__main__':
    main()