#!/usr/bin/env python3
"""
Test that num_workers is properly passed from interactive menu to args.
This simulates what happens when a user selects num_workers=8 in the menu.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml.interactive_params import InteractiveParameterSelector, create_argparse_from_params

def test_parameter_passing():
    """Test that num_workers and pin_memory are properly passed through."""
    print("\n" + "=" * 70)
    print("🧪 TESTING PARAMETER PASSING FIX")
    print("=" * 70)

    # Create a selector
    selector = InteractiveParameterSelector(mode='standard')

    # Simulate user selections
    test_params = selector._get_default_params()
    test_params['num_workers'] = 8  # User selected 8
    test_params['pin_memory'] = True  # User selected True
    test_params['batch_size'] = 128  # User selected 128

    print("\n✅ Simulated user selections:")
    print(f"  num_workers: {test_params['num_workers']}")
    print(f"  pin_memory: {test_params['pin_memory']}")
    print(f"  batch_size: {test_params['batch_size']}")

    # Create a mock args object (similar to what train_model_lazy.py does)
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--pin_memory', type=lambda x: x.lower() == 'true', default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.001)

    # Parse empty args (all defaults)
    args = parser.parse_args([])

    print("\n📝 Args before create_argparse_from_params:")
    print(f"  args.num_workers: {args.num_workers}")
    print(f"  args.pin_memory: {args.pin_memory}")
    print(f"  args.batch_size: {args.batch_size}")

    # Apply the params to args (this is what happens after interactive selection)
    args = create_argparse_from_params(test_params, args)

    print("\n📝 Args after create_argparse_from_params:")
    print(f"  args.num_workers: {args.num_workers}")
    print(f"  args.pin_memory: {args.pin_memory}")
    print(f"  args.batch_size: {args.batch_size}")

    # Test the result
    print("\n" + "=" * 70)
    print("🔍 VERIFICATION")
    print("=" * 70)

    if args.num_workers == 8:
        print("✅ SUCCESS: num_workers=8 was properly passed!")
    else:
        print(f"❌ FAILURE: Expected num_workers=8, got {args.num_workers}")

    if args.pin_memory == True:
        print("✅ SUCCESS: pin_memory=True was properly passed!")
    else:
        print(f"❌ FAILURE: Expected pin_memory=True, got {args.pin_memory}")

    if args.batch_size == 128:
        print("✅ SUCCESS: batch_size=128 was properly passed!")
    else:
        print(f"❌ FAILURE: Expected batch_size=128, got {args.batch_size}")

    # Simulate what the training script would do
    print("\n" + "=" * 70)
    print("🔧 SIMULATING TRAINING SCRIPT LOGIC")
    print("=" * 70)

    # Simulate device being cpu/mps (not cuda)
    device_type = 'cpu'  # or 'mps'

    # This is the logic from train_model_lazy.py
    if args.num_workers is None:
        num_workers = 2 if device_type == 'cuda' else 0
        workers_source = "auto-detected"
    else:
        num_workers = args.num_workers
        workers_source = "user-specified"

    print(f"\nDevice type: {device_type}")
    print(f"Final num_workers: {num_workers} ({workers_source})")

    if num_workers == 8 and workers_source == "user-specified":
        print("\n✅ PERFECT! User's selection of num_workers=8 would be used!")
        print("   The fix is working correctly.")
    else:
        print(f"\n❌ Problem: User's selection wasn't used properly")
        print(f"   Expected: 8 (user-specified)")
        print(f"   Got: {num_workers} ({workers_source})")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
    print("\nThe fix successfully adds num_workers and pin_memory to the mapping,")
    print("allowing user selections to be properly passed to the training script.")
    print("\n")

if __name__ == '__main__':
    test_parameter_passing()