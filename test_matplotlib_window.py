#!/usr/bin/env python3
"""
Minimal test script to verify matplotlib window display.
Tests basic matplotlib functionality with the small_sample.pkl data.
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("=" * 60)
    print("Matplotlib Window Test")
    print("=" * 60)

    # Check matplotlib backend
    print(f"\nMatplotlib backend: {plt.get_backend()}")

    # Load the data file
    print("\n1. Loading small_sample.pkl...")
    with open('/Users/frank/Desktop/CodingProjects/x14/small_sample.pkl', 'rb') as f:
        data = pickle.load(f)

    print(f"   Data type: {type(data)}")
    if isinstance(data, (list, tuple)):
        print(f"   Number of samples: {len(data)}")
    elif isinstance(data, dict):
        print(f"   Dictionary keys: {list(data.keys())}")

    # Get sample 100
    print("\n2. Getting sample 100...")
    if isinstance(data, (list, tuple)):
        if len(data) > 100:
            sample = data[100]
            print(f"   Sample type: {type(sample)}")
            if hasattr(sample, 'shape'):
                print(f"   Sample shape: {sample.shape}")
        else:
            print(f"   ERROR: Not enough samples (only {len(data)} available)")
            sample = data[0]  # Use first sample instead
            print(f"   Using sample 0 instead")
    else:
        sample = data
        print(f"   Sample type: {type(sample)}")

    # Create a simple matplotlib figure
    print("\n3. Creating matplotlib figure...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Plot something simple
    print("4. Plotting simple content...")

    # Draw a simple sine wave
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y, 'b-', linewidth=2, label='Sine Wave')

    # Add some text
    ax.text(5, 0, 'TEST: If you can see this, matplotlib is working!',
            fontsize=14, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title('Matplotlib Window Test - Sample Data Loaded Successfully',
                 fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    print("\n5. Calling plt.show()...")
    print("\n" + "=" * 60)
    print("LOOK FOR A WINDOW TO APPEAR NOW!")
    print("If you see a window with a sine wave, matplotlib is working.")
    print("If no window appears, there's a backend/display issue.")
    print("=" * 60 + "\n")

    plt.show()

    print("\n6. plt.show() returned (window was closed)")
    print("\nTest complete!")

if __name__ == '__main__':
    main()
