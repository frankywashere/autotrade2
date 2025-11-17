#!/usr/bin/env python3
"""
Quick test script to demonstrate the new progress bars in training
Run this to see the improved terminal output before doing full training
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Check if required libraries are installed
try:
    from tqdm import tqdm
    import psutil
except ImportError as e:
    print("\n⚠️  Missing required libraries!")
    print("Please install them first:")
    print("  pip install tqdm psutil")
    print("\nOr install all requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

def demo_progress_bars():
    """Demo the different progress bars used in training"""

    print("\n" + "=" * 70)
    print("🎬 PROGRESS BAR DEMO")
    print("=" * 70)
    print("This shows what you'll see during actual training\n")

    # 1. Data loading demo
    print("📊 Data Loading Progress:")
    print("-" * 40)

    with tqdm(total=100, desc="Loading SPY data", unit="%",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for i in range(10):
            time.sleep(0.1)
            pbar.update(10)

    print("✓ SPY data loaded")

    # 2. Feature extraction demo
    print("\n📈 Feature Extraction Progress:")
    print("-" * 40)

    features = ["Price features", "Channel features", "RSI indicators",
                "Correlations", "Volume metrics"]

    with tqdm(total=len(features), desc="Extracting features", unit="type",
              bar_format="{l_bar}{bar:30}{r_bar}") as pbar:
        for feature in features:
            pbar.set_postfix_str(feature)
            time.sleep(0.3)
            pbar.update(1)

    print("✓ All features extracted")

    # 3. Training epochs demo
    print("\n🎯 Training Progress:")
    print("-" * 40)

    # Main epoch progress
    epoch_pbar = tqdm(range(5), desc="Training epochs", unit="epoch",
                      bar_format="{l_bar}{bar:25}{r_bar}")

    for epoch in epoch_pbar:
        # Simulated metrics
        train_loss = 2.5 - (epoch * 0.3)
        val_loss = 2.3 - (epoch * 0.25)

        # Update with metrics
        epoch_pbar.set_postfix({
            'train': f'{train_loss:.3f}',
            'val': f'{val_loss:.3f}',
            'lr': '1e-3'
        })

        # Batch progress (nested)
        batch_pbar = tqdm(range(20), desc=f"  Epoch {epoch+1}/5",
                         unit="batch", leave=False,
                         bar_format="{l_bar}{bar:20}{r_bar}")

        for batch in batch_pbar:
            batch_loss = train_loss + (0.1 * (batch % 3 - 1))
            batch_pbar.set_postfix({'loss': f'{batch_loss:.3f}'})
            time.sleep(0.05)

        print(f"  Epoch {epoch+1}: Train={train_loss:.3f} | Val={val_loss:.3f}")

    # 4. System info demo
    print("\n💻 System Monitoring:")
    print("-" * 40)

    memory = psutil.virtual_memory()
    process = psutil.Process()

    print(f"  Memory available: {memory.available / 1024**3:.1f} GB")
    print(f"  Memory used by process: {process.memory_info().rss / 1024**2:.1f} MB")
    print(f"  CPU count: {psutil.cpu_count()} cores")

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)
    print("\nThis is what you'll see during actual training, but with real data!")
    print("The progress bars will help you track:")
    print("  • Data loading stages")
    print("  • Feature extraction progress")
    print("  • Training epochs and batches")
    print("  • Live loss metrics")
    print("  • Memory usage")
    print("\nRun the actual training with:")
    print("  python train_model.py --tsla_events data/tsla_events_REAL.csv --epochs 10")
    print("=" * 70)

if __name__ == "__main__":
    demo_progress_bars()