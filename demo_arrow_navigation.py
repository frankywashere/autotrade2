#!/usr/bin/env python3
"""
Demo script showing the arrow-key navigation in action.
This demonstrates what users will see when using --interactive mode.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def print_demo():
    """Print a visual demo of what the interface looks like."""
    print("\n" + "=" * 80)
    print("🎯 ARROW-KEY NAVIGATION DEMO")
    print("=" * 80)
    print("\nThis shows what the new arrow-key navigation interface looks like:")
    print("\n" + "-" * 80)

    # Simulated menu display
    menu_display = """
================================================================================
📊 PARAMETER CONFIGURATION MENU
================================================================================

📁 DATA FILES
────────────────────────────────────────────────────────────
  SPY data file               : data/SPY_1min.csv
  TSLA data file              : data/TSLA_1min.csv
❯ TSLA events file            : data/tsla_events_REAL.csv
  Macro API key               : Not set (local data available)

📅 TRAINING PERIOD
────────────────────────────────────────────────────────────
  Start year                  : 2015
  End year                    : 2023

🧠 MODEL ARCHITECTURE
────────────────────────────────────────────────────────────
  Model type                  : LNN
  Hidden size                 : 128
  Number of layers            : 2
  Sequence length             : 84

⚙️ TRAINING PARAMETERS
────────────────────────────────────────────────────────────
  Training epochs             : 50
  Pretrain epochs             : 10
* Batch size                  : 64
  Learning rate               : 0.001000
  Validation split            : 0.10

📊 FEATURE FLAGS
────────────────────────────────────────────────────────────
  Channel features            : Yes
  RSI features                : Yes
  Correlation features        : Yes
  Event features              : Yes
  Prediction horizon (hrs)    : 24

🖥️ COMPUTE DEVICE
────────────────────────────────────────────────────────────
  Device                      : mps (auto-detected)

────────────────────────────────────────────────────────────
✓ Done - Start Training
↺ Reset All to Defaults
✗ Quit

[↑↓] Navigate  [Enter] Edit  [d] Done  [r] Reset  [q] Quit
"""

    print(menu_display)
    print("-" * 80)

    print("\nKEY FEATURES:")
    print("  • Navigate with ↑/↓ arrow keys")
    print("  • Selected parameter shown with ❯ pointer")
    print("  • Modified parameters marked with *")
    print("  • Press Enter to edit the selected parameter")
    print("  • Categories are visual separators (not selectable)")
    print("  • Batch size shows RAM-based suggestions when editing")
    print("  • Macro API shows 'local data available' note")

    print("\n" + "-" * 80)
    print("\nEXAMPLE: Editing Batch Size")
    print("-" * 80)

    batch_edit_display = """
============================================================
Editing: Batch size
Current value: 16

💡 RAM-based suggestions (6.4 GB available):
Select batch size:
❯ Conservative: 128 (safest, slower)
  Balanced: 256 (recommended)
  Aggressive: 512 (fastest, may OOM)
  Custom value...
"""

    print(batch_edit_display)

    print("-" * 80)
    print("\nEXAMPLE: Editing Boolean Parameter")
    print("-" * 80)

    boolean_edit_display = """
============================================================
Editing: Channel features
Current value: Yes

Enable Channel features? (Y/n): _
"""

    print(boolean_edit_display)

    print("-" * 80)
    print("\nEXAMPLE: Editing Device")
    print("-" * 80)

    device_edit_display = """
============================================================
Editing: Device
Current value: Auto-detect

Select compute device:
  cpu
❯ mps (current)
  Auto-detect best device
"""

    print(device_edit_display)

    print("\n" + "=" * 80)
    print("COMPARISON: Old vs New")
    print("=" * 80)

    print("\nOLD (Number-based):")
    print("  1. See all parameters with numbers")
    print("  2. Type '8,13' to select parameters")
    print("  3. Answer questions for each selected parameter")
    print("  4. Confirm and start")

    print("\nNEW (Arrow-key):")
    print("  1. See all parameters visually")
    print("  2. Arrow to the parameter you want")
    print("  3. Press Enter to edit it directly")
    print("  4. Continue editing or press 'd' when done")

    print("\n" + "=" * 80)
    print("HOW TO USE")
    print("=" * 80)

    print("\nIn your training scripts:")
    print("\n  python3 train_model.py --interactive")
    print("  python3 train_model_lazy.py --interactive")

    print("\nYou'll be offered:")
    print("  1. Quick start with defaults")
    print("  2. Arrow-key navigation (recommended)")
    print("  3. Number-based menu (legacy)")

    print("\n" + "=" * 80)
    print("✅ Arrow-key navigation is now available!")
    print("=" * 80)


if __name__ == "__main__":
    print_demo()