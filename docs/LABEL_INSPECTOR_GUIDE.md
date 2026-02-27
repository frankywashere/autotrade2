# Label Inspector Usage Guide
**Generated:** 2026-01-14
**Purpose:** Quick reference for using the label inspector tools to manually review sample labels

---

## OVERVIEW

The x8 project includes two label inspector tools for validating and visualizing the generated channel labels:

1. **Root Level Inspector** (`label_inspector.py`) - Interactive visualization tool
2. **Module Inspector** (`v7/tools/label_inspector.py`) - Advanced validation with suspicious detection

---

## TOOL 1: ROOT LEVEL INSPECTOR (Interactive Visualization)

**File:** `/Users/frank/Desktop/CodingProjects/x8/label_inspector.py`

### Features
- Multi-timeframe visualization (2×2 grid: 5min, 15min, 1h, daily)
- OHLC price data with channel bounds overlay
- Channel bounds projected forward from detection window
- Break point markers with vertical lines at duration_bars forward
- Direction arrows (UP=green ↑, DOWN=red ↓)
- Label annotations (duration, direction, trigger_tf, new_channel direction)
- Validity flags display for each timeframe
- Window cycling to compare different window sizes
- Suspicious sample detection and highlighting

### Quick Start Commands

#### Basic Usage
```bash
# Interactive mode - browse samples with keyboard/buttons
python label_inspector.py

# This will:
# - Load cached samples from data/feature_cache/channel_samples.pkl
# - Display 2×2 grid (5min, 15min, 1h, daily)
# - Show channel bounds and break points
# - Allow navigation with arrow keys
```

#### View Specific Sample
```bash
# Jump to specific sample index
python label_inspector.py --sample 42

# Useful for investigating specific cases
```

#### List Available Samples
```bash
# Show all samples and their timestamps
python label_inspector.py --list

# Output shows:
# - Sample index
# - Timestamp
# - Best window size
# - Number of valid channels
```

#### Save Visualization
```bash
# View and save current sample
python label_inspector.py --sample 100 --save sample_100_visualization.png

# Creates PNG file of the current view
```

#### Custom Cache Path
```bash
# Use different cache file
python label_inspector.py --cache /path/to/custom/channel_samples.pkl
```

#### Specify Window Size
```bash
# View specific window size instead of best
python label_inspector.py --window 50

# Shows channels detected with 50-bar window
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| **LEFT ARROW** | Previous sample |
| **RIGHT ARROW** | Next sample |
| **r** | Jump to random sample |
| **f** | Jump to next flagged (suspicious) sample |
| **F** | Jump to previous flagged sample |
| **w** | Cycle through window sizes (best → 10 → 20 → 30 → 40 → 50 → 60 → 70 → 80 → best) |
| **q** or **ESC** | Quit |

### What You'll See

**Each of the 4 panels shows:**
1. **Price Chart:** OHLC bars for the timeframe
2. **Channel Bounds:** Blue lines (upper/lower) projected forward
3. **Break Point:** Vertical red/green line at predicted break
4. **Direction Arrow:** ↑ (green) for UP, ↓ (red) for DOWN
5. **Annotations:**
   - Duration: X bars (how long until break)
   - Direction: UP/DOWN (break direction)
   - Trigger TF: Which longer timeframe boundary caused break
   - New Channel: BEAR/SIDEWAYS/BULL (direction after break)
   - Valid flags: ✓/✗ for each label type

**Title shows:**
- Sample index and timestamp
- Best window size and score
- Number of valid channels across all 11 timeframes

### Typical Workflow

```bash
# 1. Start interactive mode
python label_inspector.py

# 2. Browse samples with arrow keys
# Press RIGHT to go forward, LEFT to go back

# 3. Check for suspicious samples
# Press 'f' to jump to next flagged sample

# 4. Compare window sizes
# Press 'w' to cycle through different window sizes
# See how channel detection differs with window=10 vs window=80

# 5. Save interesting samples
# When you find a sample worth documenting:
# Note the sample number from title
# Quit (q)
# Re-run with: python label_inspector.py --sample <NUM> --save output.png
```

---

## TOOL 2: MODULE INSPECTOR (Advanced Validation)

**File:** `/Users/frank/Desktop/CodingProjects/x8/v7/tools/label_inspector.py`

### Features
- Automatic suspicious pattern detection across all samples
- Color-coded panels (🔴 red=errors, 🟡 yellow=warnings, 🟢 green=OK)
- Summary panel with channel info and validation flags
- Detailed per-timeframe panels showing all label attributes
- Flag counting and categorization
- Comprehensive validation of label generation logic

### Suspicious Patterns Detected

The module inspector automatically flags samples with:

1. **Very Short Duration:** Channel breaks almost immediately (< 5 bars)
2. **NO_TRIGGER with Break:** permanent_break=True but no trigger timeframe found
3. **All Flags False:** Timeframe has expected data but all validity flags are False
4. **Inconsistent Labels:** Different timeframes predict opposite directions (e.g., 5min=UP, 15min=DOWN)
5. **Very Long Duration:** Never broke channel (might indicate data/detection issue)
6. **Missing Expected TFs:** Some timeframes have data but others don't (alignment issue)

### Quick Start Commands

#### Interactive Mode
```bash
# Launch with suspicious detection
python -m v7.tools.label_inspector

# This will:
# - Load samples and scan for suspicious patterns
# - Display detailed panels for current sample
# - Color-code issues (red/yellow/green)
# - Show summary at top
```

#### Jump to Specific Sample
```bash
# Go directly to sample 100
python -m v7.tools.label_inspector --sample 100
```

#### Show Only Suspicious Samples
```bash
# Filter to only problematic samples
python -m v7.tools.label_inspector --suspicious-only

# Useful for focused investigation
# Use 'f' to jump between flagged samples
```

#### Summary Report
```bash
# Print summary and exit (no interactive mode)
python -m v7.tools.label_inspector --summary-only

# Output shows:
# - Total samples
# - Number of suspicious samples
# - Breakdown by flag type
# - Most common issues
```

#### Custom Cache Path
```bash
# Use different cache file
python -m v7.tools.label_inspector --cache-path /path/to/channel_samples.pkl
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| **LEFT ARROW** or **p** | Previous sample |
| **RIGHT ARROW** or **n** | Next sample |
| **UP ARROW** | Jump back 10 samples |
| **DOWN ARROW** | Jump forward 10 samples |
| **f** | Jump to next suspicious sample |
| **F** | Jump to previous suspicious sample |
| **s** | Toggle showing only suspicious samples |
| **i** | Print detailed info for current sample (to terminal) |
| **q** or **ESC** | Quit |

### What You'll See

**Summary Panel (top):**
- Sample index, timestamp, best window
- Total suspicious flags count
- Channel validity summary
- Overall status (🟢 OK / 🟡 WARNING / 🔴 ERROR)

**Per-Timeframe Panels (11 total):**
For each timeframe (5min, 15min, 30min, 1h, 2h, 3h, 4h, daily, weekly, monthly, 3month):
- Channel valid status
- Duration (bars) with validity flag
- Direction (UP/DOWN) with probability and validity flag
- Next channel (BEAR/SIDE/BULL) with validity flag
- Trigger TF with validity flag
- Suspicious flags for this TF (if any)
- Color coding: 🟢 green=all good, 🟡 yellow=warnings, 🔴 red=errors

**Bottom:**
- Navigation instructions
- Current filter status (all samples vs suspicious only)

### Typical Workflow

```bash
# 1. Generate summary report
python -m v7.tools.label_inspector --summary-only > label_validation_report.txt

# Review report to see overall data quality
# Example output:
#   Total samples: 15234
#   Suspicious samples: 234 (1.5%)
#   Flags breakdown:
#     - Very short duration: 89
#     - Inconsistent labels: 67
#     - NO_TRIGGER with break: 45
#     - Very long duration: 33

# 2. Interactive investigation of suspicious samples
python -m v7.tools.label_inspector --suspicious-only

# Use 'f' to jump between flagged samples
# Press 'i' to print detailed info
# Look for patterns in issues

# 3. Investigate specific problematic sample
python -m v7.tools.label_inspector --sample 1234

# Review all 11 timeframes in detail
# Check validity flags and suspicious patterns
```

---

## WHEN TO USE EACH TOOL

### Use Root Level Inspector When:
- You want visual confirmation of channel detection
- You need to see price charts with channel overlays
- You want to compare different window sizes visually
- You're documenting/presenting channel detection quality
- You need to save visualizations as images

### Use Module Inspector When:
- You want to validate label generation logic
- You need to find problematic samples automatically
- You're debugging label inconsistencies
- You want detailed per-timeframe label attributes
- You need a quality report (--summary-only)

### Use Both When:
- Investigating a specific issue:
  1. Module inspector finds suspicious sample (#1234)
  2. Root inspector visualizes it: `python label_inspector.py --sample 1234`
  3. Compare different windows: press 'w' in root inspector
  4. Check detailed attributes: back to module inspector

---

## PROGRAMMATIC ACCESS

You can also use the module inspector functions in your own scripts:

```python
from v7.tools.label_inspector import (
    detect_suspicious_sample,
    detect_suspicious_samples,
    SuspiciousFlag,
    SuspiciousResult
)

# Load samples
import pickle
with open('data/feature_cache/channel_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

# Detect issues in a single sample
result = detect_suspicious_sample(samples[0], 0)
if result.flags:
    print(f"Found {len(result.flags)} issues:")
    for flag in result.flags:
        print(f"  - {flag.flag_type.value}: {flag.description}")

# Scan all samples
suspicious_results = detect_suspicious_samples(samples, progress=True)
print(f"Found {len(suspicious_results)} suspicious samples")

# Filter by flag type
short_duration_samples = [
    r for r in suspicious_results
    if any(f.flag_type == SuspiciousFlag.VERY_SHORT_DURATION for f in r.flags)
]
```

---

## RECOMMENDED INSPECTION WORKFLOW

### After Training Data Generation

1. **Quick Validation:**
   ```bash
   python -m v7.tools.label_inspector --summary-only
   ```
   Review the report. If < 5% suspicious, data quality is good.

2. **Investigate Issues:**
   ```bash
   python -m v7.tools.label_inspector --suspicious-only
   ```
   Use 'f' to jump through flagged samples. Press 'i' for details.

3. **Visual Confirmation:**
   ```bash
   python label_inspector.py
   ```
   Press 'f' to see suspicious samples visually.
   Press 'w' to compare window sizes.

4. **Document Examples:**
   ```bash
   # Save good example
   python label_inspector.py --sample 100 --save examples/good_label_example.png

   # Save problematic example
   python label_inspector.py --sample 1234 --save examples/problematic_label.png
   ```

### During Development/Debugging

1. **Test Changes:**
   ```bash
   # After modifying label generation code
   # Re-run training data generation
   # Then immediately validate:
   python -m v7.tools.label_inspector --summary-only

   # Compare suspicious rate before/after changes
   ```

2. **Investigate Specific Timeframes:**
   ```bash
   python -m v7.tools.label_inspector --sample <NUM>
   # Check all 11 timeframes for consistency
   ```

3. **Window Size Analysis:**
   ```bash
   python label_inspector.py --sample <NUM>
   # Press 'w' repeatedly to cycle through windows
   # See which window sizes produce best channels
   ```

---

## TROUBLESHOOTING

### Issue: "FileNotFoundError: channel_samples.pkl"
**Solution:**
```bash
# Generate cache first
python -m v7.training.dataset
# Or specify custom path
python label_inspector.py --cache /path/to/cache.pkl
```

### Issue: "Samples list is empty"
**Solution:** Cache file exists but contains no samples. Regenerate:
```bash
python -m v7.training.dataset --step 25
```

### Issue: Visualizations show no data
**Solution:** Sample might have no valid channels. Try different sample:
```bash
python label_inspector.py --list  # Find samples with valid channels
python label_inspector.py --sample <VALID_INDEX>
```

### Issue: Module inspector shows all red/errors
**Solution:** Might be data quality issue. Check:
```bash
python verify_data_coverage_efficient.py
# Ensure TSLA/SPY/VIX data is aligned and complete
```

---

## INTERPRETING RESULTS

### Good Label Quality Indicators
- ✅ < 5% suspicious samples
- ✅ Consistent directions across similar timeframes (5min ≈ 15min)
- ✅ Trigger TF always found when break occurs
- ✅ Durations are reasonable (not too short, not infinite)
- ✅ New channels detected after breaks
- ✅ Visual inspection shows clean channel detection

### Warning Signs
- ⚠️ 5-10% suspicious samples (investigate but acceptable)
- ⚠️ Some inconsistent labels (might be legitimate market complexity)
- ⚠️ Occasional NO_TRIGGER (edge cases)
- ⚠️ Some very long durations (might be sideways markets)

### Red Flags
- 🚨 > 10% suspicious samples (data/logic issue)
- 🚨 Systematic inconsistencies (e.g., all short TFs disagree with long TFs)
- 🚨 Many missing validity flags (label generation not working)
- 🚨 Visual inspection shows channels not fitting price action
- 🚨 Channels breaking immediately (window size too small?)

---

## QUICK REFERENCE COMMANDS

```bash
# === ROOT LEVEL INSPECTOR ===

# Basic interactive mode
python label_inspector.py

# View specific sample
python label_inspector.py --sample 42

# List all samples
python label_inspector.py --list

# Save visualization
python label_inspector.py --sample 100 --save output.png

# Custom cache
python label_inspector.py --cache /path/to/cache.pkl

# Specific window size
python label_inspector.py --window 50

# === MODULE INSPECTOR ===

# Interactive validation
python -m v7.tools.label_inspector

# Summary report
python -m v7.tools.label_inspector --summary-only

# Only suspicious samples
python -m v7.tools.label_inspector --suspicious-only

# Jump to sample
python -m v7.tools.label_inspector --sample 100

# Custom cache
python -m v7.tools.label_inspector --cache-path /path/to/cache.pkl
```

---

## ADDITIONAL RESOURCES

- **Label Generation Logic:** `v7/training/labels.py`
- **Channel Detection:** `v7/core/channel.py`
- **Dataset Creation:** `v7/training/dataset.py`
- **Technical Docs:** `docs/COMPREHENSIVE_TECH_SHEET.md`
- **Architecture:** `v7/docs/ARCHITECTURE.md`

---

**END OF LABEL INSPECTOR GUIDE**
