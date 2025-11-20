# Channel Visualizer Tool

**Version:** 1.0 (v3.17+)
**Purpose:** Visualize channels from cached mmap shards

---

## Features

✅ Interactive shard location selection (local or external drive)
✅ Plot channels with regression lines and touch points
✅ Compare ping_pongs vs complete_cycles metrics
✅ Browse high-quality and low-quality channels
✅ See what the model is learning from

---

## Quick Start

### 1. Run the Visualizer

```bash
cd /Users/frank/Desktop/CodingProjects/autotrade2
python tools/visualize_channels.py
```

### 2. Select Shard Location

**Interactive menu will show:**
```
📂 Select shard storage location:
  📁 Default - data/feature_cache
  💾 Last used - /Volumes/ExternalDrive/feature_cache
  🔧 Custom path (enter manually)
```

**Select your storage location** (local or external drive)

### 3. Choose Visualization Mode

```
🎯 Specific channel - Enter exact timestamp/symbol/timeframe/window
⭐ High-quality channels - Browse channels with quality > 0.8
⚠️  Low-quality channels - See "bad" channels (quality < 0.3)
📊 Compare metrics - See ping_pongs vs complete_cycles differences
```

---

## What You'll See

### Channel Plot (Top Panel):
- **Black line:** Actual price
- **Red dashed:** Upper channel boundary
- **Green dashed:** Lower channel boundary
- **Blue line:** Center (regression line)
- **Red dots:** Price touched upper boundary
- **Green dots:** Price touched lower boundary

### Metrics Table (Bottom Panel):
```
Legacy Transitions (ping_pongs):
  2.0% threshold: 4 transitions

Complete Cycles (v3.17):
  2.0% threshold: 2 full round-trips ⭐

Ratio: 2.0 transitions per complete cycle

Quality Metrics:
  R² (fit quality): 0.850
  Quality score: 0.715
  Is valid: 1.0 ✅ YES
  Position: 0.37 (37% through channel)
```

---

## Use Cases

### 1. Verify Complete Cycles Metric Makes Sense

**Browse high-quality channels:**
- See channels with complete_cycles >= 3
- Verify they look like "real" oscillating channels
- Confirm metric captures what you expect

### 2. Understand "Bad" vs "Good" Channels

**Compare:**
- High quality (quality > 0.8): Clean oscillations, clear boundaries
- Low quality (quality < 0.3): Choppy, no pattern, few cycles

**This shows why "bad" channels are useful:**
- They signal "this timeframe doesn't have a pattern right now"
- Model learns to switch to different timeframe

### 3. Compare Metrics

**See channels where:**
- ping_pongs = 6 but complete_cycles = 2 (many half-cycles)
- ping_pongs = 4 but complete_cycles = 2 (clean full cycles)

**Understand:** Complete cycles is stricter, better quality signal

### 4. Validate Model Inputs

**Before training:**
- Browse random channels
- Verify they look reasonable
- Check no weird artifacts
- Confirm model is learning from good data

---

## Configuration

### Shard Path Memory

The visualizer saves your last-used shard path to `.visualizer_config.json`:

```json
{
  "shard_path": "/Volumes/ExternalDrive/feature_cache"
}
```

**Benefit:** Next time you run, your external drive path is remembered!

### Default Locations Checked

1. `data/feature_cache` (local)
2. Last used path (from config)
3. Custom path (you enter)

---

## Requirements

**Python Packages:**
- matplotlib (for plotting)
- pandas, numpy (data handling)
- InquirerPy (interactive menus) - optional, falls back to simple input

**Install if needed:**
```bash
pip install matplotlib InquirerPy
```

**Data Requirements:**
- Feature extraction must have run at least once
- Mmap shards must exist (features_mmap_meta_*.json + *.npy files)

---

## Tips

**Best Practices:**

1. **Start with random high-quality channels**
   - See what "good" looks like
   - Verify complete_cycles metric is sensible

2. **Then browse low-quality channels**
   - Understand what model considers "unreliable"
   - See why these shouldn't be trusted

3. **Compare specific timestamps from your trading**
   - Enter dates where you made trades
   - See what the model would have seen
   - Understand model's "view" of the market

**Keyboard Shortcuts:**
- Close plot window to go back to menu
- `Ctrl+C` to exit at any time

---

## Troubleshooting

**Error: "No shard metadata found"**
- Run feature extraction first: `python train_hierarchical.py`
- Verify shard path is correct
- Check external drive is mounted

**Error: "Insufficient data"**
- Timestamp too early (before lookback window)
- Try later timestamp or smaller window

**Error: "Feature not found in shards"**
- Shards may be from old version (v3.16 vs v3.17)
- Re-extract features with new code

---

## Future Enhancements

🔄 **Coming Soon:**
- Multi-window comparison view (all 21 windows side-by-side)
- Complete cycle highlighting (shade L→U→L patterns)
- Export interesting patterns to images
- Statistical analysis (correlation between metrics and future returns)
- Integration with backtester (show channels from winning trades)

---

## Examples

### Example 1: Quick Random Inspection

```bash
python tools/visualize_channels.py

# Menu selections:
> Default - data/feature_cache
> Random high-quality channels
# Shows 5 random channels with quality > 0.8
```

### Example 2: Investigate Specific Trade

```bash
python tools/visualize_channels.py

# Menu selections:
> External drive path: /Volumes/SSD/autotrade_shards
> Specific channel
> Symbol: TSLA
> Timeframe: 1h
> Window: 168
> Timestamp: 2023-06-15 14:30
# Shows channel at your trade entry time
```

### Example 3: Understand Timeframe Switching

```bash
# Look at timestamp where 1h was weak but 4h was strong
python tools/visualize_channels.py

> Specific channel
> TSLA, 1h, window=168, timestamp=2023-08-20 10:00
# Note: quality=0.25, complete_cycles=1, is_valid=0 ❌

Then visualize same timestamp, 4h:
> TSLA, 4h, window=168, timestamp=2023-08-20 10:00
# Note: quality=0.85, complete_cycles=4, is_valid=1 ✅

This shows why model should trust 4h over 1h at that moment!
```

---

**Ready to explore your channels!** 🎨
