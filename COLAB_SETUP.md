# Google Colab Setup Guide

## Quick Start (3 steps)

### 1. Create the zip file locally

```bash
cd /Users/frank/Desktop/CodingProjects
zip -r x6.zip x6/ -x "x6/myenv/*" -x "x6/__pycache__/*" -x "x6/.git/*" -x "x6/checkpoints/*" -x "x6/*.pyc" -x "x6/data/feature_cache/*"
```

**What this includes:**
- ✅ All v7/ code
- ✅ data/ folder (TSLA_1min.csv, SPY_1min.csv, VIX_History.csv)
- ✅ train.py and all scripts
- ✅ Configuration files

**What this excludes:**
- ❌ myenv/ (virtual environment - will reinstall in Colab)
- ❌ .git/ (version control - not needed)
- ❌ __pycache__/ (Python cache)
- ❌ checkpoints/ (old model files)
- ❌ data/feature_cache/ (old cached samples - will regenerate)

**Expected zip size:** ~500-800 MB (mostly the CSV data files)

---

### 2. Upload to Colab

1. Go to https://colab.research.google.com
2. Click `File` → `Upload notebook`
3. Upload `colab_label_generation.ipynb`
4. Run the cells in order

---

### 3. Download results

After generation completes:
- Option A: Download from Google Drive: `/MyDrive/x6_results/samples_XXX_TIMESTAMP/`
- Option B: Direct download from Step 10 in the notebook

Then place `channel_samples.pkl` in your local:
```
/Users/frank/Desktop/CodingProjects/x6/data/feature_cache/
```

---

## Configuration Options

### Quick Test (30-60 minutes)
```python
CONFIG = {
    'step': 2000,        # ~40 samples
    'max_workers': None, # Auto
    'parallel': True,
}
```

### Standard Run (2-3 hours)
```python
CONFIG = {
    'step': 500,         # ~150 samples
    'max_workers': None, # Auto
    'parallel': True,
}
```

### Full Dataset (8-12 hours - Colab Pro recommended)
```python
CONFIG = {
    'step': 100,         # ~800 samples
    'max_workers': 8,    # Use all cores
    'parallel': True,
}
```

---

## Colab Tips

### Keep Runtime Active
- Don't close the browser tab
- Keep Colab tab visible (minimized is OK)
- Use Step 11 to monitor resources
- Colab may disconnect after 90 minutes of inactivity

### If Disconnected
- Results are auto-saved to Google Drive
- Re-run the notebook
- Set `force_rebuild: False` to resume from cache

### RAM Issues
If you get "Out of Memory":
1. Increase `step` (1000 or 2000)
2. Reduce `max_workers` to 2
3. Use Colab Pro (25-50 GB RAM vs 12 GB)

### Slow Performance
- Verify CPU usage >50% in Step 11
- Check `parallel: True` is enabled
- Colab Free has only 2 cores - consider Colab Pro

---

## What Gets Generated

**Main file:** `channel_samples.pkl`
- Contains all labeled samples
- Train/val/test split included
- 761 features per sample
- All 11 timeframes
- All 8 window sizes scored

**Metadata:** `channel_samples.json`
- Generation settings
- Cache version
- Sample counts
- Date ranges

**Config:** `config.json`
- Your configuration settings
- Timing information
- Sample statistics

---

## Using Results Locally

After downloading `channel_samples.pkl`:

```bash
# Place the file
mv ~/Downloads/channel_samples.pkl /Users/frank/Desktop/CodingProjects/x6/data/feature_cache/

# Run training
cd /Users/frank/Desktop/CodingProjects/x6
./myenv/bin/python train.py

# Or use label inspector
./myenv/bin/python label_inspector.py --list
./myenv/bin/python label_inspector.py --sample 0
```

The system will automatically detect and load the cached labels instead of regenerating them.
