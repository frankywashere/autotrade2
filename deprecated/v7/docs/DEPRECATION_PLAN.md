# v6 → v7 Migration - Deprecation Plan

## Files to Move to deprecated_code/v6_backup/

### Old Training & Prediction Scripts
- `train_hierarchical.py` (263KB) - Old training script
- `predict.py` (54KB) - Old prediction script
- `config.py` (17KB) - Old configuration
- `test_settings.py` - Old test settings
- `dashboard_v531.py` (29KB) - Old dashboard

### Old Source Code
- `src/` (entire directory with 22 files in ml/)
  - src/ml/features.py (6,649 lines)
  - src/ml/hierarchical_model.py (2,319 lines)
  - src/ml/hierarchical_dataset.py (3,002 lines)
  - src/linear_regression.py
  - And all other old modules

### Old Tools
- `tools/` (5 old visualizer files)
  - visualize_channels.py
  - visualize_live_channels.py
  - channel_loader.py
  - channel_inspector.py
  - README_visualizer.md

### Old Models
- `models/` (trained .pth files and metadata)
  - hierarchical_lnn.pth
  - hierarchical_training_history.json

### Old Utilities
- `utils/` (utility functions)

## Files to KEEP in Root

### v7 System
- `v7/` - Complete new system

### New Entry Points
- `train.py` - New training CLI
- `dashboard.py` - New dashboard
- `dashboard_visual.py` - Visual dashboard
- `run_dashboard.sh` - Launcher

### Data & Infrastructure
- `data/` - All data files (shared)
- `myenv/` - Virtual environment
- `.git/` - Git repo
- `docs/` - Documentation (mix of old and new - audit separately)
- `reports/` - Generated reports
- `logs/` - Logs
- `checkpoints/` - Will be generated
- `.gitignore`, `.env.example`, etc.

### Keep for Reference (Maybe)
- `gce_setup/` - Google Cloud setup (if still relevant)
- `results/` - Old results (for comparison)

## Deprecation Steps

1. Create `deprecated_code/v6_backup/` directory
2. Move old files preserving structure
3. Add README.md in deprecated explaining what was moved and why
4. Update any documentation that references old files
5. Test that v7 still works after move
