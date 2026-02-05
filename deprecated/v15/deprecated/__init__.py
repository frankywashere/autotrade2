"""
V15 Deprecation and Migration Guide
====================================

Most files from the v15/deprecated/ module have been moved to v15/archived/
for historical reference or deleted if they had no value.

Current locations:
- Visualization utilities: v15/visualization/utils.py
- Historical inspectors: v15/archived/
- Analysis scripts: v15/archived/analysis/

For new visualization needs, use:
    from v15.visualization import plot_candlesticks, plot_channel_bounds

For channel inspection:
    python -m v15.inspector --samples samples.bin
"""


def print_deprecation_guide():
    """Print migration guide from v7 to v15."""
    print("""
================================================================================
                         V7 -> V15 Migration Guide
================================================================================

  SCANNING (CHANGED)
  ------------------
  The Python scanner has been removed. Use the C++ scanner for 10x faster
  feature extraction:

    cd v15_cpp/build && ./v15_scanner --data-dir ../../data --output samples.bin

  The C++ scanner produces binary .bin files with 14,840 features per sample.

  TRAINING
  --------
  Training now accepts both .pkl (legacy) and .bin (new) formats:

    python -m v15.pipeline train --samples samples.bin --output checkpoints/

  MODELS
  ------
  OLD: v7.models.HierarchicalCfC (776 features)
  NEW: v15.models.V15Model (14,840 features via C++ scanner)

  FEATURES
  --------
  OLD: v7.features.extract_full_features() -> 776 features
  NEW: C++ scanner extracts 14,840 features per sample

  KEY IMPROVEMENTS
  ----------------
  * C++ scanner: 10x faster than Python, 14,840 features
  * Explicit feature weights (learnable attention)
  * Partial bar support (no stale TF data)
  * Automatic correlation analysis
  * Binary format (.bin) for fast sample loading

  COMMANDS
  --------
  Scan (C++ - 10x faster):
    cd v15_cpp/build && ./v15_scanner --data-dir ../../data --output samples.bin

  Train:
    python -m v15.pipeline train --samples samples.bin --output model_dir/

  Inspect:
    python -m v15.inspector --samples samples.bin

  Analyze:
    python -m v15.pipeline analyze --samples samples.bin

  Dashboard:
    streamlit run v15/dashboard.py

  Info:
    python -m v15.pipeline info

================================================================================
""")
