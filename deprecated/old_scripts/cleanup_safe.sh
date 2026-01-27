#!/bin/bash
# X8 Codebase Cleanup Script (SAFE operations only)
# Generated: 2026-01-14

set -e  # Exit on error

echo "=== X8 Codebase Cleanup (Safe Operations) ==="
echo ""

# Phase 1: Delete backup files
echo "Phase 1: Deleting backup files..."
if [ -f "v7/data/live_fetcher_backup.py" ]; then
    rm v7/data/live_fetcher_backup.py
    echo "  ✓ Deleted v7/data/live_fetcher_backup.py"
fi

if [ -f "deprecated_code/models/hierarchical_training_history.json.backup" ]; then
    rm deprecated_code/models/hierarchical_training_history.json.backup
    echo "  ✓ Deleted deprecated_code/models/hierarchical_training_history.json.backup"
fi

if [ -f "deprecated_code/Technical_Specification_v2_backup.md" ]; then
    rm deprecated_code/Technical_Specification_v2_backup.md
    echo "  ✓ Deleted deprecated_code/Technical_Specification_v2_backup.md"
fi

# Phase 2: Consolidate duplicate scripts
echo ""
echo "Phase 2: Removing duplicate analysis scripts..."

# Keep analyze_direction_labels_v2.py (latest)
if [ -f "analyze_direction_labels.py" ]; then
    rm analyze_direction_labels.py
    echo "  ✓ Deleted analyze_direction_labels.py (superseded by v2)"
fi

if [ -f "analyze_labels.py" ]; then
    rm analyze_labels.py
    echo "  ✓ Deleted analyze_labels.py (superseded by direction_labels_v2)"
fi

if [ -f "analyze_labels_simple.py" ]; then
    rm analyze_labels_simple.py
    echo "  ✓ Deleted analyze_labels_simple.py (superseded by direction_labels_v2)"
fi

# Keep verify_data_coverage_efficient.py (optimized)
if [ -f "verify_data_coverage.py" ]; then
    rm verify_data_coverage.py
    echo "  ✓ Deleted verify_data_coverage.py (superseded by efficient version)"
fi

echo ""
echo "=== Cleanup Complete ==="
echo ""
echo "Space freed: ~130KB (backup files + duplicates)"
echo ""
echo "NEXT STEPS (Manual):"
echo "1. Archive deprecated_code/ to external storage (235MB)"
echo "2. Archive old training runs (optional, ~4GB)"
echo ""
echo "See docs/CLEANUP_PLAN.md for detailed instructions."
