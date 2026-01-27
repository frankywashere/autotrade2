# X8 Codebase Cleanup Plan
**Generated:** 2026-01-14
**Purpose:** Document cleanup recommendations for deprecated and duplicate files

---

## CLEANUP SUMMARY

**Total Potential Space Savings:** ~5.2GB
- Deprecated code: 235MB
- Old training runs: ~4GB (archive older experiments)
- Backup files: ~100KB

---

## PHASE 1: DELETE BACKUP FILES (Safe, ~100KB)

These are redundant backup files that can be safely deleted:

```bash
# Delete backup files
rm v7/data/live_fetcher_backup.py
rm deprecated_code/models/hierarchical_training_history.json.backup
rm deprecated_code/Technical_Specification_v2_backup.md
```

---

## PHASE 2: CONSOLIDATE DUPLICATE SCRIPTS (Safe, ~30KB)

### Analysis Scripts (Keep Latest Versions)

**Files to DELETE:**
```bash
# Keep analyze_direction_labels_v2.py (latest), delete old versions
rm analyze_direction_labels.py
rm analyze_labels.py
rm analyze_labels_simple.py
```

**Files to KEEP:**
- `analyze_direction_labels_v2.py` (6.5KB) - Latest version

### Data Verification Scripts (Keep Optimized Version)

**Files to DELETE:**
```bash
# Keep verify_data_coverage_efficient.py (optimized), delete original
rm verify_data_coverage.py
```

**Files to KEEP:**
- `verify_data_coverage_efficient.py` (7.9KB) - Optimized version

---

## PHASE 3: ARCHIVE DEPRECATED CODE (235MB)

**Directory:** `deprecated_code/`

### Contents to Archive:
- `v6_backup/` - Complete v6 system (CLOSE-based bounce detection - INCORRECT)
- `alternator/` - Old alternator system with alerts, CLI, data handling
- `backend/` - FastAPI backend (not in use)
- `historicalevents/` - Historical events processing
- `investigation_scripts/` - Development debug scripts
- `models/` - Old model checkpoints
- `notebooks/` - Old Jupyter notebooks
- `reports/` - Old analysis reports
- 30+ deprecated markdown documentation files

### Archive Command:

```bash
# Create compressed archive
tar -czf x8_deprecated_v6_backup_$(date +%Y%m%d).tar.gz deprecated_code/

# Verify archive was created successfully
tar -tzf x8_deprecated_v6_backup_*.tar.gz | head -20

# Move archive to safe location (external drive or cloud storage)
# Example: mv x8_deprecated_v6_backup_*.tar.gz ~/Backups/

# ONLY AFTER VERIFYING ARCHIVE: Delete local directory
# rm -rf deprecated_code/
```

**IMPORTANT:** Do NOT delete `deprecated_code/` until archive is verified and backed up!

---

## PHASE 4: ARCHIVE OLD TRAINING RUNS (Optional, ~4GB)

**Directory:** `runs/`

### Current Runs (9 total):
```bash
ls -lt runs/
```

### Recommended Strategy:
1. **Keep:** Last 3 most recent runs (~1.5GB)
2. **Archive:** Older runs (6 runs, ~3-4GB)

### Archive Command:

```bash
# List runs sorted by date (newest first)
ls -lt runs/

# Archive older runs (adjust count as needed)
# Keep only the 3 newest, archive the rest
cd runs/
KEEP_COUNT=3
ls -t | tail -n +$((KEEP_COUNT + 1)) | xargs -I {} tar -czf ../archived_runs_{}.tar.gz {}

# Move archives to external storage
# mv ../archived_runs_*.tar.gz ~/Backups/

# ONLY AFTER VERIFYING ARCHIVES: Delete old run directories
# ls -t | tail -n +$((KEEP_COUNT + 1)) | xargs rm -rf
cd ..
```

**IMPORTANT:** Verify archives before deleting run directories!

---

## FILES TO KEEP (DO NOT DELETE)

### Multiple Dashboard Files (All serve different purposes)
- `dashboard.py` (37KB) - Rich terminal dashboard
- `streamlit_dashboard.py` (96KB) - Web dashboard (PRIMARY for web UI)
- `interactive_dashboard.py` (43KB) - Textual terminal UI

**Reason:** Each serves a different use case (web, terminal, interactive)

### Training Entry Points
- `train.py` (159KB) - Interactive CLI (PRIMARY)
- `train_cli.py` (30KB) - Non-interactive CLI (for automation)

**Reason:** Interactive for manual use, CLI for scripts/automation

### Label Inspector Tools
- `label_inspector.py` (33KB) - Interactive visualization
- `v7/tools/label_inspector.py` - Advanced validation with suspicious detection

**Reason:** Different features (visualization vs. validation)

### All Test Files (50+ files)
**Reason:** Essential for validation (19 core tests all passing)

---

## CLEANUP SCRIPT (SAFE OPERATIONS ONLY)

**File:** `cleanup_safe.sh`

```bash
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
echo "See CLEANUP_PLAN.md for detailed instructions."
```

---

## EXECUTION CHECKLIST

### Immediate (Safe Operations)
- [ ] Review this cleanup plan
- [ ] Run `cleanup_safe.sh` to delete backups and duplicates (~130KB freed)
- [ ] Verify no functionality is broken

### Manual (Requires Backup)
- [ ] Create archive of `deprecated_code/` directory
- [ ] Verify archive integrity
- [ ] Move archive to external storage
- [ ] Delete local `deprecated_code/` directory (235MB freed)

### Optional (Large Space Savings)
- [ ] Identify old training runs to archive
- [ ] Create archives of old runs
- [ ] Verify archives
- [ ] Move archives to external storage
- [ ] Delete archived run directories (~4GB freed)

---

## VERIFICATION COMMANDS

**Check what will be deleted:**
```bash
# List files that will be deleted by safe cleanup
ls -lh v7/data/live_fetcher_backup.py 2>/dev/null || echo "Not found"
ls -lh deprecated_code/models/hierarchical_training_history.json.backup 2>/dev/null || echo "Not found"
ls -lh deprecated_code/Technical_Specification_v2_backup.md 2>/dev/null || echo "Not found"
ls -lh analyze_direction_labels.py analyze_labels.py analyze_labels_simple.py verify_data_coverage.py 2>/dev/null || echo "Some not found"
```

**Check disk usage before cleanup:**
```bash
du -sh deprecated_code/
du -sh runs/
du -sh .
```

**Check disk usage after cleanup:**
```bash
du -sh .
```

---

## ROLLBACK PLAN

If something goes wrong:

**Safe cleanup:** Just restore from git (files were committed)
```bash
git status
git restore <deleted_file>
```

**Archived deprecated_code/:** Extract from archive
```bash
tar -xzf x8_deprecated_v6_backup_*.tar.gz
```

**Archived runs/:** Extract specific run from archive
```bash
tar -xzf archived_runs_<run_name>.tar.gz -C runs/
```

---

## RECOMMENDATIONS SUMMARY

1. **Do immediately:** Run safe cleanup script (backups + duplicates)
2. **Do soon:** Archive deprecated_code/ (235MB) to external storage
3. **Consider:** Archive old training runs (keep 3 most recent)
4. **Never delete:** Test files, dashboards, training scripts, documentation

**Total space savings: ~5.2GB** (if all phases completed)

---

**END OF CLEANUP PLAN**
