# Feature Name Audit - TSLA Rename (channel_* → tsla_channel_*, rsi_* → tsla_rsi_*)

**Date:** November 13, 2025
**Purpose:** Document all references to current feature names before renaming for SPY addition

---

## Summary

### Feature Names Being Renamed:
- **77 channel features**: `channel_{tf}_*` → `tsla_channel_{tf}_*`
  - Examples: `channel_1h_position` → `tsla_channel_1h_position`
  - 11 timeframes × 7 features each

- **33 RSI features**: `rsi_{tf}_*` → `tsla_rsi_{tf}_*`
  - Examples: `rsi_daily` → `tsla_rsi_daily`
  - 11 timeframes × 3 features each

**Total features being renamed:** 110

---

## Files Requiring Updates

### CRITICAL FILES:

#### 1. **src/ml/features.py** (PRIMARY - Feature Definitions)

**Lines 52-58:** Feature name list (channel features)
```python
f'channel_{tf}_position',  # Will become: f'tsla_channel_{tf}_position'
f'channel_{tf}_upper_dist',
f'channel_{tf}_lower_dist',
f'channel_{tf}_slope',
f'channel_{tf}_stability',
f'channel_{tf}_ping_pongs',
f'channel_{tf}_r_squared',
```

**Lines 64-66:** Feature name list (RSI features)
```python
f'rsi_{tf}',  # Will become: f'tsla_rsi_{tf}'
f'rsi_{tf}_oversold',
f'rsi_{tf}_overbought',
```

**Lines 207-213:** Feature extraction (channel assignment)
```python
features_df[f'channel_{tf_name}_position'] = ...
# Will become: features_df[f'tsla_channel_{tf_name}_position'] = ...
```

**Lines 263-269:** Feature extraction (RSI assignment)
```python
features_df[f'rsi_{tf_name}'] = ...
# Will become: features_df[f'tsla_rsi_{tf_name}'] = ...
```

**Action Required:**
- Update all 4 locations to use `tsla_` prefix
- Add SPY feature definitions (duplicate loops with `spy_` prefix)

---

#### 2. **test_multiscale_features.py** (Tests)

**Lines 51-57:** Test checks for channel feature names
```python
f'channel_{tf}_position',
# Should be updated to: f'tsla_channel_{tf}_position'
```

**Lines 62-64:** Test checks for RSI feature names
```python
f'rsi_{tf}',
# Should be updated to: f'tsla_rsi_{tf}'
```

**Lines 88, 90-91:** Prefix checks
```python
f.startswith(prefix) for prefix in ['channel_', 'rsi_']
# Should be updated to: ['tsla_channel_', 'tsla_rsi_', 'spy_channel_', 'spy_rsi_']
```

**Action Required:**
- Update all test assertions to check for `tsla_` prefixed names
- Add tests for `spy_` prefixed names
- Update prefix checks to include all 4 types

---

#### 3. **SPEC.md** (Documentation)

**Lines 820-826:** Multi-scale feature example
```
  - channel_5min_position: 0.65
  - channel_1h_position: 0.81
  # Should be: tsla_channel_5min_position, tsla_channel_1h_position
```

**Action Required:**
- Update example to use `tsla_` prefix
- This will be done in the second SPEC.md update (after implementation)

---

### NON-CRITICAL FILES (Different Context):

#### 4. **src/ml/database.py** (lines 142, 307, 632-633)
**Context:** These reference database columns `channel_position` and `rsi_value` (single values), NOT the 135-feature array names.
```python
channel_position=prediction.get('channel_position'),  # Single value, not feature name
rsi_value=prediction.get('rsi_value'),  # Single value, not feature name
```
**Action:** ❌ **NO CHANGE NEEDED** - These are separate database fields unrelated to feature extraction

---

#### 5. **src/signal_generator.py** (lines 131-132, 198-199)
**Context:** Stage 1 system fields, not feature extraction names
```python
'channel_position': channel_position,  # Stage 1 signal field
'rsi_confluence': rsi_confluence,  # Stage 1 signal field
```
**Action:** ❌ **NO CHANGE NEEDED** - Stage 1 code, separate from ML feature system

---

## Rename Strategy

### Phase 1: Update Feature Definitions (src/ml/features.py)

**Step 1.1:** Update `_build_feature_names()` (lines 35-100)
- Change channel loop to use `tsla_channel_{tf}_*`
- Change RSI loop to use `tsla_rsi_{tf}_*`
- Add SPY channel loop with `spy_channel_{tf}_*`
- Add SPY RSI loop with `spy_rsi_{tf}_*`

**Step 1.2:** Update `_extract_channel_features()` (lines 160-220)
- Add `for symbol in ['tsla', 'spy']:` outer loop
- Use `f'{symbol}_channel_{tf_name}_*'` for assignments
- Special handling: use `channel_{tf}_*` for TSLA (legacy compatibility) OR rename to `tsla_channel_{tf}_*` (consistency)
- **Decision made:** Rename to `tsla_channel_{tf}_*` for consistency

**Step 1.3:** Update `_extract_rsi_features()` (lines 222-271)
- Add `for symbol in ['tsla', 'spy']:` outer loop
- Use `f'{symbol}_rsi_{tf_name}_*'` for assignments
- **Decision made:** Rename to `tsla_rsi_{tf}_*` for consistency

---

### Phase 2: Update Tests (test_multiscale_features.py)

**Step 2.1:** Update expected feature names in assertions
```python
# OLD
expected_features.extend([f'channel_{tf}_position', ...])
# NEW
expected_features.extend([f'tsla_channel_{tf}_position', ...])
```

**Step 2.2:** Add SPY feature checks
```python
expected_features.extend([f'spy_channel_{tf}_position', ...])
```

**Step 2.3:** Update prefix checks
```python
# OLD
prefixes = ['channel_', 'rsi_']
# NEW
prefixes = ['tsla_channel_', 'tsla_rsi_', 'spy_channel_', 'spy_rsi_']
```

---

### Phase 3: Update Documentation (SPEC.md - Second Update)

Will be done after implementation is complete and tested.
Update examples to show:
```
tsla_channel_1h_position: 0.81
spy_channel_1h_position: 0.65
tsla_rsi_daily: 45.2
spy_rsi_daily: 52.8
```

---

## Backward Compatibility

**BREAKING CHANGE:** All models trained with 135 features will be incompatible.

**Reasons:**
1. Feature count changes: 135 → 245
2. Feature names change: `channel_*` → `tsla_channel_*`, `rsi_*` → `tsla_rsi_*`
3. Input size mismatch when loading old models

**Migration:**
- Archive all old 135-feature models to `models/archive_v3.3_135features/`
- Retrain all 4 timeframe models with new 245-feature extractor
- Document in SPEC.md as v3.4 breaking change

---

## Validation Checklist

After implementing changes:

- [ ] Feature extractor reports 245 features (not 135)
- [ ] Feature names include `tsla_channel_*` and `spy_channel_*`
- [ ] Feature names include `tsla_rsi_*` and `spy_rsi_*`
- [ ] NO old names (`channel_*`, `rsi_*` without prefix) exist
- [ ] test_multiscale_features.py passes
- [ ] 1-epoch test model trains successfully
- [ ] Model metadata contains 245 feature names
- [ ] Feature extraction produces 245 columns

---

## Summary Table

| File | Lines | Type | Action Required |
|------|-------|------|-----------------|
| src/ml/features.py | 52-66 | Feature name list | ✅ Update + add SPY |
| src/ml/features.py | 207-213 | Channel extraction | ✅ Update + add SPY |
| src/ml/features.py | 263-269 | RSI extraction | ✅ Update + add SPY |
| test_multiscale_features.py | 51-64 | Test assertions | ✅ Update |
| test_multiscale_features.py | 88, 90-91 | Prefix checks | ✅ Update |
| SPEC.md | 820-826 | Examples | ✅ Update (Phase 5) |
| src/ml/database.py | 142, 307, 632-633 | DB fields | ❌ NO CHANGE |
| src/signal_generator.py | 131-132, 198-199 | Stage 1 fields | ❌ NO CHANGE |

**Critical Files: 2** (features.py, test_multiscale_features.py)
**Documentation: 1** (SPEC.md - deferred to Phase 5)

---

**End of Audit**
