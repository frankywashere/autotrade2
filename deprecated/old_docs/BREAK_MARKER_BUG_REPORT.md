# Break Marker Visualization Bug Report

## Problem Summary

Break markers (vertical lines with arrows) are not being drawn for non-permanent breaks, even though the x-axis extends to show them and the metrics overlay displays "Bars to 1st Break".

## Root Cause

There is a logic inconsistency in `/Users/frank/Desktop/CodingProjects/x14/v15/dual_inspector.py` between:

1. **Break marker drawing condition** (line 738)
2. **X-axis limit calculation** (lines 774-777)

### The Inconsistency

**Break Marker Drawing** (line 738):
```python
if labels.permanent_break and labels.break_scan_valid:
    # Draw break marker at break_bar position
```
- Requires BOTH `permanent_break=True` AND `break_scan_valid=True`
- Only draws markers for permanent breaks

**X-Axis Limits** (lines 774-777):
```python
if labels.break_scan_valid and labels.bars_to_first_break > 0:
    max_x = window - 1 + labels.bars_to_first_break + 5
else:
    max_x = window - 1 + 20
```
- Only requires `break_scan_valid=True` (ignores `permanent_break`)
- Extends x-axis for ALL breaks (permanent or not)

## Observed Behavior

For samples where:
- `permanent_break = False`
- `break_scan_valid = True`
- `bars_to_first_break > 0` (e.g., 58)

**What happens:**
1. ✓ Metrics overlay shows "Bars to 1st Break: 58"
2. ✓ X-axis extends to position ~112 to accommodate the break at x=107
3. ✓ Channel projection extends to show where break occurs
4. ✗ **No break marker (vertical line + arrow) is drawn**

## Evidence from Screenshot

From the user's screenshot:
- Window: 50 bars
- TSLA shows "Bars to 1st Break: 58"
- Break should be at position: window - 1 + bars_to_first_break = 49 + 58 = 107
- X-axis extends to ~100 (projection visible)
- **No vertical line or arrow visible at x=107**

## Sample Analysis

### Sample 100 (matches screenshot description)
```
Timestamp: 2016-05-18 22:20:00
Window: 50
TSLA 5min:
  permanent_break: False  ← This prevents marker from being drawn
  break_scan_valid: True
  bars_to_first_break: 58

Calculated positions:
  break_bar: 107 (49 + 58)
  project_forward: 63 (58 + 5)
  max_x: 112 (49 + 58 + 5)
  xlim: (-0.5, 112.5)

Result:
  - X-axis extends correctly to 112.5
  - Break marker NOT drawn (permanent_break=False)
```

### Frequency of Issue

Out of first 10 samples with window=50:
- 8 samples have non-permanent breaks with bars_to_first_break > 0
- All 8 have extended x-axis but no break marker
- Only 2 samples with permanent_break=True show markers

## Impact

**User Experience:**
- Confusing: Metrics say "Break at 58 bars" but no visual marker appears
- Wasted space: X-axis extends but shows nothing important in that region
- Inconsistent: Sometimes markers appear, sometimes they don't

**Data Visibility:**
- Non-permanent breaks are invisible despite being detected
- Hard to visually correlate metrics text with chart
- Difficult to understand what "Bars to 1st Break" means visually

## Proposed Solutions

### Option A: Draw markers for all breaks (RECOMMENDED)
**Change line 738 from:**
```python
if labels.permanent_break and labels.break_scan_valid:
```
**To:**
```python
if labels.break_scan_valid:
```

**Pros:**
- Shows all detected breaks visually
- Consistent with x-axis extension logic
- Matches what metrics overlay displays
- Better user experience

**Cons:**
- May clutter view with many markers
- Non-permanent breaks might be less important

### Option B: Only extend x-axis for permanent breaks
**Change lines 774-775 from:**
```python
if labels.break_scan_valid and labels.bars_to_first_break > 0:
    max_x = window - 1 + labels.bars_to_first_break + 5
```
**To:**
```python
if labels.permanent_break and labels.break_scan_valid and labels.bars_to_first_break > 0:
    max_x = window - 1 + labels.bars_to_first_break + 5
```

**Pros:**
- Consistent with marker drawing logic
- Saves space on chart
- Focuses on permanent breaks

**Cons:**
- Hides non-permanent breaks completely
- Metrics still show "Bars to 1st Break" but it's off-screen
- Less information visible

### Option C: Different marker style for non-permanent breaks
**Keep both conditions separate, add:**
```python
# Permanent breaks: solid line
if labels.permanent_break and labels.break_scan_valid:
    plot_break_marker(ax, break_bar, break_dir, color='purple', ...)

# Non-permanent breaks: dashed line, no arrow
elif labels.break_scan_valid and labels.bars_to_first_break > 0:
    ax.axvline(break_bar, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
```

**Pros:**
- Visual distinction between permanent/non-permanent
- Shows all breaks but with different emphasis
- Most informative option

**Cons:**
- More complex code
- Needs legend to explain

## Recommendation

**Implement Option A** (simplest fix) or **Option C** (most informative).

Option B would hide information that the scanner already computed, which seems wasteful. Users expect to see what the metrics display.

## Code Locations

All issues in: `/Users/frank/Desktop/CodingProjects/x14/v15/dual_inspector.py`

- Line 738: Break marker drawing condition
- Lines 726-729: Forward projection calculation
- Line 734: plot_channel_bounds call
- Lines 774-780: X-axis limit calculation

## Related Files

- `/Users/frank/Desktop/CodingProjects/x14/v15/inspector_utils.py`: Contains `plot_break_marker()` function
- `/Users/frank/Desktop/CodingProjects/x14/v15/labels.py`: Defines `permanent_break` and `break_scan_valid` logic
- `/Users/frank/Desktop/CodingProjects/x14/v15/core/break_scanner.py`: Computes break detection
