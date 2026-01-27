# Break Marker Position - Visual Analysis

## Problem Visualization

### Current Behavior (BUGGY)

For Sample 100, TSLA 5min, Window 50:

```
Labels:
  permanent_break: False
  break_scan_valid: True
  bars_to_first_break: 58

Calculations:
  Channel ends at:    x = 49
  Break occurs at:    x = 107  (49 + 58)
  Projection ends at: x = 112  (50 + 63 - 1)
  X-axis max:         x = 112  (49 + 58 + 5)

Visual Layout:
  0                    49   50              107  112
  |----Channel--------|----Projection-------|---|
  ▲                   ▲                     ▲   ▲
  Start               End                   ?   X-axis end
                                           Break position
                                           (NO MARKER DRAWN)

Metrics Overlay Shows:
  "Bars to 1st Break: 58"
  ← But where is it on the chart? No visual marker!
```

### What User Sees

```
Chart appears to show:
  - Channel from x=0 to x=49 (blue lines)
  - Projection from x=50 to x=112 (dashed lines)
  - X-axis extending to x≈112
  - Text saying "Bars to 1st Break: 58"
  - ⚠️ NO vertical line at x=107
  - ⚠️ NO arrow showing break direction
  - ⚠️ Empty space from x=49 to x=112 with no visual markers
```

### Expected Behavior (FIXED)

```
Visual Layout SHOULD BE:
  0                    49   50              107  112
  |----Channel--------|----Projection-------|---|
  ▲                   ▲                     ▲   ▲
  Start               End                   |   X-axis end
                                           MARKER HERE
                                           (vertical line + arrow)

With marker visible:
  - User can see WHERE the break occurs
  - Visual correlation with "Bars to 1st Break: 58" text
  - Clear indication of break position at x=107
```

## Code Logic Flow

### Current (Inconsistent)

```python
# Step 1: Calculate projection distance
if labels.break_scan_valid and labels.bars_to_first_break > 0:
    project_forward = labels.bars_to_first_break + 5  # = 63
else:
    project_forward = 20

# Step 2: Plot channel with projection
plot_channel_bounds(ax, channel, 0, window, project_forward=63)
# Draws channel lines from x=0 to x=49
# Draws projection lines from x=50 to x=112

# Step 3: BROKEN - Only draw marker if permanent_break=True
if labels.permanent_break and labels.break_scan_valid:  # False AND True = False
    break_bar = window - 1 + labels.bars_to_first_break  # = 107
    plot_break_marker(ax, break_bar, ...)  # NOT EXECUTED

# Step 4: Set x-axis limits
if labels.break_scan_valid and labels.bars_to_first_break > 0:
    max_x = window - 1 + labels.bars_to_first_break + 5  # = 112
ax.set_xlim(-0.5, max_x + 0.5)  # = (-0.5, 112.5)

RESULT: Projection extends to break, x-axis shows it, but no marker drawn!
```

### Fixed (Consistent)

```python
# Step 1: Calculate projection distance
if labels.break_scan_valid and labels.bars_to_first_break > 0:
    project_forward = labels.bars_to_first_break + 5  # = 63
else:
    project_forward = 20

# Step 2: Plot channel with projection
plot_channel_bounds(ax, channel, 0, window, project_forward=63)
# Draws channel lines from x=0 to x=49
# Draws projection lines from x=50 to x=112

# Step 3: FIXED - Draw marker if break exists
if labels.break_scan_valid and labels.bars_to_first_break > 0:  # True AND True = True
    break_bar = window - 1 + labels.bars_to_first_break  # = 107

    # Optional: Different style for non-permanent breaks
    if labels.permanent_break:
        plot_break_marker(ax, break_bar, ..., linestyle='--', color='purple')
    else:
        plot_break_marker(ax, break_bar, ..., linestyle=':', color='gray')

# Step 4: Set x-axis limits
if labels.break_scan_valid and labels.bars_to_first_break > 0:
    max_x = window - 1 + labels.bars_to_first_break + 5  # = 112
ax.set_xlim(-0.5, max_x + 0.5)  # = (-0.5, 112.5)

RESULT: Everything consistent - marker visible at x=107!
```

## Affected Samples

From small_sample.pkl (249 samples total):

| Sample | Timestamp | TF | bars_to_first_break | permanent_break | Marker Drawn? |
|--------|-----------|----|--------------------|-----------------|---------------|
| 1 | 2016-01-28 09:40:00 | 5min | 7 | False | ❌ NO |
| 2 | 2016-01-28 14:15:00 | 5min | 7 | False | ❌ NO |
| 5 | 2016-02-01 13:20:00 | 5min | 15 | False | ❌ NO |
| 6 | 2016-02-02 14:20:00 | 5min | 5 | False | ❌ NO |
| 8 | 2016-02-04 14:30:00 | 5min | 7 | False | ❌ NO |
| 9 | 2016-02-05 13:35:00 | 5min | 3 | False | ❌ NO |
| **100** | **2016-05-18 22:20:00** | **5min** | **58** | **False** | **❌ NO** |
| ... | ... | ... | ... | ... | ... |

**63 out of 249 samples** have non-permanent breaks with bars_to_first_break > 20.
All of these have missing markers but extended x-axes.

## Screenshot Analysis

From user's screenshot showing "Bars to 1st Break: 58":

### What the calculations say:
```
Window: 50
bars_to_first_break: 58
break_bar position: 49 + 58 = 107
max_x: 49 + 58 + 5 = 112
xlim: (-0.5, 112.5)
```

### What the screenshot shows:
- ✓ X-axis extends to approximately 112
- ✓ Channel visible from 0 to 49
- ✓ Projection visible from 50 to ~112
- ✓ Metrics text shows "Bars to 1st Break: 58"
- ❌ No vertical line visible at x=107
- ❌ No arrow indicating break direction
- ❌ No visual marker anywhere in the projection area

### Why?
```python
# Line 738 in dual_inspector.py
if labels.permanent_break and labels.break_scan_valid:
    # Only executed if permanent_break=True
    # In screenshot case: False AND True = False
    # So this entire block is skipped
    plot_break_marker(...)
```

## Conclusion

The break markers are **correctly calculated** but **not being drawn** due to an overly restrictive condition that requires `permanent_break=True`. The x-axis extension logic doesn't have this restriction, creating a confusing situation where the chart extends to show a break that has no visual marker.

**Fix:** Remove the `permanent_break` requirement from the marker drawing condition (line 738), or add it to the x-axis calculation (lines 774-777) to make them consistent.
