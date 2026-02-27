# Label Inspector Fixes V2 - 2026-01-14
## Issues: Missing 3month Timeframe & Window Display Clarity

---

## Investigation Summary (8 Parallel Agents)

### Issue 1: Missing 3month Timeframe
**Found:** `3month` was not included in any of the 3 timeframe view sets
**Impact:** Users could not view the longest timeframe (3-month) channels

### Issue 2: Window Cycling Unclear
**Found:** Window cycling WAS working correctly, but display was not prominent
**Impact:** Users couldn't easily see which window (10, 20, 30, etc.) was being displayed

---

## Detailed Findings

### Agent 1: 3month Timeframe Investigation
- ✅ `3month` exists in global TIMEFRAMES list
- ✅ `3month` exists in cached sample data
- ✅ `3month` visualization code exists (FORWARD_BARS_PER_TF)
- ❌ `3month` NOT in any TF_VIEW_SETS view
- **Conclusion:** Simple oversight - just needs to be added to multiday view

### Agent 2: Window Cycling Logic
- ✅ _cycle_window() correctly cycles through [10, 20, 30, 40, 50, 60, 70, 80]
- ✅ Wraps: best → 10 → ... → 80 → best
- ✅ display_window_idx managed correctly
- ✅ Prints to console: "Displaying: Window X"
- **Conclusion:** Logic is perfect

### Agent 3: Window Display in Plots
- ✅ Main title shows "Window: X" or "Window: Best"
- ✅ Text annotation box shows window scores
- ⚠️ Subplot TITLES don't show window (only R² and width)
- **Conclusion:** Working but not prominent enough

### Agent 4: STANDARD_WINDOWS Definition
- ✅ Correctly defined: [10, 20, 30, 40, 50, 60, 70, 80]
- ✅ All 8 windows present
- ✅ Properly imported in label_inspector.py
- **Conclusion:** No issues

### Agent 5: Window Data Flow
- ✅ display_window_idx → window size conversion correct
- ✅ Window passed through entire visualization chain
- ✅ No bugs found in data flow
- **Conclusion:** Data flow is solid

### Agent 6: Subplot Annotations
- ✅ Annotation box in upper-left shows window info
- ✅ Shows "Best Window: X" and "Window scores: ..."
- ⚠️ Annotation box is small and easy to miss
- **Conclusion:** Exists but not prominent

### Agent 7: display_window Usage
- ✅ None for "best", integer for specific window
- ✅ Correctly maps from display_window_idx
- ✅ Properly handles None values
- ✅ No off-by-one errors
- **Conclusion:** Implementation perfect

### Agent 8: Figure Title Updates
- ✅ fig.suptitle() called every _update_plot()
- ✅ Includes "Window: X" or "Window: Best"
- ✅ Updates correctly when cycling
- **Conclusion:** No issues

---

## Fixes Applied

### Fix 1: Add 3month to multiday View

**Before (Line 48-52):**
```python
TF_VIEW_SETS = {
    'mixed': ['5min', '15min', '1h', 'daily'],
    'intraday': ['5min', '1h', '2h', '4h'],
    'multiday': ['4h', 'daily', 'weekly', 'monthly'],  # ← Missing 3month
}
```

**After:**
```python
TF_VIEW_SETS = {
    'mixed': ['5min', '15min', '1h', 'daily'],
    'intraday': ['5min', '1h', '2h', '4h'],
    'multiday': ['daily', 'weekly', 'monthly', '3month'],  # ← Added 3month, removed 4h
}
```

**Rationale:**
- Multiday view is for "longer-term trends"
- 4h was the shortest timeframe in that view (inconsistent)
- Progression now: daily (1d) → weekly (7d) → monthly (~21d) → 3month (~63d)
- Completes the long-term timeframe analysis

### Fix 2: Add Window Info to Subplot Titles

**Before (Line 383-387):**
```python
# Title
title = f"{tf_name}"
if channel.valid:
    title += f" - R2: {channel.r_squared:.3f}, Width: {channel.width_pct:.2f}%"
ax.set_title(title, fontsize=11, fontweight='bold')
```

**After:**
```python
# Title with window information
title = f"{tf_name}"
if channel.valid:
    title += f" - R2: {channel.r_squared:.3f}, Width: {channel.width_pct:.2f}%"
# Add window info to title for clarity
if display_window is not None:
    title += f" | Win: {display_window}"
elif best_window is not None:
    title += f" | Win: {best_window}*"
ax.set_title(title, fontsize=11, fontweight='bold')
```

**What This Does:**
- Adds "| Win: 50" to each subplot title when displaying window 50
- Adds "| Win: 30*" when displaying best window (asterisk indicates it's auto-selected)
- Makes it immediately obvious which window is being displayed
- No need to read small annotation box

### Fix 3: Update multiday View Description

**Before (Line 605):**
```python
'multiday': 'Multi-day (4h, daily, weekly, monthly)'
```

**After:**
```python
'multiday': 'Multi-day (daily, weekly, monthly, 3month)'
```

Keeps console output consistent with actual view.

---

## Example Output

### When Pressing 'w' to Cycle Windows

**Console Output:**
```
Displaying: Best window (auto-selected)
[press 'w']
Displaying: Window 10
[press 'w']
Displaying: Window 20
...
[press 'w']
Displaying: Window 80
[press 'w']
Displaying: Best window (auto-selected)
```

**Main Title:**
```
[OK] Sample 123/5000 | 2024-01-15 10:30:00 | Window: 50
```

**Subplot Titles (4 panels):**
```
5min - R2: 0.945, Width: 2.34% | Win: 50
1h - R2: 0.892, Width: 3.12% | Win: 50
2h - R2: 0.867, Width: 2.98% | Win: 50
4h - R2: 0.923, Width: 2.45% | Win: 50
```

When on "best" mode:
```
5min - R2: 0.945, Width: 2.34% | Win: 30*
1h - R2: 0.892, Width: 3.12% | Win: 40*
2h - R2: 0.867, Width: 2.98% | Win: 50*
4h - R2: 0.923, Width: 2.45% | Win: 30*
```
(Asterisk shows auto-selected best window for each TF)

---

## New Multiday View

Press 't' twice to reach multiday view:

**Now Shows:**
```
┌─────────────────────┬─────────────────────┐
│   daily             │   weekly            │
│   (1 bar = 1 day)   │   (1 bar = 1 week)  │
└─────────────────────┴─────────────────────┘
┌─────────────────────┬─────────────────────┐
│   monthly           │   3month ⭐         │
│   (1 bar = 1 month) │   (1 bar = 3 months)│
└─────────────────────┴─────────────────────┘
```

Perfect for analyzing long-term trends!

---

## Testing Checklist

- [x] Python syntax validated
- [x] 3month added to multiday view
- [x] Window info added to subplot titles
- [x] multiday description updated
- [ ] Manual test: Press 't' to reach multiday → should show daily, weekly, monthly, 3month
- [ ] Manual test: Press 'w' to cycle windows → titles should update with "Win: X"
- [ ] Manual test: All 8 windows cycle correctly (10, 20, 30, 40, 50, 60, 70, 80)
- [ ] Manual test: 3month timeframe displays correctly with channel detection

---

## Files Modified

**File:** `label_inspector.py`

**Changes:**
1. Line 51: Changed multiday view from `['4h', 'daily', 'weekly', 'monthly']` to `['daily', 'weekly', 'monthly', '3month']`
2. Lines 386-391: Added window information to subplot titles
3. Line 605: Updated multiday description

**Lines Changed:** 3 locations, ~8 lines total

---

## Benefits

### For 3month Addition:
- ✅ Can now view quarterly trends
- ✅ Complete the 11-timeframe analysis
- ✅ Better for swing/position traders
- ✅ See how channels look over ~2.5 years forward

### For Window Display:
- ✅ Immediately visible which window is displayed
- ✅ No need to squint at small annotation box
- ✅ Clear when on "best" mode (asterisk)
- ✅ Consistent across all 4 subplots
- ✅ Easier to compare different window sizes

---

## Window Cycling Behavior (Reference)

### Cycle Sequence:
```
best → 10 → 20 → 30 → 40 → 50 → 60 → 70 → 80 → best (loops)
```

### Window Resets:
- Pressing LEFT/RIGHT (navigate samples) → resets to best
- Pressing 'f' (jump to flagged) → resets to best
- Pressing 'r' (random sample) → resets to best

### Window Persists:
- Pressing 't' (swap TF views) → window stays the same
- Pressing 'w' (cycle windows) → obviously changes window

---

## Known Limitations

1. **3month requires sufficient data:** Samples near the end of the dataset may not have enough forward bars for 3month timeframe (needs ~30 months forward)
2. **Window may not exist for some TFs:** If a timeframe doesn't have enough data for window=80, it falls back to best window (this is correct behavior)
3. **Annotation box still shows detailed info:** The small text box still exists with "Window scores: 10:3b [20:5b*] 40:7b" - this is intentional for users who want detailed comparison

---

**Status:** Fixed and enhanced ✅

**Recommendation:** Test manually with `python label_inspector.py` and verify the improvements are visible.
