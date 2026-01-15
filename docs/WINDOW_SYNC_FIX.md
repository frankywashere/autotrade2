# Window Sync Fix - Label Inspector
**Date:** 2026-01-14
**Issue:** Window cycling shows different windows per TF
**Solution:** Option 1 - Force all TFs to show same window with N/A for unavailable

---

## Problem Discovered (8 Agent Investigation)

### User Report
"When you hit 'w' to change the windows its not cycling ALL the tf's windows being displayed"

### Root Cause Found

When pressing 'w' to cycle through windows (10, 20, 30, ..., 80), each timeframe was independently deciding which window to display:

**Before Fix:**
```
User presses 'w' → selects window=50
Console: "Displaying: Window 50"
Title: "Window: 50"

Actual windows displayed:
- 5min:  window=50 ✅ (requested)
- 15min: window=40 ❌ (fell back to its "best")
- 1h:    window=30 ❌ (fell back to its "best")
- daily: window=20 ❌ (fell back to its "best")
```

**Why this happened:**
- Each TF detects channels at all 8 windows independently
- Each TF selects its own "best" window (highest bounce count)
- When user-selected window unavailable → **silent fallback** to TF's best
- No visual indication of the fallback

---

## Investigation Summary (8 Parallel Agents)

### Agent 1: Found fallback logic at lines 689-696
- Code fell back to `best_window` when `display_window not in channels_dict`
- No enforcement of window consistency across TFs

### Agent 2: Confirmed per-TF channels_dict
- Each TF gets its own `channels_dict` from independent multi-window detection
- Different TFs can have different available windows

### Agent 3: Traced actual_display_window
- Variable set per TF, not globally synchronized
- Each TF could override with its own best

### Agent 4: Found data availability varies
- Higher TFs have fewer bars after resampling
- Monthly: 30 bars (only windows 10-20 fit)
- 3month: 6-10 bars (only window 10, sometimes)

### Agent 5-8: Confirmed architecture
- Multi-window detection is per-TF independent
- Each TF optimizes for its own data characteristics
- Fallback behavior was intentional but confusing

---

## Solution Implemented

### Change 1: Enforce Window Consistency (Lines 689-700)

**Before:**
```python
if display_window is not None and display_window in channels_dict:
    channel = channels_dict[display_window]
    actual_display_window = display_window
else:
    # FALLBACK to best window
    channel = best_channel
    actual_display_window = best_window
```

**After:**
```python
if display_window is not None:
    # User selected a specific window - enforce it for ALL TFs
    if display_window in channels_dict:
        # Window available for this TF
        channel = channels_dict[display_window]
        actual_display_window = display_window
    else:
        # Window NOT available - show as N/A (don't fall back to best)
        channel = None  # ← Triggers N/A display
        actual_display_window = display_window  # ← Keep requested window
else:
    # No specific window selected - show each TF's best window
    channel = best_channel
    actual_display_window = best_window
```

**Key changes:**
- Added check: `if display_window is not None:` (outer condition)
- When window unavailable: Set `channel = None` instead of using `best_channel`
- Keep `actual_display_window = display_window` (don't override with best)
- Still allow "best" mode when `display_window is None`

### Change 2: Enhanced N/A Display (Lines 219-259)

**Before:**
```python
if df_tf is None or channel is None:
    ax.text(0.5, 0.5, f'{tf_name}\nNo Data', ...)
    ax.set_title(tf_name)
    return
```

**After:**
```python
# Handle missing data vs unavailable window separately
if df_tf is None:
    # No data at all
    ax.text(0.5, 0.5, f'{tf_name}\nNo Data', ...)
    return

if channel is None:
    # Window not available - show styled N/A
    ax.set_facecolor('#f5f5f5')  # Gray background

    # Gray borders with dashed style
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(1.5)
        spine.set_linestyle('--')

    # Reason message
    if display_window is not None:
        reason = f'Window {display_window} unavailable\n(Insufficient historical data)'
    else:
        reason = 'Channel detection failed'

    # Styled N/A message box
    props = dict(boxstyle='round', facecolor='#e0e0e0', alpha=0.7,
                 edgecolor='#999999', linewidth=2, linestyle='--')
    ax.text(0.5, 0.5, f'{tf_name}\n\nN/A\n\n{reason}', ...)

    # Gray title
    ax.set_title(f"{tf_name} - Window {display_window} N/A", color='gray')

    # Faded grid
    ax.grid(True, alpha=0.1, linestyle=':', color='gray')
    return
```

**Visual enhancements:**
- Gray background (#f5f5f5)
- Dashed gray borders
- Styled message box with rounded corners
- Clear "Window X unavailable" message
- Explanation text
- Faded grid for consistency

---

## New Behavior

### When Pressing 'w' to Cycle Windows

**Scenario 1: All TFs have the window**
```
Press 'w' → window=20
All 4 panels show: "Win: 20"
All display the same window ✅
```

**Scenario 2: Some TFs missing the window**
```
Press 'w' → window=80
5min:  Shows window=80 ✅
15min: Shows window=80 ✅
1h:    Shows "Window 80 N/A" (grayed out) ⚠️
daily: Shows "Window 80 N/A" (grayed out) ⚠️
```

**Scenario 3: "Best" mode (cycle back to start)**
```
Press 'w' repeatedly until "best" mode
5min:  Shows "Win: 50*" (its best)
15min: Shows "Win: 40*" (its best)
1h:    Shows "Win: 30*" (its best)
daily: Shows "Win: 20*" (its best)
Each TF shows its own optimal window ✅
```

---

## Window Availability Reference

**At typical sample (position ~200,000, mid-dataset):**

| Timeframe | Available TF Bars | Available Windows |
|-----------|-------------------|-------------------|
| 5min | 200,000 | 10, 20, 30, 40, 50, 60, 70, 80 ✅ |
| 15min | 66,666 | 10, 20, 30, 40, 50, 60, 70, 80 ✅ |
| 30min | 33,333 | 10, 20, 30, 40, 50, 60, 70, 80 ✅ |
| 1h | 16,666 | 10, 20, 30, 40, 50, 60, 70, 80 ✅ |
| 2h | 8,333 | 10, 20, 30, 40, 50, 60, 70, 80 ✅ |
| 3h | 5,555 | 10, 20, 30, 40, 50, 60, 70, 80 ✅ |
| 4h | 4,166 | 10, 20, 30, 40, 50, 60, 70, 80 ✅ |
| daily | 2,564 | 10, 20, 30, 40, 50, 60, 70, 80 ✅ |
| weekly | 512 | 10, 20, 30, 40, 50, 60, 70, 80 ✅ |
| monthly | 122 | 10, 20, 30, 40, 50, 60, 70, 80 ✅ |
| 3month | 40 | 10, 20, 30, 40 (no 50-80) ⚠️ |

**Early sample (position ~50,000):**
- monthly: 30 bars → windows 10-20 only
- 3month: 10 bars → window 10 only (barely)

---

## Testing Checklist

- [x] Python syntax validated
- [x] Window sync logic implemented
- [x] N/A display enhanced with styling
- [ ] Manual test: Press 'w' to window=50 → all TFs show same window
- [ ] Manual test: Press 'w' to window=80 → monthly/3month show N/A
- [ ] Manual test: Press 'w' back to "best" → each TF shows its own best
- [ ] Manual test: N/A panels have gray background and clear message
- [ ] Manual test: Window titles update correctly for all cases

---

## Files Modified

**File:** `label_inspector.py`

**Changes:**
1. **Lines 689-700:** Window consistency enforcement (no fallback when user selects window)
2. **Lines 219-259:** Enhanced N/A display with styling and clear messaging

**Total lines changed:** ~45 lines (2 sections)

---

## Benefits

### Before Fix:
- ❌ Confusing: UI says window=50 but shows 4 different windows
- ❌ Silent: No indication some TFs used fallback
- ❌ Inconsistent: Can't compare channels across TFs at same window

### After Fix:
- ✅ Consistent: All TFs show the SAME window when user selects it
- ✅ Clear: N/A panels clearly marked with gray styling
- ✅ Informative: Message explains why window unavailable
- ✅ Flexible: Can still use "best" mode for TF-specific optimal windows

---

## Usage Guide

### To Compare Same Window Across TFs:
1. Press 'w' to select specific window (e.g., 50)
2. All TFs show window=50 (or N/A if unavailable)
3. Compare channel patterns at the same timescale

### To See Each TF's Optimal Window:
1. Press 'w' repeatedly until "best" mode
2. Console shows: "Displaying: Best window (auto-selected)"
3. Each TF shows its own best window with asterisk (Win: 40*)

### Visual Indicators:
- **Normal panel:** Full colors, price data, channel lines
- **N/A panel:** Gray background, dashed borders, "N/A" message
- **Window in title:** "Win: 50" (showing requested window) or "Win: 30*" (showing best with asterisk)

---

**Status:** Implemented and ready for testing ✅
