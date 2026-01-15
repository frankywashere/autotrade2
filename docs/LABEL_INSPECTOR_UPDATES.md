# Label Inspector Updates - 2026-01-14

## New Feature: Timeframe View Swapping

### What Was Added

The label inspector now supports **swapping between different timeframe views** to see longer timeframes including 4hr, daily, weekly, and monthly!

### Three Timeframe Views

1. **Mixed View (default):**
   - 5min, 15min, 1h, daily
   - Good for seeing both intraday and daily trends
   - Original view from before this update

2. **Intraday View:**
   - 5min, 1h, 2h, 4hr
   - **Now you can see 4hr!**
   - Perfect for short-term trading analysis
   - Shows progression from minutes to 4-hour bars

3. **Multi-day View:**
   - 4hr, daily, weekly, monthly
   - **Now you can see weekly and monthly!**
   - Perfect for longer-term trend analysis
   - Shows progression from 4-hour to monthly timeframes

### How to Use

#### Keyboard Shortcut (Recommended)
Press **`t`** to cycle through views:
```
Mixed → Intraday → Multi-day → Mixed (loops)
```

#### Button
Click the **🔄 Swap TFs** button at the bottom of the window

### Updated Keyboard Controls

```
LEFT/RIGHT : Previous/Next sample
r          : Random sample
f          : Next flagged (suspicious) sample
F          : Previous flagged sample
w          : Cycle through window sizes (best -> 10 -> 20 -> ... -> 80 -> best)
t          : Swap timeframe views (mixed -> intraday -> multiday -> mixed)  ⭐ NEW
q/ESC      : Quit
```

### Why This Was Added

**User feedback:** "The inspector only shows up to 4hr? Can we have a button that swaps the ones viewed to show 4hr, 1day, 1month, 3month?"

**Solution:** Instead of just showing 4hr, we created three different views so you can:
- See intraday progression (5min → 4hr)
- See multi-day progression (4hr → monthly)
- Keep the original mixed view for quick analysis

### Technical Details

**Implementation:**
- Added `TF_VIEW_SETS` dictionary with three view definitions
- Added `tf_view_idx` instance variable to track current view
- Added `_cycle_timeframe_view()` method
- Added button and keyboard handler for 't' key
- Updated `_update_plot()` to use current view dynamically

**Files modified:**
- `label_inspector.py` (root level interactive inspector)

**Lines changed:**
- Line 47-52: Added timeframe view sets
- Line 483-484: Added view tracking instance variable
- Line 508: Added 't' key to controls help
- Line 520-536: Added swap button
- Line 553-554: Added 't' keyboard handler
- Line 574-590: Added `_cycle_timeframe_view()` method
- Line 642: Updated axes cleanup count (5 buttons now)
- Line 661-666: Made timeframes dynamic based on current view

### Example Usage

```bash
# Start inspector
python label_inspector.py

# Press 't' to switch to intraday view (see 4hr)
# You'll now see: 5min, 1h, 2h, 4hr

# Press 't' again to switch to multi-day view
# You'll now see: 4hr, daily, weekly, monthly

# Press 't' again to return to mixed view
# You'll see: 5min, 15min, 1h, daily

# You can also click the "🔄 Swap TFs" button
```

### Benefits

1. **See longer timeframes:** Finally view weekly and monthly channels visually
2. **Flexible analysis:** Switch views without restarting
3. **Better understanding:** See how channels look at different timescales
4. **4hr visibility:** The original request - now you can see 4hr timeframe!
5. **Pattern recognition:** Compare short-term vs long-term channel behavior

### Testing

To test the new feature:
```bash
python label_inspector.py

# Try these:
# 1. Press 't' - should show intraday view (5min, 1h, 2h, 4hr)
# 2. Press 't' - should show multi-day view (4hr, daily, weekly, monthly)
# 3. Press 't' - should cycle back to mixed view
# 4. Click "🔄 Swap TFs" button - should also cycle views
# 5. Press 'w' to cycle windows while in different views - should work normally
# 6. Navigate samples with arrows - view should persist
```

### Note

The module-level inspector (`v7/tools/label_inspector.py`) was not modified as it uses a text-based interface showing all 11 timeframes in list form rather than a visual 2x2 grid.

---

**Status:** Implemented and ready to use ✅
