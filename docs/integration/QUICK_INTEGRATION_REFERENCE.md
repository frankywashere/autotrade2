# Dashboard Live Integration - Quick Reference Card

## 🚀 Quick Start (5 Minutes)

### Step 1: Test the Module
```bash
python test_live_integration.py
```
Expected: All 5 tests pass ✅

### Step 2: Edit dashboard.py (2 lines)

**Line 43** (add import):
```python
from v7.data.live import load_live_data_tuple
```

**Line 627** (replace):
```python
# OLD:
tsla_df, spy_df, vix_df = load_data(args.lookback)

# NEW:
tsla_df, spy_df, vix_df = load_live_data_tuple(args.lookback)
```

### Step 3: Run Dashboard
```bash
python dashboard.py --refresh 300
```

**Done!** You now have live data. 🎉

---

## 📊 Data Status Meanings

| Status | Data Age | Meaning |
|--------|----------|---------|
| 🟢 **LIVE** | < 15 min | Fresh live data from yfinance |
| 🟡 **RECENT** | 15-60 min | Slightly old, still usable |
| 🔴 **STALE** | > 60 min | Old data, check yfinance |
| ⚪ **HISTORICAL** | N/A | CSV only (no yfinance) |

---

## 💡 Common Commands

```bash
# Basic dashboard with live data
python dashboard.py --refresh 300

# With trained model
python dashboard.py --model checkpoints/best.pt --refresh 60

# Export predictions
python dashboard.py --export results/ --refresh 300

# Force CSV-only (no yfinance)
python dashboard.py --force-historical --refresh 300

# Test integration
python test_live_integration.py
```

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `v7/data/live.py` | Core live data module |
| `test_live_integration.py` | Test suite |
| `LIVE_INTEGRATION_README.md` | Full guide |
| `dashboard_integration_snippet.py` | Code examples |

---

## 🔧 Two Integration Options

### Option A: Minimal (Recommended First)
- **Changes**: 2 lines
- **Time**: 5 minutes
- **Code**: Use `load_live_data_tuple()`

### Option B: Full (Enhanced)
- **Changes**: ~15 lines
- **Time**: 15 minutes
- **Code**: Use `fetch_live_data()` + status display
- **Benefit**: Shows data freshness

---

## ⚡ API Functions

```python
from v7.data.live import (
    fetch_live_data,           # Main function (returns LiveDataResult)
    load_live_data_tuple,      # Backward compatible (returns tuple)
    LiveDataResult,            # Data class with metadata
    is_market_open             # Check if market is open
)

# Example 1: Full integration
result = fetch_live_data(lookback_days=90)
tsla_df = result.tsla_df
spy_df = result.spy_df
vix_df = result.vix_df
status = result.status

# Example 2: Minimal integration
tsla_df, spy_df, vix_df = load_live_data_tuple(lookback_days=90)

# Example 3: Check market
if is_market_open():
    print("Market is open!")
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | Run from `/Users/frank/Desktop/CodingProjects/x6` |
| Always HISTORICAL | Check network, test with `test_live_integration.py` |
| yfinance error | Run `pip install yfinance` |
| CSV not found | Verify files in `data/` directory |

---

## ✅ Testing Checklist

- [ ] Run `python test_live_integration.py` → all pass
- [ ] Run `python dashboard.py` → loads without error
- [ ] Verify live data appears (check timestamp)
- [ ] Test `--refresh 60` → auto-updates work
- [ ] Test `--force-historical` → CSV-only mode works
- [ ] Verify all existing features still work

---

## 📖 Documentation

1. **This Card**: Quick reference
2. **LIVE_INTEGRATION_README.md**: Quick start guide
3. **DASHBOARD_INTEGRATION_GUIDE.md**: Complete walkthrough
4. **DASHBOARD_INTEGRATION_COMPARISON.md**: Before/after comparison
5. **DASHBOARD_LIVE_INTEGRATION_SUMMARY.md**: Full summary

---

## 💾 What Changes

### Minimal Integration
- ✅ Add 1 import line
- ✅ Change 1 function call
- ✅ Everything else stays the same

### Full Integration
- ✅ Add 1-2 imports
- ✅ Replace data loading (8-10 lines)
- ✅ Add status display (4-5 lines)
- ✅ Optional: enhance header
- ✅ Everything else stays the same

---

## 🎯 Key Benefits

- ✅ Live market data (auto-fetches from yfinance)
- ✅ Data freshness status (LIVE/RECENT/STALE)
- ✅ Automatic merge with CSV history
- ✅ Error handling (falls back to CSV)
- ✅ Backward compatible (no breaking changes)
- ✅ Minimal code changes (1-15 lines)

---

## 📞 Quick Help

**Test module works?**
```bash
python test_live_integration.py
```

**Check data status?**
```python
result = fetch_live_data()
print(result.status)  # LIVE, RECENT, STALE, or HISTORICAL
```

**Force CSV only?**
```bash
python dashboard.py --force-historical
```

**Check market hours?**
```python
from v7.data.live import is_market_open
print(is_market_open())  # True if 9:30 AM - 4:00 PM ET, Mon-Fri
```

---

## 🔄 Data Flow

```
CSV Files (TSLA, SPY, VIX)
        ↓
   Load & Filter
        ↓
yfinance API (7 days, 1min) ──→ Merge ──→ Resample to 5min
        ↓                                         ↓
   Format & Clean                         Return DataFrames
```

---

## ⏱️ Expected Performance

- CSV load: 1-3 seconds
- yfinance fetch: 2-5 seconds
- Merge/resample: <1 second
- **Total: 4-9 seconds** (vs 2-4 seconds CSV-only)

---

## 🎨 Status Display Code

```python
status_colors = {
    'LIVE': 'green',
    'RECENT': 'yellow',
    'STALE': 'red',
    'HISTORICAL': 'dim'
}
status_color = status_colors.get(result.status, 'dim')
console.print(f"Status: [{status_color}]{result.status}[/{status_color}]")
console.print(f"Data age: {result.data_age_minutes:.1f} minutes")
```

---

## 🚦 Next Steps

1. ✅ Created: All files ready
2. ⏳ **Test**: Run `python test_live_integration.py`
3. ⏳ **Integrate**: Add 2 lines to dashboard.py
4. ⏳ **Test Dashboard**: Run `python dashboard.py`
5. ⏳ **Deploy**: Use with `--refresh 300`

---

**Status**: ✅ READY TO INTEGRATE

**First Command**: `python test_live_integration.py`

**Time to Deploy**: 5-30 minutes

---

*Quick Reference v1.0 | 2026-01-02*
