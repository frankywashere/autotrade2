# Channel Visualizer - Dead Simple Instructions

## What This Does

Shows you the trading channels the AI is learning from. You'll see:
- Price chart with channel lines
- Dots showing where price bounced
- Quality scores comparing different metrics

---

## How to Run It

### Step 1: Open Terminal

```bash
cd /Users/frank/Desktop/CodingProjects/autotrade2
```

### Step 2: Run This Command

```bash
python tools/visualize_channels.py
```

### Step 3: Answer the Questions

**Question 1: Where are your files?**
```
📂 Select shard storage location:
  → Pick "Default" if you stored locally
  → Pick "Last used" if you used external drive before
  → Pick "Custom" to enter external drive path like /Volumes/MyDrive/shards
```

**Question 2: What do you want to see?**
```
🎯 Specific channel → If you know exact time/date
⭐ High-quality → See "good" channels (AI trusts these)
⚠️  Low-quality → See "bad" channels (AI doesn't trust these)
```

**Question 3: If you picked "Specific":**
```
Symbol: TSLA or SPY
Timeframe: 1 Hour, 4 Hour, etc.
Window: 168 (one week of hourly bars)
Timestamp: Type a date like "2023-06-15 14:30" OR type "random"
```

### Step 4: Look at the Chart

A graph will pop up showing:
- **Black line** = Actual TSLA price
- **Red dashed line** = Top of channel
- **Green dashed line** = Bottom of channel
- **Red dots** = Price touched top
- **Green dots** = Price touched bottom

**Close the window** to go back to the menu and see another channel.

---

## Common Uses

### "Show me a good channel"
```bash
python tools/visualize_channels.py
> Default (or your external drive)
> Random high-quality channels
# Shows 5 great channels - close each to see next
```

### "Show me what happened on June 15th"
```bash
python tools/visualize_channels.py
> Default
> Specific channel
> TSLA
> 1 Hour
> 168
> 2023-06-15 14:30
# Shows channel at that exact time
```

### "Show me a bad channel to understand quality"
```bash
python tools/visualize_channels.py
> Default
> Random low-quality channels
# Shows 5 choppy/weak channels
```

---

## Requirements

**Before first use:**
```bash
pip install matplotlib InquirerPy
```

**Data needed:**
You must have run training at least once to generate the channel data files.

---

## Troubleshooting

**"No shard metadata found"**
→ You haven't run training yet. Run `python train_hierarchical.py` first.

**"Error loading shards"**
→ Check your external drive is plugged in and path is correct.

**"Insufficient data"**
→ Date you entered is too early. Try a later date or type "random".

---

## That's It!

Just run `python tools/visualize_channels.py` and follow the prompts.

Close chart windows to continue browsing.

Press `Ctrl+C` to exit anytime.
