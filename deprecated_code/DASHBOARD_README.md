# Real-Time Inference Dashboard - v7 Channel Prediction System

Two dashboard implementations for real-time channel analysis and trading signal generation.

## Overview

The v7 dashboard provides live inference on the latest market data, displaying:
- Multi-timeframe channel status (all 11 timeframes)
- Model predictions (duration, direction, confidence)
- Trading signals (LONG/SHORT/WAIT)
- Event awareness (upcoming earnings, FOMC, etc.)
- Visual channel plots with boundary projections

## Dashboard Variants

### 1. Terminal Dashboard (`dashboard.py`)

Rich terminal UI with live updates and color-coded signals.

**Features:**
- Live data refresh (configurable interval)
- Multi-timeframe channel table
- Model prediction summary
- Trading signal recommendation
- Upcoming events tracker
- Export predictions to CSV

**Usage:**
```bash
# Basic usage (no model, features only)
python dashboard.py

# With trained model
python dashboard.py --model checkpoints/best_model.pt

# Auto-refresh every 5 minutes
python dashboard.py --model checkpoints/best_model.pt --refresh 300

# Export predictions
python dashboard.py --model checkpoints/best_model.pt --export results/

# Custom data lookback
python dashboard.py --lookback 60  # 60 days of data
```

**Output Example:**
```
┌──────────────────────────────────────────────────────────────┐
│   Real-Time Channel Prediction Dashboard v7.0               │
│   Time: 2024-12-31 15:30:00 ET                              │
└──────────────────────────────────────────────────────────────┘

┌─ TSLA Trading Signal ──────────────────┐  ┌─ Multi-Timeframe Channel Status ───┐
│                                         │  │ TF     Valid  Direction  Position  │
│         LONG                            │  │ 5min     ✓    ↑ BULL      0.82    │
│                                         │  │ 15min    ✓    ↑ BULL      0.74    │
│ Action: BUY 345.67                      │  │ 30min    ✓    ↔ SIDE      0.51    │
│ Expected Duration: 23 bars              │  │ 1h       ✓    ↑ BULL      0.88    │
│ Break Direction: UP                     │  │ 2h       ✗    ↔ SIDE      0.45    │
│ Next Channel: BULL                      │  │ 3h       ✓    ↑ BULL      0.67    │
│ Confidence: 89%                         │  │ 4h       ✓    ↑ BULL      0.91    │
│                                         │  │ daily    ✓    ↑ BULL      0.73    │
│ Current Price: $345.67                  │  │ weekly   ✓    ↑ BULL      0.55    │
│ SPY: $582.34                            │  │ monthly  ✓    ↔ SIDE      0.48    │
│ VIX: 14.23                              │  │ 3month   ✓    ↑ BULL      0.62    │
└─────────────────────────────────────────┘  └─────────────────────────────────────┘

┌─ Model Predictions (Per Timeframe) ────┐  ┌─ Upcoming Events ──────────────────┐
│ TF      Duration   Break Dir  Next Dir │  │ ⚠ EARNINGS    01/22 20:00  (T-3d) │
│ 5min    23 ± 5     UP (78%)   BULL     │  │ ○ FOMC        01/29 14:00  (T-7d) │
│ 15min   12 ± 3     UP (71%)   SIDE     │  │ · CPI         02/12 08:30 (T-14d) │
│ 1h      45 ± 8     UP (89%)   BULL ⭐  │  └─────────────────────────────────────┘
│ 4h      15 ± 4     UP (82%)   BULL     │
│ daily   8 ± 2      UP (76%)   BULL     │
└─────────────────────────────────────────┘

Press Ctrl+C to exit | Model: v7 Hierarchical CfC | Data: Live CSV
```

### 2. Visual Dashboard (`dashboard_visual.py`)

Matplotlib-based visual dashboard with channel plots and projections.

**Features:**
- Channel visualization with boundary lines
- Touch point markers
- Current position indicator
- Multi-timeframe grid view
- Export to PNG/PDF

**Usage:**
```bash
# Show interactive plot
python dashboard_visual.py

# With model predictions
python dashboard_visual.py --model checkpoints/best_model.pt

# Save to file
python dashboard_visual.py --save dashboard_2024-12-31.png

# Select specific timeframes
python dashboard_visual.py --tf 1h 4h daily

# Custom lookback
python dashboard_visual.py --lookback 60
```

**Output:**
The visual dashboard creates a matplotlib figure with:
- Header: Trading signal, confidence, predictions
- Channel plots: One per timeframe showing:
  - Price line (black)
  - Center line (dashed, colored by direction)
  - Upper/lower boundaries (solid, colored)
  - Touch points (green=lower, red=upper)
  - Current position indicator
  - Channel metrics (bounces, cycles, R²)

## Data Requirements

The dashboards load data from:
```
/Volumes/NVME2/x6/data/
├── TSLA_1min.csv      # TSLA 1-minute OHLCV
├── SPY_1min.csv       # SPY 1-minute OHLCV
├── VIX_History.csv    # VIX daily data
└── events.csv         # Market events (earnings, FOMC, etc.)
```

**CSV Formats:**

TSLA/SPY (1-minute bars):
```
Datetime,Open,High,Low,Close,Volume
2024-01-02 09:30:00,248.42,248.86,248.31,248.58,1234567
...
```

VIX (daily):
```
Date,Open,High,Low,Close
2024-01-02,12.45,12.89,12.31,12.67
...
```

events.csv:
```
date,event_type,expected,actual,surprise_pct,beat_miss,source,release_time
2024-01-24,earnings,0.73,0.80,9.6,1,AlphaVantage,20:00
2024-01-31,fomc,,,,,FRED,14:00
...
```

## Model Integration

To use with a trained model:

1. Train model using `v7/training/trainer.py`
2. Save checkpoint with model state dict
3. Pass checkpoint path to dashboard: `--model path/to/checkpoint.pt`

The dashboard will:
- Load model weights
- Extract features from latest data
- Run inference to get predictions
- Display confidence-weighted signals

## Signal Generation Logic

**Confidence Thresholds:**
- High (>75%): Generate LONG/SHORT signal
- Medium (60-75%): CAUTIOUS - consider small position
- Low (<60%): WAIT - stay on sidelines

**Direction Logic:**
- Break Direction UP + High Confidence = LONG
- Break Direction DOWN + High Confidence = SHORT
- Next Channel Direction = Expected post-break trend

**Risk Indicators:**
- Position >0.8: Near upper boundary (potential reversal)
- Position <0.2: Near lower boundary (potential bounce)
- Invalid channels: No clear structure (higher risk)
- Low bounces: Weak channel (less predictable)

## Advanced Features

### Event Integration

The dashboard shows upcoming high-impact events:
- TSLA earnings (±3 days highlighted in red)
- Delivery reports
- FOMC meetings
- CPI/NFP releases
- Quad witching

Use this to adjust position sizing or avoid trading before major events.

### Multi-Timeframe Analysis

Key patterns to look for:
1. **Alignment:** All timeframes trending same direction (higher confidence)
2. **Divergence:** Shorter TFs reversing while longer TFs hold (potential early signal)
3. **Boundary Proximity:** Price near longer TF boundary (potential break/bounce)

### Attention Weights (Model-based)

When using the trained model, attention weights show which timeframes the model considers most important for current prediction. High attention on daily/weekly suggests macro trend dominance.

## Export & Logging

**CSV Export (`--export results/`):**
```
results/
├── prediction_20241231_153000.csv   # Model predictions
└── channels_20241231_153000.csv     # Channel status all TFs
```

**prediction_*.csv:**
```
timestamp,duration_mean,duration_std,break_direction,break_up_prob,next_direction,confidence,tsla_price,spy_price,vix
2024-12-31 15:30:00,23.4,5.1,1,0.89,2,0.87,345.67,582.34,14.23
```

**channels_*.csv:**
```
timestamp,timeframe,valid,direction,position,width_pct,slope_pct,bounces,cycles,r_squared
2024-12-31 15:30:00,5min,True,2,0.82,3.4,0.12,8,3,0.91
2024-12-31 15:30:00,15min,True,2,0.74,4.1,0.08,6,2,0.87
...
```

## Live Trading Considerations

**DO NOT blindly follow signals.** This is a research tool. For live trading:

1. **Validate signals:** Check actual price action, volume, news
2. **Risk management:** Use stop losses, position sizing
3. **Slippage:** Model predicts bars, not exact prices
4. **Latency:** Data may be delayed (check timestamps)
5. **Market conditions:** Model trained on specific regime
6. **Event risk:** Earnings/FOMC can invalidate channels

## Troubleshooting

**"No data loaded":**
- Check CSV files exist in `data/` directory
- Verify CSV format matches expected schema
- Check date ranges (need recent data)

**"Model prediction failed":**
- Ensure model checkpoint is compatible (v7 architecture)
- Check input feature dimensions (should be 582)
- Verify PyTorch version compatibility

**"Channel detection slow":**
- Reduce `--lookback` to load less data
- Disable history features (already done by default)
- Use SSD for data storage

## Performance

**Terminal Dashboard:**
- Initial load: ~5-10 seconds (90 days data)
- Refresh: ~3-5 seconds
- Memory: ~500MB
- CPU: Low (mostly I/O bound)

**Visual Dashboard:**
- Render time: ~8-12 seconds
- Export: ~2 seconds
- Memory: ~800MB (matplotlib)

## Future Enhancements

Potential additions:
- [ ] Live API integration (Alpaca, IBKR)
- [ ] WebSocket real-time updates
- [ ] Position tracking & P&L
- [ ] Backtesting mode (historical inference)
- [ ] Alert system (SMS/email on high-confidence signals)
- [ ] Multi-asset support (beyond TSLA)
- [ ] Web dashboard (Flask/Streamlit)

## Architecture Notes

The dashboard is stateless - it reloads data on each refresh. For production:
- Cache feature extraction
- Stream only new bars
- Maintain model hidden states
- Use database for historical predictions

## Support

For issues or questions:
1. Check data format compatibility
2. Verify model checkpoint version
3. Review error logs
4. Check v7 module imports

## License

Part of the x6 channel prediction project. For research and educational purposes.
