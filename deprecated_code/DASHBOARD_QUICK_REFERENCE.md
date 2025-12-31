# Dashboard Quick Reference Card

## Signal Interpretation (30-Second Guide)

### Trading Signals
```
LONG     = Go long (high confidence bullish)
SHORT    = Go short (high confidence bearish)
CAUTIOUS = Small position (medium confidence)
WAIT     = Stay flat (low confidence)
```

### Confidence Levels
```
> 75%  ⭐  = HIGH    → Full position size
60-75% ○   = MEDIUM  → Half position size
< 60%  △   = LOW     → No position
```

### Position in Channel
```
> 0.80 = Near upper boundary (potential reversal/breakout)
0.60-0.80 = Upper zone (bullish momentum)
0.40-0.60 = Mid-channel (neutral)
0.20-0.40 = Lower zone (bearish momentum)
< 0.20 = Near lower boundary (potential bounce/breakdown)
```

### Channel Validity
```
✓ Valid   = Has bounces/cycles (reliable)
✗ Invalid = No bounces (unreliable, ignore)
```

### RSI Quick Guide
```
> 70 = Overbought (potential reversal down)
50-70 = Bullish but not extreme
30-50 = Bearish but not extreme
< 30 = Oversold (potential reversal up)
```

## Decision Matrix

| Confidence | Position | Direction | Action | Stop Loss |
|------------|----------|-----------|---------|-----------|
| >75% | >0.80 | BULL | LONG (tight stop) | Below 1h lower |
| >75% | 0.40-0.60 | BULL | LONG | Below 4h lower |
| >75% | <0.20 | BEAR | SHORT (tight stop) | Above 1h upper |
| 60-75% | Any | BULL | Small LONG | Below daily lower |
| <60% | Any | Any | WAIT | N/A |

## Multi-Timeframe Alignment

### Strong Signal (Act)
- 7+ timeframes same direction
- Shorter TFs confirming longer TFs
- High confidence (>75%)
- Valid channels across scales

### Weak Signal (Wait)
- Timeframes conflicting
- Invalid channels
- Low confidence (<60%)
- Near major events

### Reversal Signal (Caution)
- Shorter TFs reversing
- Longer TFs still trending
- Positions near boundaries
- Could be early reversal or false signal

## Event Risk Management

| Days to Event | Action |
|---------------|--------|
| T-14 to T-7 | Normal trading |
| T-7 to T-3 | Reduce position 25% |
| T-3 to T-1 | Reduce position 50% |
| T-1 to T+1 | Exit or minimal exposure |
| T+1 to T+3 | Resume gradually |

## Channel Metrics

### Bounce Count
```
>10 = Very reliable channel
5-10 = Moderately reliable
<5 = Weak channel (less predictable)
```

### Complete Cycles
```
>3 = Strong pattern
1-3 = Emerging pattern
0 = No confirmed pattern (invalid)
```

### R-Squared (Fit Quality)
```
>0.90 = Excellent fit
0.80-0.90 = Good fit
<0.80 = Poor fit (noisy)
```

### Width %
```
>5% = Wide channel (high volatility)
3-5% = Normal channel
<3% = Narrow channel (compression, breakout likely)
```

## Model Predictions

### Duration
```
Mean ± Std
23 ± 5 bars = Expect break in 18-28 bars
             (1h TF = 18-28 hours)
High std = Uncertain timing
Low std = Confident timing
```

### Break Direction
```
UP (>80%) = Very likely break upward
UP (60-80%) = Likely break upward
UP (<60%) = Uncertain direction
```

### Next Channel Direction
```
BULL after BULL = Continuation (common)
BEAR after BULL = Reversal (less common)
SIDE after BULL = Consolidation
```

## Common Patterns

### Pattern 1: Nested Bullish Channels
```
All TFs BULL + High positions + High confidence
→ Strong uptrend, ride until confidence drops
```

### Pattern 2: Divergence
```
Long TFs BULL + Short TFs BEAR
→ Potential reversal starting, wait for confirmation
```

### Pattern 3: Compression
```
Narrow width + Low bounces + Mid position
→ Breakout imminent, watch for direction signal
```

### Pattern 4: Channel Walk
```
Position consistently >0.80 or <0.20
→ Strong trend, channel boundaries walking
→ Continue with trend until break
```

## Risk Checks (Before Trading)

### Pre-Trade Checklist
- [ ] Confidence >75% for full position
- [ ] Channel valid (has bounces)
- [ ] No major events in T-3 days
- [ ] Position not extreme (0.2-0.8 safer)
- [ ] Multi-TF alignment confirmed
- [ ] Stop loss level identified
- [ ] Position size appropriate for risk

### Red Flags (Don't Trade)
- Confidence <60%
- Invalid channels
- Conflicting timeframes
- Major event T-1 day
- Extreme RSI (>80 or <20)
- VIX spiking (>25)

## Dashboard Symbols

### Icons
```
✓ = Valid/Good
✗ = Invalid/Bad
⭐ = Highest confidence
○ = Medium confidence
△ = Low confidence
⚠ = Warning/High risk event
↑ = Upward/Bullish
↓ = Downward/Bearish
↔ = Sideways/Neutral
```

### Colors (Terminal)
```
Green = Bullish/Good/Long
Red = Bearish/Bad/Short
Yellow = Neutral/Caution/Wait
Cyan = Info/Headers
White/Gray = Data
```

## Typical Workflow

### 1. Quick Scan (10 seconds)
- Check main signal (LONG/SHORT/WAIT)
- Note confidence level
- Glance at upcoming events

### 2. Validation (30 seconds)
- Review multi-TF alignment
- Check channel validity
- Verify positions not extreme

### 3. Decision (20 seconds)
- If confidence >75% + alignment: Trade
- If 60-75%: Small position
- If <60%: Wait

### 4. Execution (Variable)
- Set entry at current price
- Calculate stop from channel boundaries
- Set position size based on confidence
- Monitor for signal changes

## Performance Tracking

### Daily Log Format
```
Date: 2024-12-31
Time: 15:30:00
Signal: LONG
Confidence: 89%
Entry: $345.67
Duration Pred: 23 bars
Actual Break: [Fill when occurs]
Outcome: [Win/Loss/Breakeven]
Notes: [What worked/didn't]
```

### Weekly Review Questions
1. What was average confidence on winning trades?
2. Did high confidence signals perform better?
3. Were event-risk exits profitable?
4. Which timeframes were most predictive?
5. What patterns repeated?

## Troubleshooting

### "All signals show WAIT"
→ Low volatility/unclear structure
→ Normal during consolidation
→ Wait for clarity

### "Signals contradicting price action"
→ Check data timestamp (is it delayed?)
→ Verify model is appropriate for current regime
→ Consider manual override

### "Confidence fluctuating rapidly"
→ Market in transition
→ Use longer refresh interval
→ Wait for stability

### "Different TFs showing opposite signals"
→ Normal during trend changes
→ Focus on longer TFs for direction
→ Reduce position size

## Emergency Procedures

### If Trade Goes Against You
1. Check if signal changed to WAIT/opposite
2. Evaluate stop loss hit
3. Don't average down without confirmation
4. Exit if confidence drops below 40%

### If Major Event Announced Mid-Trade
1. Immediately assess impact
2. Consider reducing position 50%
3. Tighten stops
4. Re-evaluate after event

### If Dashboard Fails/Crashes
1. Don't panic trade
2. Use manual channel analysis
3. Conservative position sizing
4. Wait for dashboard recovery before new positions

## Advanced Tips

### Tip 1: Event-Driven Edge
Watch for pre-event drift matching historical patterns
→ If positive drift INTO earnings + beat expected
→ Strong signal for continuation

### Tip 2: Multi-Asset Confirmation
Check SPY alignment
→ TSLA BULL + SPY BULL = Higher confidence
→ TSLA BULL + SPY BEAR = Lower confidence (divergence risk)

### Tip 3: VIX Regime Adaptation
VIX <15: Trade normally
VIX 15-20: Reduce position 25%
VIX 20-30: Reduce position 50%
VIX >30: Only highest confidence trades

### Tip 4: Time of Day Patterns
9:30-10:30: Higher volatility (wider stops)
10:30-15:30: More reliable signals
15:30-16:00: Closing squeeze (exit before)

### Tip 5: Attention Weights (If Model Loaded)
High attention on longer TFs → Macro trend dominant
High attention on shorter TFs → Tactical trading
Balanced attention → Normal market

## Summary Card (Wallet Size)

```
╔═══════════════════════════════════════════╗
║  DASHBOARD QUICK REFERENCE               ║
╠═══════════════════════════════════════════╣
║  SIGNALS                                 ║
║  >75% + LONG → Full Position             ║
║  60-75% → Half Position                  ║
║  <60% → WAIT                             ║
║                                          ║
║  POSITION                                ║
║  >0.80 → Near upper (caution)            ║
║  0.40-0.60 → Mid-channel (safe)          ║
║  <0.20 → Near lower (caution)            ║
║                                          ║
║  EVENTS                                  ║
║  T-3 → Reduce 50%                        ║
║  T-1 → Exit or minimal                   ║
║                                          ║
║  VALIDATION                              ║
║  ✓ Valid channels                        ║
║  ✓ Multi-TF alignment                    ║
║  ✓ High confidence                       ║
║  ✓ No near events                        ║
║                                          ║
║  RED FLAGS                               ║
║  ✗ Confidence <60%                       ║
║  ✗ Invalid channels                      ║
║  ✗ Conflicting TFs                       ║
║  ✗ Event T-1 day                         ║
╚═══════════════════════════════════════════╝
```

Print this section for desk reference!

## Help & Support

For detailed explanations, see:
- DASHBOARD_README.md (full documentation)
- DASHBOARD_EXAMPLE_OUTPUT.md (detailed examples)
- v7/docs/ (technical specifications)

Dashboard version: 1.0
Last updated: 2024-12-31
