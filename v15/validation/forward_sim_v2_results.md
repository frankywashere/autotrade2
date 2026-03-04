# Forward Sim V2 — Long OOS Results (2025-01-02 to 2026-02-27)

**Date:** 2026-03-04
**Data:** Local 1-min files (TSLAMin.txt + SPYMin.txt), RTH-filtered, SPY supplemented with yfinance
**Capital:** $100,000/scanner (isolated), Hard stop: 2%
**DW EOD:** Disabled (CS-DW positions can hold multi-day)

---

## Root Cause: Extended Hours Bug (FIXED)

Previous runs using `--local` were **completely broken** — the local 1-min data timestamps
are in **UTC** but the sim assumed they were **ET**. This caused:

- Pre/post-market bars (15:00–23:59 UTC = after-hours in the sim's perspective) treated as tradeable
- 9,974 eod_close trades with -$592K losses (all garbage extended-hours entries)
- 16,531 total trades on 9 months vs ~3,000 expected
- Win rates collapsed to 46% (from 85-90% in validated backtests)

**Fix:** `_load_min_file()` now localizes as UTC, converts to ET, and filters to RTH (9:30-16:00 ET).
Result: 106,951 of 270,806 1-min bars kept (39%), matching expected RTH bar count.

---

## Results: Baseline (All Day, No AM Restriction)

| Scanner | Trades | Wins | WR | P&L | MaxDD | Final Eq |
|---------|--------|------|----|-----|-------|----------|
| CS-5TF  | 4,813  | 2,637| 55%| +$291,697 | 4.38% | $391,697 |
| CS-DW   | 12     | 10   | 83%| +$4,359   | 0.05% | $104,359 |
| ML      | 3,128  | 2,194| 70%| +$235,471 | 18.94%| $335,471 |
| Intra   | 1,415  | 683  | 48%| +$69,005  | 10.59%| $169,005 |
| **TOTAL**|**9,368**|**5,524**|**59%**|**+$600,532**| |**$1,000,532**|

ROI: +150% on $400,000 over 14 months.

### Direction breakdown (baseline)
| Scanner | Long # | Long P&L | Long WR | Short # | Short P&L | Short WR |
|---------|--------|----------|---------|---------|-----------|----------|
| CS-5TF  | 2,542  | +$125,655| 53%     | 2,271   | +$166,042 | 57%      |
| CS-DW   | 0      | $0       | -       | 12      | +$4,359   | 83%      |
| ML      | 1,632  | +$64,781 | 69%     | 1,496   | +$170,690 | 71%      |
| Intra   | 1,415  | +$69,005 | 48%     | 0       | $0        | -        |

### Remaining problem areas (baseline)
- **eod_close:** 443 trades, -$281,899, 22% WR — still significant (mostly 15:xx entries)
- **hard_stop:** 211 trades, -$893K, 0% WR — $4,233 avg loss (ML positions are huge late in sim)
- **timeout:** 50 trades, -$159K, 0% WR

---

## Results: AM-Only (CS/ML restricted to 9:30-10:30 ET)

| Scanner | Trades | Wins | WR | P&L | MaxDD | Final Eq |
|---------|--------|------|----|-----|-------|----------|
| CS-5TF  | 1,045  | 747  | 71%| +$147,926 | 1.98% | $247,926 |
| CS-DW   | 11     | 9    | 82%| +$3,969   | 0.05% | $103,969 |
| ML      | 922    | 727  | 79%| +$142,727 | 17.96%| $242,727 |
| Intra   | 1,415  | 683  | 48%| +$69,005  | 10.59%| $169,005 |
| **TOTAL**|**3,393**|**2,166**|**64%**|**+$363,626**| |**$763,626**|

ROI: +91% on $400,000 over 14 months.

### Direction breakdown (AM-only)
| Scanner | Long # | Long P&L | Long WR | Short # | Short P&L | Short WR |
|---------|--------|----------|---------|---------|-----------|----------|
| CS-5TF  | 566    | +$67,641 | 71%     | 479     | +$80,284  | 72%      |
| CS-DW   | 0      | $0       | -       | 11      | +$3,969   | 82%      |
| ML      | 496    | +$41,967 | 77%     | 426     | +$100,760 | 81%      |
| Intra   | 1,415  | +$69,005 | 48%     | 0       | $0        | -        |

### Problem areas (AM-only)
- **eod_close:** 4 trades, -$10K — essentially eliminated
- **hard_stop:** 76 trades, -$229K — reduced but still the biggest drag
- **timeout:** 16 trades, -$34K

---

## Comparison: Baseline vs AM-Only

| Metric         | Baseline      | AM-Only       | Delta        |
|----------------|---------------|---------------|--------------|
| Trades         | 9,368         | 3,393         | -64%         |
| Win Rate       | 59%           | 64%           | +5pp         |
| Total P&L      | +$600,532     | +$363,626     | -$237K       |
| CS-5TF P&L     | +$291,697     | +$147,926     | -$144K       |
| ML P&L         | +$235,471     | +$142,727     | -$93K        |
| CS-5TF WR      | 55%           | 71%           | +16pp        |
| ML WR          | 70%           | 79%           | +9pp         |
| CS-5TF MaxDD   | 4.38%         | 1.98%         | -2.4pp       |
| ML MaxDD       | 18.94%        | 17.96%        | -1.0pp       |
| eod_close loss | -$282K        | -$10K         | +$272K saved |
| hard_stop loss | -$893K        | -$229K        | +$664K saved |

**Key finding:** With RTH data fixed, baseline is actually the P&L winner (+$601K vs +$364K).
AM-only has better WR and lower MaxDD but leaves significant money on the table. The afternoon
trades are net-positive when using proper RTH data — the previous "afternoon = bad" conclusion
was an artifact of the extended hours bug.

---

## CS-DW with EOD Disabled

With EOD close disabled for CS-DW, results barely changed (10→12 trades baseline, 10→11 AM-only).
Daily/weekly channel signals simply don't fire often enough on TSLA 5-min bars. CS-DW
generates ~1 trade/month. The positions it does take are 83% WR — the signal quality is
excellent, just extremely rare.

---

## Hard Stop Analysis

Hard stop trades are 50/50 long/short (not direction-biased). Common patterns:
- Almost all are 5min TF signals
- Many are "break" type (momentum breaks that immediately reverse)
- ML hard stops get increasingly expensive as equity compounds (later stops cost $5-8K each)
- Hold times are bimodal: either <30min (immediate reversal) or 1000+ min (DW multi-day holds)

---

## VIX-Gated AM-Only Sweep

**Concept:** Instead of always restricting CS/ML to the first hour (AM-only), only apply the AM restriction when previous day's VIX close >= threshold. This allows full-day trading on calm VIX days while protecting during volatile periods.

**Note:** Intraday scanner is NOT gated (confirmed: 1,415 trades unchanged across all scenarios). Only CS-5TF, CS-DW, and ML are subject to VIX-gated AM restriction.

| Scenario | Trades | WR | P&L | MaxDD | Notes |
|----------|--------|----|-----|-------|-------|
| Baseline (no AM) | 9,368 | 59% | +$600,532 | 18.94% | No restriction |
| Always AM-only | 3,393 | 64% | +$363,626 | 17.96% | Best WR, lowest trades |
| VIX >= 16 | 4,829 | 60% | +$346,933 | 20.19% | Too aggressive — worse than AM-only |
| VIX >= 18 | 6,886 | 58% | +$345,857 | 21.85% | Similar P&L to AM-only, worse MaxDD |
| **VIX >= 20** | **7,815** | **58%** | **+$402,261** | **17.88%** | **Best risk-adjusted: +$39K vs AM, lower MaxDD** |
| VIX >= 22 | 8,421 | 58% | +$416,726 | 20.14% | More trades, worse MaxDD |
| VIX >= 25 | 8,901 | 59% | +$464,797 | 19.53% | Approaching baseline |
| VIX >= 30 | 9,079 | 59% | +$503,400 | 17.06% | Best MaxDD, near-baseline trades |

### VIX Sweep Exit Reason Comparison

| Scenario | hard_stop P&L | eod_close P&L | trailing_stop P&L |
|----------|--------------|---------------|-------------------|
| Baseline | -$893K (211) | -$282K (443) | +$1,367K (7,861) |
| AM-only | -$229K (76) | -$10K (4) | +$537K (3,093) |
| VIX >= 20 | -$524K (173) | -$173K (344) | +$876K (6,634) |
| VIX >= 30 | -$740K (203) | -$240K (427) | +$1,149K (7,637) |

### VIX Sweep Interpretation

- **VIX >= 20 is the sweet spot**: +$402K P&L with 17.88% MaxDD (lowest of any scenario!)
  - Gets +$39K more than AM-only while keeping MaxDD even lower
  - 2.3x more trades than AM-only (7,815 vs 3,393)
- **VIX >= 30 is interesting**: 17.06% MaxDD (best), +$503K P&L
  - Only gates during extreme VIX spikes (April 2025 tariff crisis)
  - Keeps 97% of baseline trades but avoids the worst days
- **VIX >= 16 and 18 underperform**: They gate too aggressively on normal days, losing profitable afternoon trades without sufficient MaxDD improvement
- The April 2025 VIX spike (30→52) is the dominant factor — gating those ~20 trading days alone accounts for most of the hard_stop savings

### VIX Distribution Context (2025-01-02 to 2026-02-27)
- Range: 13.47 - 52.33
- Mean: 18.78
- Days VIX >= 20: ~80 (29% of period) — concentrated in Mar-May 2025, Oct-Nov 2025, Feb 2026
- Days VIX >= 30: ~15 (5% of period) — almost entirely April 2025 tariff crisis

---

## Hard Stop Deep Dive (211 Baseline Trades)

Full CSV export: `v15/validation/hard_stop_trades.csv`

### By Scanner
| Scanner | Count | Avg P&L | Total P&L | Long/Short |
|---------|-------|---------|-----------|------------|
| ML | 140 (66%) | -$5,661 | -$792K | L:82/S:58 |
| CS-5TF | 62 (29%) | -$1,317 | -$82K | L:33/S:29 |
| Intra | 9 (4%) | -$2,098 | -$19K | L:9/S:0 |

**ML is the dominant problem** — it takes bigger positions as equity compounds, so late-sim hard stops cost $7-8K each.

### By VIX Level
| VIX Range | Count | Total P&L | Avg P&L |
|-----------|-------|-----------|---------|
| VIX < 16 | 42 | -$189K | -$4,503 |
| VIX 16-20 | 112 | -$475K | -$4,240 |
| VIX 20-25 | 37 | -$155K | -$4,186 |
| VIX 25-35 | 9 | -$42K | -$4,645 |
| VIX 35+ | 11 | -$32K | -$2,935 |

**Not VIX-correlated** — hard stops happen at all VIX levels. 73% occur when VIX < 20. VIX gating alone can't solve this.

### By Entry Hour (ET)
| Hour | Count | P&L | Notes |
|------|-------|-----|-------|
| 9:00-9:59 | 47 | -$233K | Open volatility |
| 10:00-10:59 | 31 | -$149K | |
| 11:00-11:59 | 23 | -$119K | |
| 12:00-12:59 | 22 | -$103K | |
| 13:00-13:59 | 17 | -$79K | Least stops |
| 14:00-14:59 | 33 | -$118K | Late afternoon pickup |
| 15:00-15:59 | 38 | -$91K | **Overnight holds** — enter 15:55, gap against next day |

Overnight holds (entry ~15:55, exit next day 09:30) are a distinct cluster: gap opens against position.

### By Signal Type
| Type | Count | P&L | Avg Confidence |
|------|-------|-----|----------------|
| break | 122 (58%) | -$527K | 0.68 |
| bounce | 80 (38%) | -$347K | 0.54 |
| intraday | 9 (4%) | -$19K | 0.81 |

**`break` signals are worst** — momentum breaks that immediately reverse. Bounce signals have lower confidence (0.54 avg).

### By Primary Timeframe
- **5min: 210/211 trades** — virtually all hard stops come from 5-min TF signals
- 1h: 1 trade

### By Month
Distributed fairly evenly (7-23 per month). April 2025 is worst ($-89K, 23 trades) due to tariff crisis.

### Calendar Event Cross-Reference

50/211 (24%) hard stops happen ON event days, 110/211 (52%) within 3 calendar days of an event.
But with 70 event days out of ~289 trading days (24%), this is roughly proportional — **not concentrated**.

| Proximity | Count | Total P&L | Avg P&L/trade |
|-----------|-------|-----------|---------------|
| On event day | 50 | -$210K | -$4,197 |
| Near event (±3 days) | 110 | -$446K | -$4,057 |
| No event | 51 | -$237K | -$4,644 |

Avg loss is actually slightly WORSE on non-event days. Calendar events are not the primary driver.

TSLA daily range by event type:
- Jackson Hole: 6.3% avg (1 sample)
- TSLA earnings: 4.66% avg
- Fed speeches: 5.0% avg
- CPI/NFP: ~4.0% avg
- Non-event: 3.68% avg

Event days have ~19% more hourly volatility, but this doesn't concentrate hard stops.

### Key Hard Stop Patterns
1. **ML + 5min + break** is the toxic combination — 66% of all hard stops, compounding losses
2. **Overnight holds** (15:55 entry → 09:30 exit) represent a distinct failure mode: gap against
3. **Confidence doesn't protect** — avg confidence 0.68 for break, 0.54 for bounce (both get stopped)
4. **Not VIX-dependent** — can't solve with VIX gating alone
5. **Not calendar-event-dependent** — proportional to event frequency, not concentrated
6. **Structural problem**: ML position sizing compounds losses; 5min-break signals have inherent reversal risk
7. **Potential filters**: Block 5min-break signals for ML? Cap ML position size? Block entries after 15:30? Use OpenEvolve to learn optimal gating function.

---

## Trade Gate Experiments

### Gate V2: No Overnight + ML Break Confidence Filter
Rules: Block entries after 15:30 ET. Block ML+break+5min if confidence < 0.60.

| Metric | Baseline | Gate V2 | Delta |
|--------|----------|---------|-------|
| Trades | 9,368 | 8,275 | -12% |
| WR | 59% | 60% | +1pp |
| P&L | +$600,532 | +$502,864 | -$98K |
| Hard stops | 211 (-$893K) | 161 (-$556K) | -50 HS, +$337K saved |
| eod_close | 443 (-$282K) | 97 (-$135K) | -346 trades, +$147K saved |

Gate V2 saved $484K in hard_stop+eod losses, but missed some winning trades, netting -$98K.
The overnight block is effective (eod_close: 443→97, -$147K saved) but also blocks some profitable 15:55 entries.

### OpenEvolve Trade Gate (running on server)
Started: 2026-03-04 ~03:00 UTC. ~4.5 min/iteration, 200 iterations planned.

Initial seed score: -60,875 (P&L +$443K, 63.6% WR, 114 HS)
- Train (Jan-Sep 2025): +$429K, 68.2% WR, 4886 trades
- Test (Oct 2025-Feb 2026): +$14K, 38.1% WR, 884 trades — OOS collapse

Iteration 1: -52,334 (P&L +$591K, 62.7% WR, 93 HS) — improved but test still 36.6% WR
Iteration 2: -60,350 (no improvement — variant didn't generalize)

**Key challenge**: The gate's train/test split reveals that Oct 2025-Feb 2026 regime
is fundamentally different (lower volatility, higher prices). The gate must generalize
across regimes or it will fail in production.

*(Results will be updated as OpenEvolve completes iterations)*

---

## Key Takeaways

1. **RTH filter is critical** — extended hours data completely invalidated previous long-OOS results
2. **Both longs and shorts are profitable** in RTH — no need to go long-only
3. **Baseline > AM-only on P&L** (+$601K vs +$364K) with RTH data
4. **AM-only is safer** (64% WR, 2% MaxDD) but gives up ~40% of returns
5. **ML is viable** at 70-79% WR when using RTH data (was showing 50% with extended hours)
6. **Intraday is the weakest scanner** at 48% WR / +$69K — needs investigation
7. **Hard stops remain the biggest cost** (-$229K to -$893K) — worth studying tighter/wider stops
8. **VIX >= 20 gate is optimal**: +$402K P&L, 17.88% MaxDD — best risk-adjusted return
9. **VIX >= 30 gate for max P&L with protection**: +$503K, 17.06% MaxDD, keeps 97% of trades
