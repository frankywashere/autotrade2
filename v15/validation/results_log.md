# Validation Results Log

All backtest and forward-return analysis results are recorded here.
Scripts add entries; never delete — append only.

---

## tf_state_backtest.py — IS backtest results
**Script**: `v15/validation/tf_state_backtest.py`
**Default params**: hold=10d, stop=20%, capital=$100K, IS=2015-2024
**Status**: NOT YET RUN with optimal params. Run `tf_state_oos.py` for IS/OOS split.

---

## tf_state_targets.py — Forward-return expectancy (IS 2015-2024)
**Script**: `v15/validation/tf_state_targets.py`
**Run date**: Feb 2026 (full `--detail` run on server, 1265 lines output)
**Capital**: $100K per firing, next-day open entry, no stop

| Signal | n | E[30d] | +10% hit | +5% stop | Notes |
|--------|---|--------|----------|----------|-------|
| E1 wMT+con5 | 5 | $26,308 | 100% | ~20% | Rarest — weekly MT + all 5 TFs near bottom |
| D3 all-5-NB | 8 | $26,308 | 100% | ~20% | All 5 TFs simultaneously near lower bound |
| A2 wMT+1hATB | 18 | $17,549 (E[45d]) | 94% | ~35% | 1h at bottom + weekly turning; best at 45d |
| B4 1hATB+wMT | 18 | $17,549 (E[45d]) | 94% | ~35% | Same conditions as A2 |
| A5 wMT+4NB | 12 | $12,303 | 92% | ~30% | 4+ TFs near bottom with weekly turning |
| D2 4NB | 20 | $8,000 | 90% | ~35% | 4 TFs near bottom, no specific TF required |
| A6 wMT+dMT | 28 | $7,500 | 89% | ~35% | Weekly + daily both turning |
| A1 wMT | 45 | $4,500 | 82% | ~45% | Broadest — weekly alone turning |

**Key universal finding**:
- Expectancy peaks at 30-45 days across ALL 36 signals
- Tight stops (-5%) triggered on 65-92% of entries — kills the edge
- Optimal: 30-45d time-based exit, -10% or wider stop

---

## tf_state_oos.py — IS/OOS + Walk-forward validation
**Script**: `v15/validation/tf_state_oos.py`
**Status**: PENDING — not yet run
**Params**: hold=30d, stop=20%, capital=$100K

Run command:
```
cd C:\AI\x14 && python -m v15.validation.tf_state_oos --tsla data/TSLAMin.txt >> v15/validation/tf_state_oos_results.log 2>&1
```

---

## rsi_bottom_targets.py — RSI percentile forward-return analysis
**Script**: `v15/validation/rsi_bottom_targets.py`
**Status**: PENDING — not yet run
**Params**: RSI(14) Wilder, percentile window=252 bars, IS=2015-2024

Tests:
- Per-TF: 5 TFs × 8 thresholds (p10/p15/p20/p25 + abs25/30/35/40)
- Multi-TF: 3+/4+/5 TFs simultaneously at p10/p15/p20/p25
- Weekly combos: weekly RSI bottom + lower TF confirmation (~16 combos)

Run command:
```
cd C:\AI\x14 && python -m v15.validation.rsi_bottom_targets --tsla data/TSLAMin.txt >> v15/validation/rsi_bottom_results.log 2>&1
cd C:\AI\x14 && python -m v15.validation.rsi_bottom_targets --tsla data/TSLAMin.txt --end-year 2025 >> v15/validation/rsi_bottom_results_2025.log 2>&1
```

---

## Phase 11R — RSI smooth/divergence/hook (S1195-S1200)
**Script**: `v15/validation/swing_backtest.py`
**Status**: RUNNING on server (launched Feb 26, 2026)
**Signals**: S1195-S1200 (RSI smooth, RSI div, RSI hook variants on S1041 base)

Run command used:
```
python -m v15.validation.swing_backtest --signal S119 --end-year 2025
```

Results: PENDING

---

## OpenEvolve Runs — Status as of Feb 26, 2026

### 3B2 (WR≥90%)
- Port: 5555, seed: 24t/100%WR/$1,588K
- Status: Running, ~iter 22+
- Issue: oscillating at WR cliff (27t/92.6% ↔ 28t/89.3%)
- Best score seen: 25 (seed)

### 3B3 (WR≥85%)
- Port: 5556, seed: 24t/100%WR/$1,588K
- Status: Running
- Iter 1 result: 30t/86.7%/score=26 — already beat seed
- Goal: explore 85-90% WR range invisible to 3B2

### 3D (5-min intraday)
- Port: 5558
- Status: Running with Claude Haiku (previous Codex run failed on token limits)
- First clean run with Haiku

---

## walk_forward_filters.py — 5-min filter cascade OOS
**Script**: `v15/validation/walk_forward_filters.py`
**Status**: COMPLETE (prior session)
**Result**: NO filter beats baseline OOS — sq50_bp baseline wins all 6 windows
- sq_gate=0.50: best IS but NOT best OOS
- break_pred, swing_regime: negative OOS
- Conclusion: 5-min system is self-contained; filters don't add OOS value

---

## combined_backtest.py — 5-min filter grid
**Script**: `v15/validation/combined_backtest.py`
**Status**: COMPLETE
**Result**: baseline (no filters) wins OOS; see walk_forward_filters.py

---

## swing_backtest.py — S1041 IS/OOS
**Script**: `v15/validation/swing_backtest.py`
**Run**: `--signal S1041 --end-year 2025`
**Result**: n=23 IS (2015-2024), 100% WR, $2,037K + OOS 2025: 3 trades $632K
**Status**: COMPLETE — S1041 confirmed champion

---

## Arch418 (c9) Walk-forward
**Status**: COMPLETE — 6/6 windows profitable, OOS/IS=1.85x
**Result**: $16,552,438 IS 2015-2025, p-value excellent

---

## Signal Quality Model (c10)
**Status**: COMPLETE — AUC=0.816, +21.5% on 2025 OOS vs baseline
**File**: `v15/validation/signal_quality_model_c10_arch2.pkl`

---
*Append new results below this line*
