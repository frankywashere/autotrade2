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

**RESULTS** (run Feb 26, 2026 — hold=30d, stop=20%, capital=$100K, IS=2015-2023):

| Signal | IS n | IS WR | IS P&L | OOS24 n | OOS24 P&L | OOS25 n | OOS25 P&L |
|--------|------|-------|--------|---------|-----------|---------|-----------|
| C1 5min_MT | 49 | 57% | $507K | 7 | $58K | 7 | $28K |
| D1 consensus_3+ | 63 | 56% | $471K | 8 | $77K | 8 | $19K |
| C3 5min_MT+1hATB | 28 | 64% | $355K | 3 | $22K | 3 | $13K |
| D5 consensus_4++MT | 31 | 65% | $340K | 4 | $19K | 3 | **-$31K** |
| B1 1h_MT | 55 | 53% | $296K | 8 | $83K | 5 | $41K |
| A2 wMT+1hATB | 21 | 57% | $243K | 4 | $50K | 4 | **-$19K** |
| A9 wMT+4hATB | 17 | 59% | $150K | 4 | $53K | 3 | $3K |
| D3 all-5-NB | 7 | 71% | $187K | 1 | -$4K | 1 | $11K |
| E1 wMT+con5 | 4 | 75% | $89K | 1 | -$4K | 1 | $11K |
| **TOTAL** | | | **$6.8M** | | **$677K** | | **-$32K** |

Walk-forward (IS=5yr, OOS=1yr, 6 windows):
- 6/6 windows OOS positive
- Best IS signal: C1 (5min_MT) wins 4/6 windows
- OOS/IS ratio: 0.09x–0.41x — much weaker than S1041 walk-forward
- Total WF OOS: $638K on $4.4M IS = 0.15x ratio

**Key findings**:
- C1 (5min_MT alone) is the most robust OOS signal — simple, high-frequency
- D3/E1 have only 1 OOS trade each — too few to conclude
- 2025 OOS is flat/negative overall ($-32K across all 36 signals)
- Channel-position signals (NB/ATB) are weaker OOS than momentum-turn signals (MT)
- Highest IS performers (D6, B6) degrade significantly OOS

---

## rsi_bottom_targets.py — RSI percentile forward-return analysis
**Script**: `v15/validation/rsi_bottom_targets.py`
**Run date**: Feb 26, 2026 — IS=2015-2025 (all years), capital=$100K

**Top signals by +10% hit rate (n≥3, E[30d] sorted):**

| Signal | n | +10% hit | -5% hit | E[10d] | E[30d] | Notes |
|--------|---|----------|---------|--------|--------|-------|
| weekly RSI<30 | 13 | **100%** | 69% | $17.9K | **$75.3K** | Best single-TF signal |
| wkly_RSI<30 + daily_RSI<35 | 9 | 100% | 78% | $12.4K | **$78.8K** | Narrower, even better |
| 4+TFs RSI<30 | 4 | 100% | 75% | $13.6K | **$65.3K** | Rare but exceptional |
| 5+TFs RSI<35 | 4 | 100% | 100% | $0.3K | **$68.3K** | Ultra-rare, 100%+10%, stops always hit too |
| wkly_RSI<35 + daily_RSI<40 | 24 | 100% | 67% | $9.9K | **$54.1K** | More frequent version |
| wkly_p10 + 4h_p15 | 24 | 96% | 79% | $6.1K | $17K | Percentile version |
| daily RSI<25 | 26 | 88% | 77% | $7.8K | $28.5K | Daily absolute |
| weekly RSI<35 | 37 | 100% | 54% | $14.2K | **$43.3K** | Broader weekly extreme |
| daily RSI<30 | 93 | 82% | 86% | $3.9K | $14.1K | Common enough to act on |

**Universal pattern confirmed again**:
- Stops (-5%) triggered 54-100% of all RSI bottom entries
- E[10d] is low or negative; E[30d] always 3-5× better than E[10d]
- **Zero cases where neither +10% nor -10% hits in 60d** (race col always 100%)
- Weekly RSI < 30 → 100% hit +10% AND +20% within 60d — no exceptions in 10yr

**Best confluence signal**: `wkly_RSI<30 + daily_RSI<35` — n=9, E[30d]=$78,773, 100% hit all profit targets
- Outperforms channel-position signals (D3/E1 at $26K) by 3×
- Has enough frequency (9 over 10yr = ~1/yr) to be tradeable

---

## Phase 11R — RSI smooth/divergence/hook (S1195-S1200)
**Script**: `v15/validation/swing_backtest.py`
**Run**: Feb 26, 2026 — `--signal S119 --end-year 2025` (IS 2015-2024 + OOS 2025)
**Status**: COMPLETE

| Signal | n | WR | P&L IS | avg/trade | Yrs |
|--------|---|----|--------|-----------|-----|
| S1196_rsi_smooth | 8 | **100%** | $832K | $104K | 5/5 |
| S1199_rsi_hook_or_div | 9 | **100%** | $614K | $68K | 6/6 |
| S1197_rsi_div | 7 | **100%** | $372K | $53K | 4/4 |
| S1195_rsi_hook | 2 | **100%** | $242K | $121K | 2/2 |
| S1198_rsi_smooth_div | 4 | **100%** | $205K | $51K | 3/3 |
| S1200_rsi_smooth_bot | 4 | 75% | $365K | $91K | 2/3 |

**Key finding**: All are subsets of S1041 (smaller n, higher avg/trade).
- S1196 (smoothed RSI<40) best: $104K avg vs S1041's $89K avg, but only 5/9 years.
- No Phase 11R signal beats S1041's total P&L ($2,037K IS + $632K OOS).
- **S1041 remains champion** — RSI refinements select best subset but reduce frequency.

**Bugs fixed**: _s1041_core had look-ahead bias (len(tw) instead of searchsorted); Phase 11R+11S code placed after SIGNALS list (NameError).

---

## Phase 11S — Weekly RSI extremes (S1200-S1210)
**Script**: `v15/validation/swing_backtest.py`
**Run**: Feb 26, 2026 — `--signal S120,S1210 --end-year 2025`
**Status**: COMPLETE

| Signal | n | WR | P&L IS | avg/trade | Yrs | Description |
|--------|---|----|--------|-----------|-----|-------------|
| S1207_s1041core_wkly40 | 8 | **100%** | $768K | $96K | 6/6 | S1041_core + wkly RSI<40 |
| S1202_wkly_rsi35 | 8 | 62% | $693K | $87K | 4/5 | wkly RSI<35 standalone |
| S1201_wkly_rsi30 | 4 | **100%** | $335K | $84K | 4/4 | wkly RSI<30 standalone |
| S1204_wkly30_daily40 | 4 | **100%** | $266K | $67K | 4/4 | wkly30 + daily40 |
| S1203_wkly30_daily35 | 3 | **100%** | $160K | $53K | 3/3 | wkly30 + daily35 |
| S1210_s1041_or_wkly30 | 27 | 93% | $1,942K | $72K | 9/9 | union (S1041 OR wkly30+d35) |

**Key finding**: Weekly RSI signals work AS SUBSETS within S1041's channel structure.
- S1207 (S1041_core + weekly RSI<40): n=8, 100% WR, $96K avg — quality within structure.
- Standalone weekly RSI<30 (S1201): only 4 backtest trades vs 13 in forward-analysis (deduplication of overlapping positions in backtest).
- Union S1210 (n=27, WR=93%, $1,942K) is WORSE than S1041 alone ($2,037K) — adding RSI-only trades degrades quality.
- **S1041 channel structure is the essential quality gate** — RSI confirms it but doesn't replace it.

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
