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

## combined_backtest.py -- 5-min filter grid (ALL 13 configs)
**Script**: `v15/validation/combined_backtest.py`
**Run date**: Feb 26, 2026 -- all 13 configs run in parallel on server
**Status**: COMPLETE

| Config | IS P&L (10yr) | Delta vs baseline | OOS 2025 P&L | OOS trades |
|--------|--------------|-------------------|--------------|------------|
| **baseline** | **$15,772,582** | -- | **$1,557,826** | 1,058 |
| mtf_exhaust | $15,894,706 | +$122,124 (+0.8%) | $1,562,133 | 1,055 |
| bp_only | $15,867,248 | +$94,666 (+0.6%) | $1,555,535 | 1,062 |
| sq50_bp | $15,867,248 | +$94,666 (+0.6%) | $1,555,535 | 1,062 |
| sq50 | $15,772,582 | +$0 | $1,557,826 | 1,058 |
| sq55 | $15,772,582 | +$0 | $1,557,826 | 1,058 |
| swing_only | $15,772,582 | +$0 | $1,557,826 | 1,058 |
| mtf_conflict | $15,772,582 | +$0 | $1,557,826 | 1,058 |
| mtf_full | $15,772,582 | +$0 | $1,557,826 | 1,058 |
| sq50_mtf | $15,772,582 | +$0 | $1,557,826 | 1,058 |
| sq50_swing | $15,772,582 | +$0 | $1,557,826 | 1,058 |
| all_50 | $15,772,582 | +$0 | $1,557,826 | 1,058 |
| all_55 | $15,772,582 | +$0 | $1,557,826 | 1,058 |

**Verdict**: Baseline wins. 10/13 configs are identical to baseline (zero effect).
- mtf_exhaust: marginal IS gain (+0.8%), marginal OOS gain (+$4,307) -- consistent with walk-forward (5/6 wins)
- bp_only: small IS gain (+0.6%), LOSES OOS (-$2,291)
- SQ gate, swing regime, mtf_conflict, mtf_full, all combos: literally zero difference from baseline
- **5-min channel surfer is self-contained. No filter improves it meaningfully.**

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

## medium_tf_backtest.py — 1h primary TF
**Script**: `v15/validation/medium_tf_backtest.py`
**Run**: Feb 26, 2026 — `--tf 1h --years 2015-2024 --oos-year 2025`
**Status**: COMPLETE

**IS 2015-2024:**
| Config | Trades | WR | Total P&L | Avg/trade | Sharpe | Trd/yr |
|--------|--------|----|-----------|-----------|--------|--------|
| baseline | 9,061 | 95.2% | $28,656,476 | $3,163 | 2.19 | 906 |
| mtf_exhaust | 9,059 | 95.2% | $28,852,035 | $3,185 | 2.16 | 906 |
| mtf_conflict | 9,081 | 95.2% | $27,263,785 | $3,002 | 2.14 | 908 |
| mtf_full | 9,079 | 95.2% | $27,322,312 | $3,009 | 2.14 | 908 |

**OOS 2025:** baseline/mtf_exhaust tied — 753 trades, WR=98.4%, **$3,324,459**

**Key findings:**
- Momentum filter: neutral/harmful. Baseline wins OOS.
- MaxDD=5%, AvgHold=1.8 bars (1.8h ≈ same-session exit)

---

## medium_tf_backtest.py — 4h primary TF
**Script**: `v15/validation/medium_tf_backtest.py`
**Run**: Feb 26, 2026 — `--tf 4h --years 2015-2024 --oos-year 2025`
**Status**: COMPLETE

**IS 2015-2024:**
| Config | Trades | WR | Total P&L | Avg/trade | Sharpe | Trd/yr |
|--------|--------|----|-----------|-----------|--------|--------|
| baseline | 2,746 | 97.9% | $23,579,567 | $8,587 | 1.27 | 275 |
| mtf_exhaust | 2,746 | 97.9% | $23,583,372 | $8,588 | 1.27 | 275 |
| mtf_conflict | 2,754 | 97.9% | $23,102,628 | $8,389 | 1.27 | 275 |

**OOS 2025:** 219 trades, WR=100%, **$2,597,819**
- WR=100% verified: all exits via trailing stop (inherently positive); zero hard-stop exits
- Real signal: 67 bounce trades = $2.597M (avg $38.8K); 152 break trades = $746 (noise)

**Key findings:**
- Momentum filter: neutral/harmful. Baseline wins.
- 4h vs 1h: 4h has higher WR (97.9% vs 95.2%) but lower frequency and lower total P&L

---

## medium_tf_walk_forward.py — 1h walk-forward (6 rolling windows)
**Script**: `v15/validation/medium_tf_walk_forward.py`
**Run**: Feb 26, 2026 — `--tf 1h`
**Status**: COMPLETE

| Window | IS P&L (5yr) | IS avg/yr | OOS P&L | OOS/IS_avg | OOS WR | Result |
|--------|--------------|-----------|---------|------------|--------|--------|
| IS 2015-2019 → OOS 2020 | $8.6M | $1.7M | $4.6M | 2.69x | 96.4% | WIN |
| IS 2016-2020 → OOS 2021 | $11.6M | $2.3M | $3.4M | 1.46x | 97.4% | WIN |
| IS 2017-2021 → OOS 2022 | $13.4M | $2.7M | $5.1M | 1.89x | 97.3% | WIN |
| IS 2018-2022 → OOS 2023 | $17.6M | $3.5M | $3.3M | 0.94x | 97.1% | WIN |
| IS 2019-2023 → OOS 2024 | $18.5M | $3.7M | $3.7M | 1.00x | 97.5% | WIN |
| IS 2020-2024 → OOS 2025 | $20.1M | $4.0M | $3.3M | 0.83x | 98.4% | WIN |

- **6/6 OOS positive**
- **Avg OOS/IS ratio: 1.47x** — OOS beats average IS year (system getting stronger over time)
- **Total OOS P&L: $23,422,015**
- Compare to tf_state signals (0.15x OOS/IS) — 1h channel surfer is far more robust

---
*Append new results below this line*

---

## medium_tf_walk_forward.py — 4h Walk-Forward (IS 5yr → OOS 1yr)
**Script**: `v15/validation/medium_tf_walk_forward.py --tf 4h`
**Run date**: Feb 25-26, 2026 (server logs/wf_4h.log)
**6 rolling windows: IS 5yr → OOS 1yr**

| Window | IS 5yr P&L | IS avg/yr | OOS P&L | OOS/IS_avg | OOS WR | OOS Trades | Result |
|--------|-----------|-----------|---------|------------|--------|------------|--------|
| IS 2015-2019 → OOS 2020 | $6,088,543 | $1,217,709 | $4,132,187 | 3.39x | 98.6% | 285 | WIN |
| IS 2016-2020 → OOS 2021 | $9,844,184 | $1,968,837 | $1,407,051 | 0.71x | 98.1% | 269 | WIN |
| IS 2017-2021 → OOS 2022 | $10,164,920 | $2,032,984 | $6,932,321 | 3.41x | 99.0% | 293 | WIN |
| IS 2018-2022 → OOS 2023 | $16,266,632 | $3,253,326 | $1,828,504 | 0.56x | 98.9% | 278 | WIN |
| IS 2019-2023 → OOS 2024 | $16,101,175 | $3,220,235 | $3,190,961 | 0.99x | 99.3% | 289 | WIN |
| IS 2020-2024 → OOS 2025 | $17,491,024 | $3,498,205 | $2,597,819 | 0.74x | 100% | 219 | WIN |

- **OOS profitable windows: 6/6**
- **Avg OOS/IS_avg ratio: 1.64x** (OOS consistently beats average IS year)
- **Total OOS P&L: $20,088,843**
- OOS 2025: WR=100% (all trailing-stop exits, inherently positive)
- Compare to 1h: 6/6 WIN, 1.47x OOS/IS — 4h has slightly better OOS/IS ratio

---

## medium_tf_backtest.py — 1h Momentum Filter Configs (IS 2015-2024, OOS 2025)
**Script**: `v15/validation/medium_tf_backtest.py`
**Run date**: Feb 26, 2026 (server logs/mtf_1h.log)
**TF**: 1h primary, context_tfs=['daily','weekly'], min_tfs=2
**All 4 configs, IS 10yr aggregate:**

| Config | Trades | WR | P&L IS | Avg/trade | Sharpe | Δ vs baseline |
|--------|--------|----|--------|-----------|--------|---------------|
| baseline | 9,061 | 95.2% | $28,656,476 | $3,163 | 2.19 | — |
| mtf_exhaust | 9,059 | 95.2% | $28,852,035 | $3,185 | 2.16 | +$195,560 |
| mtf_full | 9,079 | 95.2% | $27,322,312 | $3,009 | 2.14 | -$1,334,164 |
| mtf_conflict | 9,081 | 95.2% | $27,263,785 | $3,002 | 2.14 | -$1,392,690 |

**OOS 2025:**

| Config | Trades | WR | P&L OOS | Avg/trade | Δ vs baseline |
|--------|--------|----|---------|-----------|---------------|
| baseline | 753 | 98.4% | $3,324,459 | $4,415 | — |
| mtf_exhaust | 753 | 98.4% | $3,324,459 | $4,415 | **+$0** (identical) |
| mtf_full | 753 | 98.4% | $3,240,118 | $4,303 | -$84,341 |
| mtf_conflict | 753 | 98.4% | $3,240,118 | $4,303 | -$84,341 |

**Verdict**: Baseline wins at 1h. Momentum filter neutral-to-negative at medium TF.
- Exhaust-only: trivially better IS (+0.7%) but identical OOS → not worth it
- Conflict blocking: actively hurts (-4.9% IS, -2.5% OOS) — blocking valid 1h signals
- Same pattern as 4h: momentum filter designed for 5-min context, doesn't apply at 1h/4h

---

## walk_forward_filters.py — 5-min System Momentum Filter Walk-Forward
**Script**: `v15/validation/walk_forward_filters.py` (or wf_filters equivalent)
**Run date**: Feb 26, 2026 (server logs/wf_filters.log)
**Configs tested**: sq50_bp vs mtf_exhaust vs baseline (5-min system, 6 rolling windows)

**Last window IS/OOS (IS 2020-2024 → OOS 2025):**
- IS baseline: $10,709,619, WR=94.8%, Sharpe=3.48, 6,563 trades
- IS sq50_bp: $10,691,590 (-$18K), Sharpe=3.46, 6,572 trades
- IS mtf_exhaust: $10,793,369 (+$84K), Sharpe=3.44, 6,556 trades
- OOS baseline: $1,557,826, WR=96.3%, 1,058 trades
- OOS sq50_bp: $1,555,535 (-$2,291) [LOSS]
- OOS mtf_exhaust: $1,562,133 (+$4,307) [WIN]

**Walk-Forward Summary (6 windows):**

| Window | sq50_bp Δ OOS | mtf_exhaust Δ OOS | sq WIN | ex WIN |
|--------|---------------|-------------------|--------|--------|
| IS 2015-2019 → OOS 2020 | -$971 | +$13,432 | LOSS | WIN |
| IS 2016-2020 → OOS 2021 | -$6,791 | +$5,653 | LOSS | WIN |
| IS 2017-2021 → OOS 2022 | -$706 | +$8,860 | LOSS | WIN |
| IS 2018-2022 → OOS 2023 | -$7,021 | -$9,404 | LOSS | LOSS |
| IS 2019-2023 → OOS 2024 | -$2,540 | +$65,210 | LOSS | WIN |
| IS 2020-2024 → OOS 2025 | -$2,291 | +$4,307 | LOSS | WIN |
| **TOTAL** | **-$20,320** | **+$88,057** | **0/6** | **5/6** |

**Verdict**:
- **sq50_bp (SQ gate)**: 0/6 OOS wins — definitively fails at 5-min level
- **mtf_exhaust**: 5/6 OOS wins, +$88K total — **modest but consistent improvement**
  - Effect size small (+5.6% of OOS P&L), but directional consistency is real
  - "Opponent momentum decelerating on 2+ higher TFs -> boost confidence" transfers OOS
  - Does NOT mean this is production-ready -- investigate 2023 loss window further

---

## short_term_tf_targets.py -- Short-term Forward Return Analysis
**Script**: `v15/validation/short_term_tf_targets.py`
**Run date**: Feb 26, 2026 (server, IS=2015-2024)
**Phase 1**: Forward return analysis, hold periods=[1,2,5,10,20]d, capital=$100K
**Phase 2**: IS/OOS backtest, hold=2d, stop=5%
**Phase 3**: Walk-forward (IS=5yr, OOS=1yr, 6 windows)
**Signals**: 32 total across 4 groups (5min/1h/4h top, higher-TF turning, divergence combos, baselines)

### Top Phase 1 Results (selected signals, IS 2015-2024):

| Signal | n | E[1d] | E[2d] | E[5d] | E[10d] | E[20d] | Notes |
|--------|---|-------|-------|-------|--------|--------|-------|
| G3_5min_top_4h_wkly | 3 | $2,252 | $10,103 | $12,011 | $29,015 | $42,177 | n too small, highly bullish |
| G3_5min_top_weekly_turning | 15 | $1,411 | $2,896 | $6,123 | $15,802 | $27,304 | Best robust divergence |
| G3_5min_top_4h_turning | 24 | $611 | $1,228 | $2,455 | $7,104 | $14,884 | Solid positive all holds |
| G2_4h_weekly_turning | 29 | $841 | $1,199 | $2,108 | $6,612 | $13,977 | 4h+weekly both turning up |
| G2_4h_turning_up | 102 | $392 | $652 | $1,288 | $3,942 | $8,201 | Most frequent turning signal |
| G4_5min_bottom | 128 | $521 | $1,087 | $2,180 | $5,890 | $11,654 | Control: bottom IS bullish |
| G1_5min_top | 201 | $214 | $387 | $831 | $2,431 | $4,921 | Top NOT bearish short-term |
| G1_1h_top | 482 | $12 | $1 | $145 | $1,102 | $2,877 | 1h top ~flat at 1-2d |
| G1_4h_top | 524 | $-10 | $-4 | $-89 | $487 | $1,934 | 4h top slightly negative 5d |
| G4_consensus_5 | 312 | $231 | $444 | $912 | $2,688 | $5,613 | All TFs bullish -- slight edge |
| G4_consensus_3 | 1,042 | $157 | $312 | $633 | $1,877 | $3,887 | Broad baseline |
| G3_1h_top_4h_wkly | 0 | -- | -- | -- | -- | -- | Geometrically impossible |
| G3_full_user_pattern | 0 | -- | -- | -- | -- | -- | Geometrically impossible |

### Key Findings:
1. **The user's observation pattern (5min top + 4h/weekly turning up) = bullish 24-48h**:
   - G3_5min_top_4h_wkly: E[2d]=$10,103 -- confirms 4h/weekly turn dominates
   - G3_5min_top_weekly_turning: E[2d]=$2,896, E[5d]=$6,123 -- robust with n=15
   - The higher-TF turning-up signal OVERWHELMS the 5min exhaustion at the top

2. **5min top is NOT bearish short-term** (counter to intuition):
   - G1_5min_top: E[2d]=$387 -- slightly positive, NOT a shorting signal
   - Only 4h top shows slightly negative 5d return (-$89) -- insufficient edge

3. **Geometric impossibility**: `G3_1h_top_4h_wkly` and `G3_full_user_pattern` fire 0 times.
   - When 1h is at top (pos>0.80), 4h covering more history is almost certainly also in upper half (pos>0.5)
   - The 5min+4h combo works because 5min covers only 2 days, allowing divergence

4. **Best robust signal**: G3_5min_top_weekly_turning (n=15, E[2d]=$2,896) -- the user's observed
   pattern confirmed: 5min exhaustion + weekly selling exhaustion = net bullish 2-5d

### Phase 2 IS/OOS (hold=2d, stop=5%, top 3 signals):
- G3_5min_top_weekly_turning: IS 2015-2024 n=15, WR=53%, P&L=$20K (thin edge with stop)
- G3_5min_top_4h_turning: IS n=24, WR=54%, P&L=$18K
- G3_5min_top_4h_wkly: IS n=3 (too rare for backtest)

### Phase 3 Walk-Forward (G3_5min_top_weekly_turning):
- Mixed -- IS edge too thin for clean walk-forward passing. Signal useful as confluence
  indicator, not as standalone trade entry.

### Conclusion:
- Short-term signals (2-5d) from TF state are weaker than 30-45d (tf_state_targets.py)
- Best use: DIRECTION CONFIRMATION. When 4h/weekly are turning up, buy dips even if 5min top.
- Does NOT provide actionable standalone signals -- confirm with S1041 channel structure.

---

## cross_tf_selector.py -- 1h+4h Trade Alignment Analysis
**Script**: `v15/validation/cross_tf_selector.py`
**Run date**: Feb 26, 2026 (server, logs/cross_tf.log)
**Hypothesis**: 1h trades that coincide with an active 4h bounce are higher quality.

### Method:
- Run 1h backtest + 4h backtest independently per year (IS 2015-2024, OOS 2025)
- Match 1h entries where: 4h_entry <= 1h_entry <= 4h_entry + 4h_hold_bars
- Split 1h trades into "aligned" (during 4h bounce) vs "non-aligned"

### OOS 2025 Results:
| Group | n | Avg P&L/trade | Total P&L |
|-------|---|---------------|-----------|
| All 1h trades | 222 | $14,969 | $3,323K |
| Aligned (4h active) | 6 | $29,463 | $177K |
| Non-aligned | 216 | $14,509 | $3,134K |

### IS 2015-2024 Results:
- Aligned 1h trades = ~3-4% of all 1h trades per year
- Avg P&L lift: ~1.8-2.0x vs non-aligned
- n too small per window for walk-forward (6 aligned trades/yr average)

### Key Findings:
- 4h confluence DOES improve 1h trade quality (1.97x avg P&L lift OOS)
- But only 2.7% of 1h trades are aligned -- far too rare to use as a standalone filter
- Using it as a filter would remove 97.3% of valid 1h trades, destroying capacity
- Best use: POSITION SIZING BOOST on aligned trades (1.5x-2x size when 4h bounce active)
  rather than using it as a gate.

---

## OpenEvolve Runs -- Final Status (Feb 26, 2026)

### 3B2 (WR>=90%, port 5555)
- Seed: S1041-based, 24t/100%WR/$1,588K, score=24
- Status after 300+ iterations: **seed remains best** -- no program beat score=24
- OE struggled at WR cliff: programs that add trades fall below 90% WR, scoring 0
- Both 3b2 and 3b3 write to shared `C:\AI\openevolve_signals\output\`

### 3B3 (WR>=85%, port 5557)
- Seed: 26t/96.15%WR/$1,580K, score=25
- Status after 300+ iterations: **seed remains best** -- stuck at score=25
- Early iter 1 generated 30t/86.7% (score=26) but did not persist

### run5 (P&L x WR scoring, port 5558)
- Seed: 24t/100%WR/$1,588K
- Status: only iteration 0 (seed), never evolved -- task likely died early

### Conclusion:
- **OE has NOT beaten S1041 seed across any run after 300 iterations**
- WR constraint (>=85-90%) creates optimization cliff: adding trades drops WR below threshold
- The constraint-based scoring may be too strict for evolutionary search
- Recommendation: try unconstrained scoring (pure P&L or Sharpe) in next OE run
- S1041 remains champion by default

---

## Portfolio Backtest -- Multi-System Allocation (Feb 26, 2026)

**Script**: `v15/validation/portfolio_backtest.py`
**Run date**: Feb 26, 2026 (server, all 4 phases)
**Capital**: $100K per system baseline, $1M total for allocation search
**Years**: 2015-2025 (11yr), walk-forward 6 windows IS 5yr -> OOS 1yr

### Phase 1 -- Per-System Independent Baseline ($100K each)

| System | Total P&L (11yr) | Trades | WR | Sharpe | Trd/yr | Max Hold |
|--------|-----------------|--------|-----|--------|--------|----------|
| 5min | $17,330,408 | 12,158 | 94.5% | 2.19 | 1,105 | 5h (intraday) |
| 1h | $31,980,935 | 9,814 | 95.4% | 2.32 | 892 | 10h (~1.5d) |
| 4h | $26,177,386 | 2,965 | 98.1% | 1.35 | 270 | 20h (~2.5d) |
| swing | $112,703 | 29 | 69.0% | 0.89 | 3 | 10d |

Runtime: 869s total (parallel). Swing 4.5s, 4h 107s, 1h 351s, 5min 868s.

### Annual P&L Correlation Matrix

|       | 5min  | 1h    | 4h    | swing |
|-------|-------|-------|-------|-------|
| 5min  | 1.000 | 0.917 | 0.831 | 0.203 |
| 1h    | 0.917 | 1.000 | 0.865 | 0.176 |
| 4h    | 0.831 | 0.865 | 1.000 | -0.008 |
| swing | 0.203 | 0.176 | -0.008 | 1.000 |

Key: intraday systems highly correlated (0.83-0.92). Swing near-zero with everything.

### Phase 2 -- Allocation Grid Search (286 combos, $1M total, 10% step)

Top 10 by Sharpe -- ALL have 0% to 4h:

| Rank | 5min | 1h | 4h | Swing | Total P&L | Sharpe | MaxDD | Worst Yr |
|------|------|-----|-----|-------|-----------|--------|-------|----------|
| 1 | 0% | 10% | 0% | 90% | $33.0M | 2.35 | -2.7% | $908K |
| 2 | 10% | 10% | 0% | 80% | $50.2M | 2.34 | -1.9% | $1.4M |
| 3 | 10% | 20% | 0% | 70% | $82.1M | 2.34 | -1.2% | $2.3M |
| 4 | 0% | 20% | 0% | 80% | $64.9M | 2.34 | -1.6% | $1.8M |
| 5 | 10% | 30% | 0% | 60% | $114M | 2.33 | -0.8% | $3.2M |

### Phase 3 -- Walk-Forward Validation (6/6 WIN for all top 3)

**Allocation #1** (0/10/0/90): 6/6 WIN, total OOS $24.1M, avg $4.0M/yr
**Allocation #2** (10/10/0/80): 6/6 WIN, total OOS $36.3M, avg $6.1M/yr
**Allocation #3** (10/20/0/70): 6/6 WIN, total OOS $59.7M, avg $9.9M/yr

### Phase 4 -- Overlap Analysis (top-1 allocation: 0/10/0/90)

- Zero concurrent positions (1h and swing on completely different timescales)
- Max concurrent: 0, overlap time: 0% of market hours
- No capital conflict between systems

### Key Findings

1. **1h is the best single system** -- highest P&L ($32M), highest Sharpe (2.32), 95.4% WR
2. **4h is redundant** -- 86.5% correlated with 1h, lower Sharpe (1.35 vs 2.32), adds correlated risk
3. **Swing inflates Sharpe** -- 90% allocation is a math artifact (idle capital = low variance)
4. **Practical deployment**: 60-70% 1h / 20-30% 5min / swing overlay when triggered
5. **Gap identified**: dropping 4h leaves no system covering 1.5-day to 10-day holds
   - 1h max hold = 10h (~1.5 trading days)
   - Swing = 2-3 trades/yr (not continuous coverage)
   - **Daily TF (1d) never tested** -- config exists (5-day hold, weekly+monthly context) but no results
   - TODO: run `medium_tf_backtest.py --tf 1d` to fill the multi-day gap
