
================================================================================
c10 Branch — Validation Results
================================================================================
Base: c9/Arch418 — AUC=0.813, Cal Brier=0.050, 189 features, 10,604 trades
Goal: Improve signal quality model via better SPY market-context features

Validation methodology:
  - LOO CV (2015-2024): quick per-iteration AUC comparison vs 0.813 baseline
  - 11-year backtest (2015-2025): confirms P&L impact of each arch
  - Walk-forward (final winner only): 5yr IS → 1yr OOS, 6 windows
  - True holdout (2025): locked out until very end

================================================================================
Arch1 — SPY RSI Features (+4 correlation features)
================================================================================
Hypothesis: SPY RSI(14) and TSLA/SPY RSI divergence capture market context
that raw SPY return misses. When TSLA is oversold (low RSI) but SPY is NOT,
that's isolated TSLA weakness → better bounce. When both oversold → market
selloff → worse bounce quality.

Changes from c9:
  - CORRELATION_FEATURES: 5 → 9 features
  - spy_rsi_14: SPY RSI(14) at signal time
  - spy_tsla_rsi_divergence: TSLA RSI(14) - SPY RSI(14) (+ = TSLA oversold vs mkt)
  - spy_intraday_return: SPY % return from today's open (green/red market day)
  - spy_vol_ratio_20: SPY volume / 20-bar avg (market-driven move indicator)
  - Base feature vector: 169 → 173 dims; trained feature vec: 189 → 193 dims
  - Fixed: sig_vix_x_position index hardcode 162 → dynamic lookup (was using
    wrong temporal feature as VIX proxy in interaction term)

Status: COMPLETE — NEUTRAL (features don't hurt, but no improvement)

Arch1 (default params, no tuned_params):
  AUC: 0.793  Cal Brier: 0.051  (worse — default params insufficient for 193-dim space)
  Per-year AUC: 0.704, 0.737, 0.783, 0.795, 0.801, 0.848, 0.761, 0.800, 0.846, 0.811

Arch1b (same tuned params as c9 — fair comparison):
  AUC: 0.813  Cal Brier: 0.050  (matches c9 exactly)
  Per-year AUC: 0.740, 0.745, 0.790, 0.806, 0.826, 0.863, 0.814, 0.825, 0.862, 0.818
  Model file: v15/validation/signal_quality_model_c10_arch1b.pkl
  Feature dim: 193 (173 base + 21 extended meta)

New feature importances in Arch1b (win classifier):
  spy_vol_ratio_20: rank 10 (250 importance) — genuinely useful
  spy_intraday_return, spy_tsla_corr_20: rank 7-11
  spy_rsi_14, spy_tsla_rsi_divergence: rank 20-30 (low, redundant with existing features)
  sig_vix_x_position: rank 20 in win, rank 11 in pnl — now working (was broken in c9)

11-year backtest (2015-2025, bounce_cap=12x, max_trade_usd=$1M):
  Baseline:       $16,552,438  (identical to c9)
  Old-tiers:      $16,746,182  (+1.2%, identical to c9)
  Upscale-only:   $17,174,699  (+3.8%, identical to c9)

Per-year upscale-only delta (identical to c9 — predictions are near-equivalent):
  2015: +$118   2016: +$357   2017: +$1    2018: +$4,839  2019: +$65
  2020: +$228,992  2021: +$13,212  2022: +$6,822
  2023: +$260,760  2024: +$97,111  2025: +$9,983

CONCLUSION: New SPY RSI features are information-redundant — already captured by
spy_return_5bar, vix_level. Model is near its information ceiling at 10K trades.
spy_vol_ratio_20 is the most useful new feature but not enough to shift P&L.
sig_vix_x_position bug fix (index 162→167) now works correctly.
NEXT: Arch2 — try lag features (recent trade outcomes) to capture system state.

================================================================================
Arch2 — Lag Features (recent trade win/loss history + system state)
================================================================================
Hypothesis: The 4 c9 lag features (recent_win_rate_10, streak, day_pnl_running,
time_since_last_trade_min) capture SYSTEM STATE that static features miss.
When system is on a hot streak → quality scores may be higher. When in cold streak →
lower. These are genuinely novel (can't be derived from existing features).
No look-ahead bias: only uses trades that closed BEFORE the current signal.

Status: PENDING

================================================================================
