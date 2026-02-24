
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

Status: RUNNING

Results:
  (pending training run)

Per-year AUC (LOO CV):
  (pending)

11-year backtest:
  (pending)

================================================================================
