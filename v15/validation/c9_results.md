
================================================================================
Walk-Forward Validation — Arch415 (c9 branch) [2026-02-23]
================================================================================
Method: 5yr rolling IS → 1yr OOS, 6 windows (2020-2025 as OOS years)
bounce_cap=12x, $100K/yr, $500K max trade

RESULTS: ALL 6 WINDOWS PASS — OOS/IS ratio=1.85x (GENERALIZES, not overfit)

  Window | IS years    | OOS  | OOS P&L    | OOS/IS ratio
  -------|-------------|------|------------|-------------
  1      | 2015-2019   | 2020 | $1,224,622 | 2.53x ✓
  2      | 2016-2020   | 2021 |   $768,435 | 1.21x ✓
  3      | 2017-2021   | 2022 | $1,209,186 | 1.77x ✓
  4      | 2018-2022   | 2023 |   $611,891 | 0.71x ✓
  5      | 2019-2023   | 2024 |   $938,770 | 1.12x ✓
  6      | 2020-2024   | 2025 |   $615,817 | 0.65x ✓

OOS aggregate (2020-2025):
  Trades=7,183 | WR=95.0% | PF=105.98 | P&L=$5,368,721
  Sharpe=3.54  | MaxDD=3.5% | 6/6 profitable years

CONCLUSION: DOW/TOD patterns are genuine signal. Arch415 safe to build on in c9.
