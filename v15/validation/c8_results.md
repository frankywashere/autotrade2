Loaded signal quality model from v15/validation/signal_quality_model_tuned.pkl
  CV AUC: 0.8060365762122162

======================================================================
LOADING DATA
======================================================================
Loading TSLA minute data from data/TSLAMin.txt...
  Loaded 1,854,183 1-min bars: 2015-01-02 11:40:00-05:00 to 2025-09-27 00:00:00-04:00
  Resampling to 5-min...
  5-min: 440,405 bars
  Resampling to 1h...
  1h: 44,229 bars
  Resampling to 4h...
  4h: 13,280 bars
  Resampling to daily...
  daily: 3,174 bars
Loading SPY minute data from data/SPYMin.txt...
  Loaded 2,144,644 1-min bars
  SPY 5-min: 504,398 bars
  Fetching daily VIX from yfinance...
  VIX: 3017 daily bars (2014-01-02 to 2025-12-30)
  Data loaded in 8.8s

======================================================================
RUNNING BACKTESTS: BASELINE vs ML-SIZED
======================================================================
⚠️  WARNING: ML MODEL NOT LOADED — trading without ML filtering!
⚠️  All signals will pass through unfiltered by ML models.
⚠️  This is ONLY acceptable for backtesting baseline comparisons.
[PRE-LOADED] 35486 bars: 2025-01-01 00:00:00-05:00 to 2025-09-27 00:00:00-04:00
[REALISTIC] max_leverage=4.0x, slippage=3.0bps, commission=$0.005/share

Backtesting from bar 100 to 35486 (interval=6)...
  [1%] bar=502/35486, trades=14, equity=$106,073
  [3%] bar=1006/35486, trades=34, equity=$113,030
  [4%] bar=1510/35486, trades=52, equity=$124,276
  [5%] bar=2014/35486, trades=62, equity=$129,503
  [7%] bar=2518/35486, trades=80, equity=$135,544
  [8%] bar=3022/35486, trades=96, equity=$148,533
  [10%] bar=3526/35486, trades=108, equity=$148,970
  [11%] bar=4030/35486, trades=116, equity=$151,822
  [13%] bar=4534/35486, trades=123, equity=$151,910
  [14%] bar=5038/35486, trades=141, equity=$167,737
  [15%] bar=5542/35486, trades=156, equity=$184,844
  [17%] bar=6046/35486, trades=172, equity=$188,746
  [18%] bar=6550/35486, trades=183, equity=$189,525
  [20%] bar=7054/35486, trades=199, equity=$197,172
  [21%] bar=7558/35486, trades=215, equity=$210,001
  [23%] bar=8062/35486, trades=227, equity=$223,535
  [24%] bar=8566/35486, trades=244, equity=$232,832
  [25%] bar=9070/35486, trades=260, equity=$241,465
  [27%] bar=9574/35486, trades=280, equity=$245,601
  [28%] bar=10078/35486, trades=289, equity=$245,602
  [30%] bar=10582/35486, trades=314, equity=$254,139
  [31%] bar=11086/35486, trades=341, equity=$257,285
  [32%] bar=11590/35486, trades=362, equity=$270,802
  [34%] bar=12094/35486, trades=376, equity=$270,935
  [35%] bar=12598/35486, trades=392, equity=$271,260
  [37%] bar=13102/35486, trades=417, equity=$290,161
  [38%] bar=13606/35486, trades=434, equity=$293,388
  [40%] bar=14110/35486, trades=449, equity=$301,404
  [41%] bar=14614/35486, trades=465, equity=$310,403
  [42%] bar=15118/35486, trades=486, equity=$326,560
  [44%] bar=15622/35486, trades=505, equity=$327,428
  [45%] bar=16126/35486, trades=519, equity=$349,232
  [47%] bar=16630/35486, trades=536, equity=$351,381
  [48%] bar=17134/35486, trades=547, equity=$358,784
  [50%] bar=17638/35486, trades=564, equity=$367,363
  [51%] bar=18142/35486, trades=582, equity=$369,437
  [52%] bar=18646/35486, trades=600, equity=$375,831
  [54%] bar=19150/35486, trades=613, equity=$376,005
  [55%] bar=19654/35486, trades=632, equity=$379,169
  [57%] bar=20158/35486, trades=644, equity=$381,368
  [58%] bar=20662/35486, trades=670, equity=$391,593
  [60%] bar=21166/35486, trades=689, equity=$391,654
  [61%] bar=21670/35486, trades=708, equity=$392,120
  [62%] bar=22174/35486, trades=727, equity=$393,610
  [64%] bar=22678/35486, trades=739, equity=$396,385
  [65%] bar=23182/35486, trades=751, equity=$404,950
  [67%] bar=23686/35486, trades=767, equity=$405,129
  [68%] bar=24190/35486, trades=783, equity=$407,078
  [70%] bar=24694/35486, trades=791, equity=$407,080
  [71%] bar=25198/35486, trades=806, equity=$409,512
  [72%] bar=25702/35486, trades=814, equity=$410,166
  [74%] bar=26206/35486, trades=828, equity=$410,239
  [75%] bar=26710/35486, trades=845, equity=$410,988
  [77%] bar=27214/35486, trades=860, equity=$415,631
  [78%] bar=27718/35486, trades=877, equity=$417,396
  [79%] bar=28222/35486, trades=885, equity=$417,585
  [81%] bar=28726/35486, trades=903, equity=$420,606
  [82%] bar=29230/35486, trades=916, equity=$430,423
  [84%] bar=29734/35486, trades=927, equity=$435,017
  [85%] bar=30238/35486, trades=943, equity=$435,582
  [87%] bar=30742/35486, trades=956, equity=$435,622
  [88%] bar=31246/35486, trades=970, equity=$437,037
  [89%] bar=31750/35486, trades=978, equity=$437,377
  [91%] bar=32254/35486, trades=988, equity=$437,724
  [92%] bar=32758/35486, trades=999, equity=$437,488
  [94%] bar=33262/35486, trades=1011, equity=$437,497
  [95%] bar=33766/35486, trades=1019, equity=$437,556
  [97%] bar=34270/35486, trades=1025, equity=$437,704
  [98%] bar=34774/35486, trades=1032, equity=$438,647
  [99%] bar=35278/35486, trades=1042, equity=$440,073

Completed in 54.3s

======================================================================
CHANNEL SURFER BACKTEST RESULTS [NO ML — UNFILTERED]
======================================================================
Trades: 1047 | Win Rate: 97% | PF: 224.75 | Total P&L: $340,114.89 | Avg Hold: 5 bars | Avg Win: 0.53% | Avg Loss: -0.06% | Max DD: 0.5% | Expectancy: $324.85/trade

  Signal Stats:
    Total physics signals: 1369
    Trades taken:          1047
    Pass-through rate:     76.5%

  ⚠️  ML MODELS INACTIVE — all 16 sub-models skipped!
      Signals passed only physics + confidence filters.
      This is ONLY valid for baseline comparison.

Exit reason breakdown:
  stop        :  21 trades, WR=0%, P&L=$-562.47
  trail       : 1024 trades, WR=99%, P&L=$341,634.87
  ou_timeout  :   2 trades, WR=0%, P&L=$-957.52
  BUY         : 518 trades, WR=96%, P&L=$117,168.77
  SELL        : 529 trades, WR=97%, P&L=$222,946.11
  bounce      : 384 trades, WR=98%, P&L=$337,770.71, avg size=$211,826
  break       : 663 trades, WR=96%, P&L=$2,344.18, avg size=$489

Trade quality (MAE/MFE):
  Avg MAE: 0.718% (worst drawdown before exit)
  Avg MFE: 0.544% (best unrealized gain)
  Winner efficiency: 85% (% of MFE captured at exit)
  Loser MAE/MFE: 99.4x (how far wrong vs best)
  Winner MAE: 0.722%
  Loser  MAE: 0.622%

Performance by hour (ET):
   3:00   29 trades  WR=97%  avg=+$278.9  🟢███████████████████████████████████████████████████████
   4:00   33 trades  WR=97%  avg=+$240.3  🟢████████████████████████████████████████████████
   5:00   46 trades  WR=91%  avg=+$244.2  🟢████████████████████████████████████████████████
   6:00   48 trades  WR=98%  avg=+$193.1  🟢██████████████████████████████████████
   7:00   58 trades  WR=97%  avg=+$53.4  🟢██████████
   8:00  102 trades  WR=98%  avg=+$511.1  🟢██████████████████████████████████████████████████████████████████████████████████████████████████████
   9:00  115 trades  WR=100%  avg=+$220.5  🟢████████████████████████████████████████████
  10:00   84 trades  WR=100%  avg=+$291.8  🟢██████████████████████████████████████████████████████████
  11:00   81 trades  WR=99%  avg=+$377.3  🟢███████████████████████████████████████████████████████████████████████████
  12:00   69 trades  WR=96%  avg=+$383.0  🟢████████████████████████████████████████████████████████████████████████████
  13:00   95 trades  WR=99%  avg=+$594.9  🟢██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
  14:00   93 trades  WR=98%  avg=+$428.8  🟢█████████████████████████████████████████████████████████████████████████████████████
  15:00   61 trades  WR=92%  avg=+$295.4  🟢███████████████████████████████████████████████████████████
  16:00   49 trades  WR=92%  avg=+$271.6  🟢██████████████████████████████████████████████████████
  17:00   50 trades  WR=94%  avg=+$134.4  🟢██████████████████████████
  18:00   27 trades  WR=81%  avg=+$78.5  🟢███████████████
  19:00    6 trades  WR=100%  avg=+$759.5  🟢███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
  20:00    1 trades  WR=100%  avg=+$390.6  🟢██████████████████████████████████████████████████████████████████████████████

Performance by day:
  Mon  184 trades  WR=95%  P&L=$41,404  avg=+$225.0
  Tue  224 trades  WR=98%  P&L=$63,236  avg=+$282.3
  Wed  194 trades  WR=98%  P&L=$51,853  avg=+$267.3
  Thu  219 trades  WR=96%  P&L=$101,641  avg=+$464.1
  Fri  225 trades  WR=96%  P&L=$78,164  avg=+$347.4
  Sat    1 trades  WR=100%  P&L=$3,817  avg=+$3816.9

Signal component analysis — ALL (1047 trades):
  Component          Avg(Win)   Avg(Loss)  WinCorr    PnlCorr   
  position_score     0.942      0.941      +0.002     +0.026       
  energy_score       0.712      0.779      -0.028     +0.113       
  entropy_score      0.701      0.683      +0.012     +0.114       
  confluence_score   0.808      0.771      +0.040     -0.001       
  timing_score       0.213      0.132      +0.045     -0.129       
  channel_health     0.414      0.418      -0.005     -0.129       
  confidence         0.629      0.645      -0.023     +0.129       

Signal component analysis — BOUNCE (384 trades):
  Component          Avg(Win)   Avg(Loss)  WinCorr    PnlCorr   
  position_score     0.930      0.905      +0.020     +0.030       
  energy_score       0.256      0.296      -0.016     -0.060       
  entropy_score      0.469      0.232      +0.114     +0.012     **
  confluence_score   0.851      0.838      +0.014     +0.029       
  timing_score       0.575      0.513      +0.033     -0.011       
  channel_health     0.521      0.526      -0.008     -0.063       
  confidence         0.541      0.545      -0.007     -0.003       

Signal component analysis — BREAK (663 trades):
  Component          Avg(Win)   Avg(Loss)  WinCorr    PnlCorr   
  position_score     0.950      0.953      -0.010     +0.008       
  energy_score       0.981      0.947      +0.105     +0.040     **
  entropy_score      0.838      0.839      -0.003     +0.049       
  confluence_score   0.782      0.748      +0.039     +0.031       
  channel_health     0.351      0.380      -0.068     -0.024       
  confidence         0.681      0.679      +0.004     +0.070       

============================================================
SIGNAL FILTER DIAGNOSTICS (1369 total physics signals)
============================================================
  not_buy_sell                     4511
  total_signals                    1369
  peak_boost                       1078
  eq_scale                         1078
  quad_eq_scale                    1078
  ch_health_sizing                 1078
  conf_score_sizing                1078
  cycle_size                       1078
  trade_decay                      1078
  near_ath                         1073
  dyn_cap_hot                      1042
  streak_accel                      966
  big_win_regime                    963
  equity_delever                    962
  eq_growth_mom                     962
  exp_win_streak                    901
  drought_boost                     854
  low_var_boost                     802
  high_mfe_regime                   801
  hot_streak                        761
  confluence_boost                  740
  long_streak                       736
  break_hard_reduce                 694
  break_aligned_te                  694
  brk_univ                          694
  break_low_rev                     685
  break_freq_reduce                 682
  momentum_confirmed                675
  brk_sub500                        663
  break_dir_align                   645
  triple_eq                         618
  perfect_wr                        553
  big_opp_history                   553
  pnl_accel                         528
  extreme_break                     490
  trade_eff                         477
  trail_momentum                    463
  pnl_trend_up                      459
  accel_wins                        431
  parity_reduce                     429
  eq_accel                          418
  amp_decline                       413
  high_ke_break_boost               411
  bounce_sized_up                   384
  wide_ch_bounce                    384
  quality_scored                    384
  comprehensive_score               384
  comprehensive_v2                  384
  exp_score                         384
  quad_be                           384
  composite_q                       384
  pos_extremity                     384
  wide_trend                        384
  width_theta                       384
  triple_qual                       384
  wsh_sizing                        384
  mom_width                         384
  conf_pos_prod                     384
  tp_narrow                         384
  trail_stop_compound               384
  brk_lowconf                       383
  w_break_prob                      382
  all_established                   378
  trend_break_boost                 377
  vol_confirm                       371
  clean_brk_20                      367
  multi_tf_be_cont                  366
  ou_trend_break                    364
  p25_be                            363
  break_sell_boost                  361
  high_rr_ratio                     360
  timing_boost                      358
  median_be                         356
  weighted_be                       354
  high_bc_bounce                    346
  low_vol_bounce                    343
  p75_be                            342
  fast_vs_mature                    342
  max_pe_boost                      335
  max_pe_loaded                     335
  extreme_pe                        335
  clean_exit_bounce                 335
  slope_aligned_break               334
  pos_score_boost                   324
  good_half_life                    323
  quad_ke_boost                     322
  extreme_pos_sig                   317
  pnl_accel_boost                   317
  late_funded                       316
  mid_recovered                     306
  high_te_break                     304
  weighted_ke_boost                 298
  low_volume                        297
  high_pe_bounce                    297
  multi_ke_boost                    294
  te_health                         294
  multi_tf_edge_boost               293
  late_established                  293
  narrow_break_skip                 289
  mature_system                     282
  mfe_mae_ratio                     281
  long_osc_reduce                   280
  median_pe_boost                   279
  wide_dz                           277
  low_bp_bounce                     276
  final_stretch                     275
  bc_tight                          271
  harmonic_bounce                   266
  low_pnl_vol                       265
  wide_bounce_boost                 262
  low_energy_bounce                 260
  w_energy_prod                     252
  contrarian_boost                  250
  be_cont_penalty                   248
  all_tf_low_pe                     246
  high_pe_boost                     245
  min_ke_penalty                    241
  weighted_health                   224
  edge_bounce_boost                 223
  anti_cluster                      219
  double_weak                       219
  free_spring                       218
  fm_boost                          216
  p25_pe                            213
  4fq_avg                           207
  ch_break_sizeup                   207
  5f_best                           205
  slope_align_bounce                204
  osc_timing                        194
  sys_total_e                       194
  pos_n3                            186
  ent_pos_penalty                   185
  lo_ent_sig                        185
  counter_trend_boost               184
  max_vol_conf                      184
  p90_be                            169
  lev_cap                           167
  p25_ke                            165
  primary_best                      164
  multi_tf_pe_boost                 164
  hi_avg_pe                         164
  median_edge                       158
  low_conf_reduce                   155
  low_conf                          155
  danger_zone                       151
  lossless_regime                   144
  sys_spring                        141
  diversity_reduce                  139
  pnl_trend                         137
  near_bounce_timing                127
  dir_consensus                     125
  pe_vol_h                          123
  edge_vol                          122
  free_pe                           120
  same_dir_reduce                   117
  free_spr_edge                     117
  theta_pe_prod                     114
  hot_zone_boost                    113
  all_qfit                          112
  trade_usd_cap                     108
  spring_edge                       106
  min_health_ok                     106
  loss_cluster                      105
  good_regime_bounce                104
  low_ke_reduce                     101
  geo_health                         99
  mom_turn_reduce                    98
  very_hi_pe                         97
  high_be_reduce                     96
  all_energized                      95
  high_theta_bounce                  95
  max_theta_boost                    95
  theta_var                          95
  exposure_cap                       94
  high_rsq_bounce                    94
  fast_hl_bounce                     90
  channel_mature                     84
  ke_consensus                       83
  high_vol_break                     83
  mom_dir_consensus                  81
  same_type_loss                     77
  geo_ke                             76
  momentum_break                     76
  premium_sig                        75
  best_prh                           74
  chaos_mom_bounce                   73
  spring_eq_400                      73
  low_mae_regime                     71
  hi_energy_sig                      70
  extreme_pos_boost                  64
  break_after_stop                   64
  ou_revert_break                    63
  buy_bull_mom                       60
  spring_bounce_boost                59
  slope_consensus                    59
  vol_pos_edge                       55
  4f_q_avg                           54
  geo_hpr                            54
  sell_all_down                      50
  wide_tp_bounce                     49
  early_conf_brk                     48
  timed_eq                           46
  extreme_gw_gl                      43
  te5_300k                           43
  wt_geo_peh                         42
  mid_ch_reduce                      42
  poor_tf                            41
  te_low_cv                          39
  extreme_dual                       38
  daily_pnl_cap                      38
  early_dz                           38
  post_loss_reduce                   37
  loss_cooldown                      37
  4f_evidence                        37
  ordered_energy                     34
  safe_support                       33
  multi_tf_be_penalty                33
  bounce_stop_pen                    33
  loss_growing                       32
  theta_decr                         31
  bp_asym_align                      31
  mom_aligned_boost                  30
  brk_weak_pos                       28
  all_extreme                        28
  bounce_low_pe                      26
  anti_revenge                       26
  dyn_cap_normal                     26
  severe_cluster                     26
  conf_energy                        25
  loss_proximity                     25
  mom_turning                        24
  post_stop_reduce                   23
  dir_unanim                         23
  low_conf_buy_bounce                22
  ch_break_sizedown                  21
  hot_momentum                       20
  hi_conf_bounce                     19
  bounce_high_be                     19
  late_compound                      18
  worst_bounce_reduce                16
  5f_geo_full                        15
  6f_geo                             13
  ke_monotonic                       13
  vol_health_pos                     11
  pnl_vol_spike                      11
  dead_zone_reduce                   10
  ordered_ke                          9
  pe_be_ratio                         8
  te_uniform                          7
  full_energy                         6
  all_tf_floor                        6
  big_loss_reduce                     6
  sideways_reduce                     6
  big_win_mom                         3
  anti_pyramid                        2
  timeout_avoid                       2
  steep_slope                         2
  hot_bounce_10                       2
  timed_conf                          1
  strong_revert                       1
  consec_loss_reduce                  1
  q_streak_3                          1
  pe_pos_rev                          1
  geo_vep                             1
  trail_loss_cluster                  1
  TRADES TAKEN                     1047
============================================================

⚠️  WARNING: ML MODEL NOT LOADED — trading without ML filtering!
⚠️  All signals will pass through unfiltered by ML models.
⚠️  This is ONLY acceptable for backtesting baseline comparisons.
[ML-SIZING] Signal quality model loaded (169 base features)
[PRE-LOADED] 35486 bars: 2025-01-01 00:00:00-05:00 to 2025-09-27 00:00:00-04:00
[REALISTIC] max_leverage=4.0x, slippage=3.0bps, commission=$0.005/share

Backtesting from bar 100 to 35486 (interval=6)...
  [1%] bar=502/35486, trades=14, equity=$106,073
  [3%] bar=1006/35486, trades=34, equity=$113,030
  [4%] bar=1510/35486, trades=52, equity=$124,276
  [5%] bar=2014/35486, trades=62, equity=$129,503
  [7%] bar=2518/35486, trades=80, equity=$135,544
  [8%] bar=3022/35486, trades=96, equity=$148,533
  [10%] bar=3526/35486, trades=108, equity=$148,970
  [11%] bar=4030/35486, trades=116, equity=$151,822
  [13%] bar=4534/35486, trades=123, equity=$151,910
  [14%] bar=5038/35486, trades=141, equity=$167,737
  [15%] bar=5542/35486, trades=156, equity=$184,844
  [17%] bar=6046/35486, trades=172, equity=$188,746
  [18%] bar=6550/35486, trades=183, equity=$189,525
  [20%] bar=7054/35486, trades=199, equity=$197,172
  [21%] bar=7558/35486, trades=215, equity=$210,829
  [23%] bar=8062/35486, trades=227, equity=$223,287
  [24%] bar=8566/35486, trades=244, equity=$231,279
  [25%] bar=9070/35486, trades=260, equity=$240,133
  [27%] bar=9574/35486, trades=280, equity=$244,647
  [28%] bar=10078/35486, trades=289, equity=$244,648
  [30%] bar=10582/35486, trades=314, equity=$253,207
  [31%] bar=11086/35486, trades=341, equity=$256,355
  [32%] bar=11590/35486, trades=362, equity=$270,085
  [34%] bar=12094/35486, trades=376, equity=$270,408
  [35%] bar=12598/35486, trades=392, equity=$270,617
  [37%] bar=13102/35486, trades=417, equity=$289,526
  [38%] bar=13606/35486, trades=434, equity=$292,772
  [40%] bar=14110/35486, trades=449, equity=$300,792
  [41%] bar=14614/35486, trades=465, equity=$309,815
  [42%] bar=15118/35486, trades=486, equity=$325,979
  [44%] bar=15622/35486, trades=505, equity=$326,886
  [45%] bar=16126/35486, trades=519, equity=$348,745
  [47%] bar=16630/35486, trades=536, equity=$350,902
  [48%] bar=17134/35486, trades=547, equity=$358,305
  [50%] bar=17638/35486, trades=564, equity=$366,887
  [51%] bar=18142/35486, trades=582, equity=$368,966
  [52%] bar=18646/35486, trades=600, equity=$375,363
  [54%] bar=19150/35486, trades=613, equity=$375,539
  [55%] bar=19654/35486, trades=632, equity=$378,709
  [57%] bar=20158/35486, trades=644, equity=$380,908
  [58%] bar=20662/35486, trades=670, equity=$391,160
  [60%] bar=21166/35486, trades=689, equity=$391,221
  [61%] bar=21670/35486, trades=708, equity=$391,690
  [62%] bar=22174/35486, trades=727, equity=$393,183
  [64%] bar=22678/35486, trades=739, equity=$395,966
  [65%] bar=23182/35486, trades=751, equity=$404,531
  [67%] bar=23686/35486, trades=767, equity=$404,715
  [68%] bar=24190/35486, trades=783, equity=$406,723
  [70%] bar=24694/35486, trades=791, equity=$406,725
  [71%] bar=25198/35486, trades=806, equity=$409,160
  [72%] bar=25702/35486, trades=814, equity=$409,814
  [74%] bar=26206/35486, trades=828, equity=$409,888
  [75%] bar=26710/35486, trades=845, equity=$410,638
  [77%] bar=27214/35486, trades=860, equity=$415,282
  [78%] bar=27718/35486, trades=877, equity=$417,046
  [79%] bar=28222/35486, trades=885, equity=$417,236
  [81%] bar=28726/35486, trades=903, equity=$420,258
  [82%] bar=29230/35486, trades=916, equity=$430,077
  [84%] bar=29734/35486, trades=927, equity=$434,677
  [85%] bar=30238/35486, trades=943, equity=$435,242
  [87%] bar=30742/35486, trades=956, equity=$435,282
  [88%] bar=31246/35486, trades=970, equity=$436,699
  [89%] bar=31750/35486, trades=978, equity=$437,039
  [91%] bar=32254/35486, trades=988, equity=$437,387
  [92%] bar=32758/35486, trades=999, equity=$437,150
  [94%] bar=33262/35486, trades=1011, equity=$437,159
  [95%] bar=33766/35486, trades=1019, equity=$437,219
  [97%] bar=34270/35486, trades=1025, equity=$437,367
  [98%] bar=34774/35486, trades=1032, equity=$438,312
  [99%] bar=35278/35486, trades=1042, equity=$439,741

Completed in 58.6s

======================================================================
CHANNEL SURFER BACKTEST RESULTS [NO ML — UNFILTERED]
======================================================================
Trades: 1047 | Win Rate: 97% | PF: 224.35 | Total P&L: $339,782.33 | Avg Hold: 5 bars | Avg Win: 0.53% | Avg Loss: -0.06% | Max DD: 0.5% | Expectancy: $324.53/trade

  Signal Stats:
    Total physics signals: 1369
    Trades taken:          1047
    Pass-through rate:     76.5%

  ⚠️  ML MODELS INACTIVE — all 16 sub-models skipped!
      Signals passed only physics + confidence filters.
      This is ONLY valid for baseline comparison.

Exit reason breakdown:
  stop        :  21 trades, WR=0%, P&L=$-563.71
  trail       : 1024 trades, WR=99%, P&L=$341,303.58
  ou_timeout  :   2 trades, WR=0%, P&L=$-957.54
  BUY         : 518 trades, WR=96%, P&L=$118,517.96
  SELL        : 529 trades, WR=97%, P&L=$221,264.37
  bounce      : 384 trades, WR=98%, P&L=$337,463.59, avg size=$211,798
  break       : 663 trades, WR=96%, P&L=$2,318.74, avg size=$482

Trade quality (MAE/MFE):
  Avg MAE: 0.718% (worst drawdown before exit)
  Avg MFE: 0.544% (best unrealized gain)
  Winner efficiency: 85% (% of MFE captured at exit)
  Loser MAE/MFE: 99.4x (how far wrong vs best)
  Winner MAE: 0.722%
  Loser  MAE: 0.622%

Performance by hour (ET):
   3:00   29 trades  WR=97%  avg=+$276.8  🟢███████████████████████████████████████████████████████
   4:00   33 trades  WR=97%  avg=+$240.4  🟢████████████████████████████████████████████████
   5:00   46 trades  WR=91%  avg=+$248.2  🟢█████████████████████████████████████████████████
   6:00   48 trades  WR=98%  avg=+$192.9  🟢██████████████████████████████████████
   7:00   58 trades  WR=97%  avg=+$53.6  🟢██████████
   8:00  102 trades  WR=98%  avg=+$511.9  🟢██████████████████████████████████████████████████████████████████████████████████████████████████████
   9:00  115 trades  WR=100%  avg=+$228.1  🟢█████████████████████████████████████████████
  10:00   84 trades  WR=100%  avg=+$299.1  🟢███████████████████████████████████████████████████████████
  11:00   81 trades  WR=99%  avg=+$389.0  🟢█████████████████████████████████████████████████████████████████████████████
  12:00   69 trades  WR=96%  avg=+$383.0  🟢████████████████████████████████████████████████████████████████████████████
  13:00   95 trades  WR=99%  avg=+$597.4  🟢███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
  14:00   93 trades  WR=98%  avg=+$429.1  🟢█████████████████████████████████████████████████████████████████████████████████████
  15:00   61 trades  WR=92%  avg=+$236.1  🟢███████████████████████████████████████████████
  16:00   49 trades  WR=92%  avg=+$277.3  🟢███████████████████████████████████████████████████████
  17:00   50 trades  WR=94%  avg=+$134.4  🟢██████████████████████████
  18:00   27 trades  WR=81%  avg=+$78.6  🟢███████████████
  19:00    6 trades  WR=100%  avg=+$774.4  🟢██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
  20:00    1 trades  WR=100%  avg=+$390.6  🟢██████████████████████████████████████████████████████████████████████████████

Performance by day:
  Mon  184 trades  WR=95%  P&L=$41,427  avg=+$225.1
  Tue  224 trades  WR=98%  P&L=$62,323  avg=+$278.2
  Wed  194 trades  WR=98%  P&L=$50,312  avg=+$259.3
  Thu  219 trades  WR=96%  P&L=$102,412  avg=+$467.6
  Fri  225 trades  WR=96%  P&L=$79,402  avg=+$352.9
  Sat    1 trades  WR=100%  P&L=$3,906  avg=+$3906.1

Signal component analysis — ALL (1047 trades):
  Component          Avg(Win)   Avg(Loss)  WinCorr    PnlCorr   
  position_score     0.942      0.941      +0.002     +0.026       
  energy_score       0.712      0.779      -0.028     +0.113       
  entropy_score      0.701      0.683      +0.012     +0.114       
  confluence_score   0.808      0.771      +0.040     -0.001       
  timing_score       0.213      0.132      +0.045     -0.129       
  channel_health     0.414      0.418      -0.005     -0.129       
  confidence         0.629      0.645      -0.023     +0.129       

Signal component analysis — BOUNCE (384 trades):
  Component          Avg(Win)   Avg(Loss)  WinCorr    PnlCorr   
  position_score     0.930      0.905      +0.020     +0.030       
  energy_score       0.256      0.296      -0.016     -0.060       
  entropy_score      0.469      0.232      +0.114     +0.012     **
  confluence_score   0.851      0.838      +0.014     +0.029       
  timing_score       0.575      0.513      +0.033     -0.011       
  channel_health     0.521      0.526      -0.008     -0.063       
  confidence         0.541      0.545      -0.007     -0.003       

Signal component analysis — BREAK (663 trades):
  Component          Avg(Win)   Avg(Loss)  WinCorr    PnlCorr   
  position_score     0.950      0.953      -0.010     +0.008       
  energy_score       0.981      0.947      +0.105     +0.040     **
  entropy_score      0.838      0.839      -0.003     +0.049       
  confluence_score   0.782      0.748      +0.039     +0.031       
  channel_health     0.351      0.380      -0.068     -0.024       
  confidence         0.681      0.679      +0.004     +0.070       

============================================================
SIGNAL FILTER DIAGNOSTICS (1369 total physics signals)
============================================================
  not_buy_sell                     4511
  total_signals                    1369
  peak_boost                       1078
  eq_scale                         1078
  quad_eq_scale                    1078
  ch_health_sizing                 1078
  conf_score_sizing                1078
  cycle_size                       1078
  trade_decay                      1078
  near_ath                         1073
  dyn_cap_hot                      1042
  streak_accel                      966
  big_win_regime                    963
  equity_delever                    962
  eq_growth_mom                     962
  exp_win_streak                    901
  drought_boost                     854
  low_var_boost                     802
  high_mfe_regime                   801
  hot_streak                        761
  confluence_boost                  740
  long_streak                       736
  break_hard_reduce                 694
  break_aligned_te                  694
  brk_univ                          694
  break_low_rev                     685
  break_freq_reduce                 682
  momentum_confirmed                675
  brk_sub500                        663
  break_dir_align                   645
  triple_eq                         618
  perfect_wr                        553
  big_opp_history                   553
  pnl_accel                         547
  extreme_break                     490
  trade_eff                         477
  trail_momentum                    463
  pnl_trend_up                      456
  accel_wins                        434
  parity_reduce                     429
  amp_decline                       424
  eq_accel                          418
  high_ke_break_boost               411
  bounce_sized_up                   384
  wide_ch_bounce                    384
  quality_scored                    384
  comprehensive_score               384
  comprehensive_v2                  384
  exp_score                         384
  quad_be                           384
  composite_q                       384
  pos_extremity                     384
  wide_trend                        384
  width_theta                       384
  triple_qual                       384
  wsh_sizing                        384
  mom_width                         384
  conf_pos_prod                     384
  tp_narrow                         384
  trail_stop_compound               384
  brk_lowconf                       383
  w_break_prob                      382
  all_established                   378
  trend_break_boost                 377
  vol_confirm                       371
  clean_brk_20                      367
  multi_tf_be_cont                  366
  ou_trend_break                    364
  p25_be                            363
  break_sell_boost                  361
  high_rr_ratio                     360
  timing_boost                      358
  median_be                         356
  weighted_be                       354
  high_bc_bounce                    346
  low_vol_bounce                    343
  p75_be                            342
  fast_vs_mature                    342
  max_pe_boost                      335
  max_pe_loaded                     335
  extreme_pe                        335
  clean_exit_bounce                 335
  slope_aligned_break               334
  pos_score_boost                   324
  good_half_life                    323
  quad_ke_boost                     322
  extreme_pos_sig                   317
  pnl_accel_boost                   316
  late_funded                       316
  mid_recovered                     306
  high_te_break                     304
  weighted_ke_boost                 298
  low_volume                        297
  high_pe_bounce                    297
  multi_ke_boost                    294
  te_health                         294
  multi_tf_edge_boost               293
  late_established                  293
  narrow_break_skip                 289
  mature_system                     282
  mfe_mae_ratio                     281
  long_osc_reduce                   280
  median_pe_boost                   279
  wide_dz                           277
  low_bp_bounce                     276
  final_stretch                     275
  bc_tight                          271
  harmonic_bounce                   266
  wide_bounce_boost                 262
  low_energy_bounce                 260
  low_pnl_vol                       255
  w_energy_prod                     252
  contrarian_boost                  250
  be_cont_penalty                   248
  all_tf_low_pe                     246
  high_pe_boost                     245
  min_ke_penalty                    241
  weighted_health                   224
  edge_bounce_boost                 223
  anti_cluster                      219
  double_weak                       219
  free_spring                       218
  fm_boost                          216
  p25_pe                            213
  4fq_avg                           207
  ch_break_sizeup                   207
  5f_best                           205
  slope_align_bounce                204
  osc_timing                        194
  sys_total_e                       194
  pos_n3                            186
  ent_pos_penalty                   185
  lo_ent_sig                        185
  counter_trend_boost               184
  max_vol_conf                      184
  p90_be                            169
  lev_cap                           166
  p25_ke                            165
  primary_best                      164
  multi_tf_pe_boost                 164
  hi_avg_pe                         164
  median_edge                       158
  low_conf_reduce                   155
  low_conf                          155
  danger_zone                       151
  lossless_regime                   144
  sys_spring                        141
  diversity_reduce                  139
  pnl_trend                         137
  near_bounce_timing                127
  dir_consensus                     125
  pe_vol_h                          123
  edge_vol                          122
  free_pe                           120
  same_dir_reduce                   117
  free_spr_edge                     117
  theta_pe_prod                     114
  hot_zone_boost                    113
  all_qfit                          112
  trade_usd_cap                     108
  spring_edge                       106
  min_health_ok                     106
  good_regime_bounce                105
  loss_cluster                      105
  low_ke_reduce                     101
  geo_health                         99
  mom_turn_reduce                    98
  very_hi_pe                         97
  high_be_reduce                     96
  all_energized                      95
  high_theta_bounce                  95
  max_theta_boost                    95
  theta_var                          95
  high_rsq_bounce                    94
  exposure_cap                       93
  fast_hl_bounce                     90
  channel_mature                     84
  ke_consensus                       83
  high_vol_break                     83
  mom_dir_consensus                  81
  same_type_loss                     77
  geo_ke                             76
  momentum_break                     76
  premium_sig                        75
  best_prh                           74
  chaos_mom_bounce                   73
  spring_eq_400                      73
  low_mae_regime                     71
  hi_energy_sig                      70
  extreme_pos_boost                  64
  break_after_stop                   64
  ou_revert_break                    63
  buy_bull_mom                       60
  spring_bounce_boost                59
  slope_consensus                    59
  vol_pos_edge                       55
  4f_q_avg                           54
  geo_hpr                            54
  sell_all_down                      50
  wide_tp_bounce                     49
  early_conf_brk                     48
  timed_eq                           46
  extreme_gw_gl                      43
  te5_300k                           43
  wt_geo_peh                         42
  mid_ch_reduce                      42
  poor_tf                            41
  te_low_cv                          39
  daily_pnl_cap                      39
  extreme_dual                       38
  early_dz                           38
  post_loss_reduce                   37
  loss_cooldown                      37
  4f_evidence                        37
  ordered_energy                     34
  safe_support                       33
  multi_tf_be_penalty                33
  bounce_stop_pen                    33
  loss_growing                       32
  theta_decr                         31
  bp_asym_align                      31
  mom_aligned_boost                  30
  brk_weak_pos                       28
  all_extreme                        28
  bounce_low_pe                      26
  anti_revenge                       26
  dyn_cap_normal                     26
  severe_cluster                     26
  conf_energy                        25
  loss_proximity                     25
  mom_turning                        24
  post_stop_reduce                   23
  dir_unanim                         23
  low_conf_buy_bounce                22
  ch_break_sizedown                  21
  hot_momentum                       20
  hi_conf_bounce                     19
  bounce_high_be                     19
  late_compound                      18
  worst_bounce_reduce                16
  5f_geo_full                        15
  6f_geo                             13
  ke_monotonic                       13
  vol_health_pos                     11
  pnl_vol_spike                      11
  dead_zone_reduce                   10
  ordered_ke                          9
  pe_be_ratio                         8
  te_uniform                          7
  full_energy                         6
  all_tf_floor                        6
  big_loss_reduce                     6
  sideways_reduce                     6
  big_win_mom                         3
  anti_pyramid                        2
  timeout_avoid                       2
  steep_slope                         2
  hot_bounce_10                       2
  timed_conf                          1
  strong_revert                       1
  consec_loss_reduce                  1
  q_streak_3                          1
  pe_pos_rev                          1
  geo_vep                             1
  trail_loss_cluster                  1
  TRADES TAKEN                     1047
============================================================

  2025: baseline 1047 trades $340,115 | ML-sized 1047 trades $339,782 | delta $-333 (113.1s)

======================================================================
AGGREGATE RESULTS
======================================================================
                         Trades       WR       PF      Total P&L    Avg P&L   Max DD
--------------------------------------------------------------------------
Baseline                  1,047    96.7%   224.75       $340,115       $325     0.5%
ML-sized (tiers)          1,047    96.7%   224.35       $339,782       $325     0.5%

======================================================================
DELTA ANALYSIS
======================================================================
  Trade count delta:  +0
  P&L delta:          $-333 (-0.1%)
  WR delta:           +0.0pp
  PF delta:           -0.40
  Max DD delta:       +0.0pp

======================================================================
POST-HOC COMPARISON
======================================================================
  Post-hoc estimate:  $16,136,048
  Real backtest:      $339,782
  Gap:                $-15,796,266 (-97.9%)
  Verdict:            ERODED — leverage caps reduced 97.9% of gains

======================================================================
PER-YEAR BREAKDOWN
======================================================================
Year   B.Trades M.Trades        B.P&L        M.P&L        Delta   Delta%
------------------------------------------------------------------------
2025      1,047    1,047     $340,115     $339,782        $-333    -0.1%

================================================================================
Arch416 — 11yr validation (2015-2025) bounce_cap=12x  [2026-02-23]
================================================================================
Change: Mon/Tue/Wed/Fri DOW boosts 1.35x → 1.40x (closing gap toward Thu 1.45x)
Result: REVERTED — identical to Arch415 ($7,785,562 baseline). Cap is binding.

AGGREGATE (11yr, $100K/yr starting capital):
  Baseline:        11,578 trades | WR=94.4% | PF=105.42 | $7,785,562 | Sharpe=2.43 | MaxDD=3.5%
  Old tiers:       11,572 trades | WR=94.4% | PF=105.39 | $7,785,954 | Sharpe=2.45 | +$393  (+0.0%)
  Upscale-only:    11,571 trades | WR=94.5% | PF=103.38 | $7,844,188 | Sharpe=2.44 | +$58,626 (+0.8%)

FINDING: At bounce_cap=12x, DOW multipliers >1.35x are fully absorbed by the cap.
         Arch415 (1.35x Mon/Tue/Wed/Fri, 1.45x Thu) is the settled architecture.
         Further gains require a different approach (not DOW/TOD multipliers).

Log: v15/validation/c8_416_11yr.log
