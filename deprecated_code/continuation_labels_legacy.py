"""
DEPRECATED: Legacy continuation label generation function

This file contains the original (unoptimized) continuation label generation code.

Deprecated on: 2025-01-19
Reason: Replaced with optimized version that is 20-60x faster
Replacement: TradingFeatureExtractor.generate_continuation_labels() in src/ml/features.py

The optimized version implements:
1. Pre-resampling of entire dataframe (5-8x speedup)
2. Parallelization with joblib (2-4x speedup)
3. Batch future price window computation (2x speedup)
4. Total speedup: 20-60x while maintaining 100% identical mathematical results

This legacy code is preserved for:
- Historical reference
- Verification of mathematical equivalence
- Debugging if needed

DO NOT USE THIS CODE IN PRODUCTION - use the optimized version instead.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import config


def generate_continuation_labels_legacy(self, df: pd.DataFrame, timestamps: list, prediction_horizon: int = 24, debug: bool = False) -> pd.DataFrame:
    """
    LEGACY: Generate continuation prediction labels using multi-timeframe analysis.

    This is the original unoptimized version. Use the optimized version in
    TradingFeatureExtractor.generate_continuation_labels() instead.

    Implements the pseudo-code logic:
    - Pull 1h and 4h OHLC chunks
    - Calculate RSI for both timeframes
    - Check slope alignment
    - Apply continuation scoring
    - Look ahead for actual duration/gain

    Args:
        df: Full OHLC DataFrame (5-min bars)
        timestamps: List of timestamps to process
        prediction_horizon: How many bars ahead to look for continuation
        debug: Enable debug logging for troubleshooting

    Returns:
        DataFrame with continuation labels
    """
    labels = []
    skip_reasons = {
        'insufficient_raw_data': 0,
        'insufficient_resampled_data': 0,
        'channel_fit_failed': 0,
        'scoring_failed': 0,
        'other_errors': 0
    }

    # Progress bar for continuation label generation
    with tqdm(total=len(timestamps), desc="   Continuation labels",
              unit="timestamps", ncols=100, leave=False, ascii=True, mininterval=0.5) as pbar:

        for i, ts in enumerate(timestamps):
            try:
                # Get current index and price
                current_idx = df.index.get_loc(ts)
                current_price = df.loc[ts, 'tsla_close']

                # Step 1: Pull multi-timeframe OHLC chunks
                # 1h chunk: Use config value (default 1512 bars = ~3 months)
                one_h_lookback = min(current_idx, config.CONTINUATION_LOOKBACK_1H)
                one_h_start = max(0, current_idx - one_h_lookback)
                one_h_chunk = df.iloc[one_h_start:current_idx+1].copy()

                # 4h chunk: Use config value (default 6048 bars = ~1 year)
                four_h_lookback = min(current_idx, config.CONTINUATION_LOOKBACK_4H)
                four_h_start = max(0, current_idx - four_h_lookback)
                four_h_chunk = df.iloc[four_h_start:current_idx+1].copy()

                # DEBUG: Log data availability for first few timestamps
                if debug and i < 5:
                    print(f"  TS {ts}: 1h_chunk={len(one_h_chunk)}, 4h_chunk={len(four_h_chunk)}")

                # Resample to 1h and 4h timeframes
                one_h_ohlc = one_h_chunk.resample('1h').agg({
                    'tsla_open': 'first',
                    'tsla_high': 'max',
                    'tsla_low': 'min',
                    'tsla_close': 'last'
                }).dropna()

                four_h_ohlc = four_h_chunk.resample('4h').agg({
                    'tsla_open': 'first',
                    'tsla_high': 'max',
                    'tsla_low': 'min',
                    'tsla_close': 'last'
                }).dropna()

                # Rename columns to generic OHLC names IMMEDIATELY after resampling
                # RSI and channel calculators expect 'close', not 'tsla_close'
                one_h_ohlc.columns = [c.replace('tsla_', '') for c in one_h_ohlc.columns]
                four_h_ohlc.columns = [c.replace('tsla_', '') for c in four_h_ohlc.columns]

                # DEBUG: Verify column renaming worked
                if debug and i < 3:
                    print(f"  DEBUG: 1h columns after rename: {one_h_ohlc.columns.tolist()}")
                    print(f"  DEBUG: 4h columns after rename: {four_h_ohlc.columns.tolist()}")

                if len(one_h_ohlc) < 3 or len(four_h_ohlc) < 2:
                    skip_reasons['insufficient_resampled_data'] += 1
                    if debug and skip_reasons['insufficient_resampled_data'] < 3:
                        print(f"  SKIP {ts}: Insufficient resampled data ({len(one_h_ohlc)} 1h, {len(four_h_ohlc)} 4h)")
                    pbar.update(1)
                    continue

                # Step 2: Compute RSI for both timeframes
                rsi_1h = self.rsi_calc.get_rsi_data(one_h_ohlc).value or 50.0
                rsi_4h = self.rsi_calc.get_rsi_data(four_h_ohlc).value or 50.0

                # Step 3: Fit channels and get slopes (allow using more data)
                channel_1h = self.channel_calc.find_optimal_channel_window(
                    one_h_ohlc, timeframe='1h', max_lookback=min(60, max(5, len(one_h_ohlc)-2)), min_ping_pongs=2
                )

                channel_4h = self.channel_calc.find_optimal_channel_window(
                    four_h_ohlc, timeframe='4h', max_lookback=min(120, max(10, len(four_h_ohlc)-2)), min_ping_pongs=2
                )

                # Get slopes (default to 0 if no channel found)
                slope_1h = channel_1h.slope if channel_1h else 0.0
                slope_4h = channel_4h.slope if channel_4h else 0.0

                # Check if channel fitting failed
                if channel_1h is None or channel_4h is None:
                    skip_reasons['channel_fit_failed'] += 1
                    if debug and skip_reasons['channel_fit_failed'] < 3:
                        print(f"  SKIP {ts}: Channel fitting failed")
                    pbar.update(1)
                    continue

                # Step 4: Apply continuation scoring logic
                score = 0

                # +1 for low RSI on short frame (room to run upward)
                if rsi_1h < 40:
                    score += 1

                # +1 for low RSI on long frame (broader support)
                if rsi_4h < 40:
                    score += 1

                # +1 if slopes align (both bull or both bear)
                slope_1h_direction = 1 if slope_1h > 0.0001 else (-1 if slope_1h < -0.0001 else 0)
                slope_4h_direction = 1 if slope_4h > 0.0001 else (-1 if slope_4h < -0.0001 else 0)

                if slope_1h_direction == slope_4h_direction and slope_1h_direction != 0:
                    score += 1
                elif slope_1h_direction != slope_4h_direction and slope_1h_direction != 0 and slope_4h_direction != 0:
                    # Conflict: e.g., 1h bull vs 4h bear
                    score -= 1

                # -1 if overbought on higher frame (break likely)
                if rsi_4h > 70:
                    score -= 1

                # Step 5: Look ahead for actual duration/gain
                future_end = min(current_idx + prediction_horizon, len(df) - 1)
                future_prices = df.iloc[current_idx:future_end+1]['tsla_close'].values

                if len(future_prices) < 2:
                    pbar.update(1)
                    continue

                # Calculate actual continuation metrics
                future_high = np.max(future_prices)
                future_low = np.min(future_prices)
                max_gain = (future_high - current_price) / current_price * 100

                # Find break point (first time price moves >2% from entry)
                break_threshold = current_price * 1.02  # +2%
                break_indices = np.where(future_prices >= break_threshold)[0]

                if len(break_indices) > 0:
                    break_idx = break_indices[0]
                    actual_duration_hours = break_idx * 5 / 60  # 5-min bars to hours
                    continues = True
                    label = f"continues {actual_duration_hours:.1f}h, +{max_gain:.1f}%"
                else:
                    # No break within horizon
                    actual_duration_hours = len(future_prices) * 5 / 60
                    continues = True
                    label = f"continues {actual_duration_hours:.1f}h, +{max_gain:.1f}%"

                # If score <= 1, mark as breaking soon regardless
                if score <= 1:
                    skip_reasons['scoring_failed'] += 1
                    continues = False
                    early_break_hours = min(actual_duration_hours * 0.5, 1.0)  # Assume breaks in half the time
                    label = f"breaks in {early_break_hours:.1f}h"

                    if debug and skip_reasons['scoring_failed'] < 3:
                        print(f"  SKIP {ts}: Scoring failed (score={score})")
                    pbar.update(1)
                    continue

                # Calculate confidence based on score
                confidence = min(max(abs(score) * 0.2, 0.1), 0.9)

                labels.append({
                    'timestamp': ts,
                    'label': label,
                    'continues': float(continues),
                    'duration_hours': actual_duration_hours,
                    'projected_gain': max_gain if continues else 0.0,
                    'confidence': confidence,
                    'score': score,
                    'rsi_1h': rsi_1h,
                    'rsi_4h': rsi_4h,
                    'slope_1h': slope_1h,
                    'slope_4h': slope_4h
                })

                # Update progress bar with current stats
                processed = len(labels)
                success_rate = processed / (i + 1) * 100
                pbar.set_postfix({
                    'processed': f'{processed}/{len(timestamps)}',
                    'success_rate': f'{success_rate:.1f}%'
                })

            except Exception as e:
                skip_reasons['other_errors'] += 1
                if debug and skip_reasons['other_errors'] < 3:
                    print(f"  ERROR {ts}: {str(e)}")
                # Still update progress on errors
                processed = len(labels)
                success_rate = processed / (i + 1) * 100 if i > 0 else 0
                pbar.set_postfix({
                    'processed': f'{processed}/{len(timestamps)}',
                    'success_rate': f'{success_rate:.1f}%'
                })

            pbar.update(1)

    # Final debug summary
    if debug:
        total_skipped = sum(skip_reasons.values())
        print(f"  DEBUG: Generated {len(labels)} labels, skipped {total_skipped}")
        for reason, count in skip_reasons.items():
            if count > 0:
                print(f"    {reason}: {count}")

    return pd.DataFrame(labels)
