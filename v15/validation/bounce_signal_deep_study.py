#!/usr/bin/env python3
"""
Deep bounce signal study v2: enhanced factor analysis with additional tech
indicators and channel window sensitivity testing.

Builds on bounce_signal_study.py by adding:
  - MACD histogram (daily + weekly)
  - Bollinger %B (daily + weekly)
  - Stochastic %K (daily)
  - ATR (normalized) for volatility regime
  - Volume indicators (OBV trend, relative volume)
  - Channel window sensitivity (test 40/50/60/70/80)
  - Multi-factor combination scoring
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.tf_state_backtest import (
    load_all_tfs, compute_daily_states, _norm_cols, _lin_reg_channel,
)
from v15.validation.bounce_timing import _compute_rsi
from v15.data.native_tf import fetch_native_tf
from v15.features.utils import (
    calc_macd, calc_bollinger_bands, calc_stochastic, calc_atr, calc_ema,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _lookup_weekly(series, date):
    idx = series.index.searchsorted(date, side='right') - 1
    return float(series.iloc[idx]) if 0 <= idx < len(series) else np.nan

def _lookup_daily(series, date):
    if date in series.index:
        return float(series.loc[date])
    return np.nan

def _lookup_daily_arr(arr_series, date):
    """Lookup value from a pandas Series with datetime index."""
    if date in arr_series.index:
        return float(arr_series.loc[date])
    idx = arr_series.index.searchsorted(date, side='right') - 1
    if 0 <= idx < len(arr_series):
        return float(arr_series.iloc[idx])
    return np.nan

# ── Channel window sensitivity ───────────────────────────────────────────────

def compute_pos_pct_multiwindow(close_arr, windows=[40, 50, 60, 70, 80]):
    """Compute pos_pct for multiple window sizes. Returns dict {window: pos_pct}."""
    results = {}
    for w in windows:
        if len(close_arr) >= w:
            pos, r2, width = _lin_reg_channel(close_arr[-w:])
            results[w] = {'pos_pct': pos, 'r2': r2, 'width': width}
        else:
            results[w] = {'pos_pct': np.nan, 'r2': np.nan, 'width': np.nan}
    return results


def main():
    start, end = '2016-01-01', '2026-12-31'

    print("Loading data...")
    tf_data = load_all_tfs('data/TSLAMin.txt', start, end)
    daily_df = tf_data['daily']
    weekly_df = tf_data['weekly']

    # ── Precompute all indicators ─────────────────────────────────────────────
    print("Computing indicators...")

    # SPY
    spy = _norm_cols(fetch_native_tf('SPY', 'daily', start, end))
    spy.index = pd.to_datetime(spy.index).tz_localize(None)
    spy_rsi = _compute_rsi(spy['close'], 14)

    # TSLA RSI
    tsla_rsi_d = _compute_rsi(daily_df['close'], 14)
    tsla_rsi_w = _compute_rsi(weekly_df['close'], 14)
    tsla_rsi_w_sma = tsla_rsi_w.rolling(14, min_periods=1).mean()
    tsla_52w_sma = daily_df['close'].rolling(252, min_periods=60).mean()

    # MACD (daily)
    d_close = daily_df['close'].values
    macd_line_d, macd_sig_d, macd_hist_d = calc_macd(d_close)
    macd_hist_d_series = pd.Series(macd_hist_d, index=daily_df.index)
    macd_line_d_series = pd.Series(macd_line_d, index=daily_df.index)

    # MACD (weekly)
    w_close = weekly_df['close'].values
    macd_line_w, macd_sig_w, macd_hist_w = calc_macd(w_close)
    macd_hist_w_series = pd.Series(macd_hist_w, index=weekly_df.index)

    # Bollinger Bands (daily)
    bb_upper_d, bb_mid_d, bb_lower_d = calc_bollinger_bands(d_close, 20, 2.0)
    bb_width_d = bb_upper_d - bb_lower_d
    bb_pctb_d = np.where(bb_width_d > 0, (d_close - bb_lower_d) / bb_width_d, 0.5)
    bb_pctb_d_series = pd.Series(bb_pctb_d, index=daily_df.index)
    bb_width_d_series = pd.Series(bb_width_d / d_close, index=daily_df.index)  # normalized

    # Bollinger Bands (weekly)
    bb_upper_w, bb_mid_w, bb_lower_w = calc_bollinger_bands(w_close, 20, 2.0)
    bb_width_w = bb_upper_w - bb_lower_w
    bb_pctb_w = np.where(bb_width_w > 0, (w_close - bb_lower_w) / bb_width_w, 0.5)
    bb_pctb_w_series = pd.Series(bb_pctb_w, index=weekly_df.index)

    # Stochastic (daily)
    stoch_k, stoch_d = calc_stochastic(
        daily_df['high'].values, daily_df['low'].values, d_close
    )
    stoch_k_series = pd.Series(stoch_k, index=daily_df.index)
    stoch_d_series = pd.Series(stoch_d, index=daily_df.index)

    # ATR (daily, normalized by close)
    atr_d = calc_atr(daily_df['high'].values, daily_df['low'].values, d_close, 14)
    atr_pct_d = atr_d / d_close  # normalized ATR
    atr_pct_d_series = pd.Series(atr_pct_d, index=daily_df.index)

    # Volume: relative to 20-day average
    if 'volume' in daily_df.columns:
        vol_sma20 = daily_df['volume'].rolling(20, min_periods=5).mean()
        rel_volume = daily_df['volume'] / vol_sma20
    else:
        rel_volume = pd.Series(1.0, index=daily_df.index)

    # OBV trend (20-day OBV slope)
    if 'volume' in daily_df.columns:
        delta_close = daily_df['close'].diff()
        obv = (daily_df['volume'] * np.sign(delta_close)).cumsum()
        obv_sma20 = obv.rolling(20, min_periods=5).mean()
        obv_trend = (obv - obv_sma20) / (obv_sma20.abs() + 1e-10)
    else:
        obv_trend = pd.Series(0.0, index=daily_df.index)

    # EMA 9/21 relationship (daily) for short-term trend
    ema9 = pd.Series(calc_ema(d_close, 9), index=daily_df.index)
    ema21 = pd.Series(calc_ema(d_close, 21), index=daily_df.index)
    ema_cross = (ema9 - ema21) / daily_df['close']  # normalized spread

    # ── TF states ─────────────────────────────────────────────────────────────
    trading_dates = daily_df.index
    state_rows = compute_daily_states(tf_data, trading_dates, warmup_bars=260)

    # ── Measure forward returns ───────────────────────────────────────────────
    def measure_forward(date, horizons=[1, 2, 3, 5, 10, 20]):
        if date not in daily_df.index:
            return None
        idx = daily_df.index.get_loc(date)
        ref_close = daily_df.iloc[idx]['close']
        if idx + 1 >= len(daily_df):
            return None
        entry_price = daily_df.iloc[idx + 1]['open']

        result = {'ref_close': ref_close, 'entry_open': entry_price}
        for h in horizons:
            end_idx = idx + 1 + h
            if end_idx >= len(daily_df):
                result[f'fwd_{h}d'] = np.nan
                result[f'dd_{h}d'] = np.nan
                result[f'peak_{h}d'] = np.nan
                continue
            fwd_slice = daily_df.iloc[idx+1:end_idx+1]
            exit_close = fwd_slice.iloc[-1]['close']
            result[f'fwd_{h}d'] = (exit_close / entry_price - 1) * 100
            result[f'dd_{h}d'] = (fwd_slice['low'].min() / entry_price - 1) * 100
            result[f'peak_{h}d'] = (fwd_slice['high'].max() / entry_price - 1) * 100
        return result

    # ── Scan for oversold conditions ──────────────────────────────────────────
    THRESHOLD = 0.35
    COOLDOWN = 10

    events = []
    last_fire_date = None

    for row in state_rows:
        date = row['date']
        d = row.get('daily')
        w = row.get('weekly')
        m = row.get('monthly')
        if not d or not w:
            continue

        d_pos = d['pos_pct']
        w_pos = w['pos_pct']

        if d_pos >= THRESHOLD or w_pos >= THRESHOLD:
            continue

        # Cooldown
        if last_fire_date and (date - last_fire_date).days < COOLDOWN:
            continue

        # Lookup all indicators
        spy_val = np.nan
        idx = spy_rsi.index.searchsorted(date)
        if 0 < idx <= len(spy_rsi):
            spy_val = float(spy_rsi.iloc[idx - 1])

        tw_rsi = _lookup_weekly(tsla_rsi_w, date)
        tw_sma = _lookup_weekly(tsla_rsi_w_sma, date)
        td_rsi = _lookup_daily(tsla_rsi_d, date)

        dist_52w = 0.0
        if date in tsla_52w_sma.index and date in daily_df.index:
            sma_val = tsla_52w_sma.loc[date]
            close_val = daily_df.loc[date, 'close']
            dist_52w = (close_val - sma_val) / sma_val if sma_val > 0 else 0.0

        # New indicators
        macd_hist_val = _lookup_daily_arr(macd_hist_d_series, date)
        macd_line_val = _lookup_daily_arr(macd_line_d_series, date)
        macd_hist_w_val = _lookup_weekly(macd_hist_w_series, date)
        bb_pctb_val = _lookup_daily_arr(bb_pctb_d_series, date)
        bb_width_val = _lookup_daily_arr(bb_width_d_series, date)
        bb_pctb_w_val = _lookup_weekly(bb_pctb_w_series, date)
        stoch_k_val = _lookup_daily_arr(stoch_k_series, date)
        stoch_d_val = _lookup_daily_arr(stoch_d_series, date)
        atr_pct_val = _lookup_daily_arr(atr_pct_d_series, date)
        rel_vol_val = _lookup_daily_arr(rel_volume, date)
        obv_trend_val = _lookup_daily_arr(obv_trend, date)
        ema_cross_val = _lookup_daily_arr(ema_cross, date)

        # Multi-window channel pos_pct (daily)
        if date in daily_df.index:
            di = daily_df.index.get_loc(date)
            prices = daily_df['close'].iloc[max(0, di-90):di+1].values.astype(float)
            mw = compute_pos_pct_multiwindow(prices)
        else:
            mw = {}

        fwd = measure_forward(date)
        if fwd is None:
            continue

        m_pos = m['pos_pct'] if m else np.nan

        event = {
            'date': date,
            'close': fwd['ref_close'],
            'entry': fwd['entry_open'],
            'd_pos': d_pos,
            'w_pos': w_pos,
            'm_pos': m_pos,
            'd_turning': d['is_turning'],
            'w_turning': w['is_turning'],
            'd_at_bottom': d.get('at_bottom', False),
            'energy_ratio': d.get('energy_ratio', 1.0),
            'spy_rsi': spy_val,
            'tsla_rsi_d': td_rsi,
            'tsla_rsi_w': tw_rsi,
            'tsla_rsi_w_sma': tw_sma,
            'rsi_w_above_sma': tw_rsi > tw_sma if not np.isnan(tw_rsi) and not np.isnan(tw_sma) else False,
            'dist_52w': dist_52w,
            # New indicators
            'macd_hist_d': macd_hist_val,
            'macd_line_d': macd_line_val,
            'macd_hist_w': macd_hist_w_val,
            'bb_pctb_d': bb_pctb_val,
            'bb_width_d': bb_width_val,
            'bb_pctb_w': bb_pctb_w_val,
            'stoch_k': stoch_k_val,
            'stoch_d': stoch_d_val,
            'stoch_cross': stoch_k_val > stoch_d_val if not np.isnan(stoch_k_val) and not np.isnan(stoch_d_val) else False,
            'atr_pct': atr_pct_val,
            'rel_volume': rel_vol_val,
            'obv_trend': obv_trend_val,
            'ema_cross': ema_cross_val,
            'ema9_above_21': ema_cross_val > 0 if not np.isnan(ema_cross_val) else False,
            # Multi-window
            **{f'pos_{w}': mw.get(w, {}).get('pos_pct', np.nan) for w in [40, 50, 60, 70, 80]},
            **{f'r2_{w}': mw.get(w, {}).get('r2', np.nan) for w in [40, 50, 60, 70, 80]},
            # Forward returns
            **{k: v for k, v in fwd.items() if k not in ('ref_close', 'entry_open')},
        }
        events.append(event)
        last_fire_date = date

    df = pd.DataFrame(events)
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    n = len(df)

    print(f"\n{'='*120}")
    print(f"DEEP BOUNCE SIGNAL STUDY: {n} oversold events (d_pos<{THRESHOLD} & w_pos<{THRESHOLD}, {COOLDOWN}d cooldown)")
    print(f"Period: {start} to {end}")
    print(f"Entry: next day's open after signal fires")
    print(f"{'='*120}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 1: Aggregate stats
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("SECTION 1: AGGREGATE FORWARD RETURNS")
    print(f"{'='*120}")

    for h in [1, 2, 3, 5, 10, 20]:
        col = f'fwd_{h}d'
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        wr = (valid > 0).mean() * 100
        avg = valid.mean()
        med = valid.median()
        print(f"  {h:>2}d forward:  WR={wr:.0f}%  avg={avg:+.1f}%  median={med:+.1f}%  "
              f"best={valid.max():+.1f}%  worst={valid.min():+.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 2: Winner vs Loser factor comparison (ALL indicators)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("SECTION 2: FACTOR ANALYSIS — Winners (5d > +2%) vs Losers (5d < 0%)")
    print(f"{'='*120}")

    winners = df[df['fwd_5d'] > 2.0]
    losers = df[df['fwd_5d'] < 0.0]

    print(f"\n  Winners: {len(winners)} ({len(winners)/n*100:.0f}%)  |  "
          f"Losers: {len(losers)} ({len(losers)/n*100:.0f}%)")

    cont_factors = [
        'd_pos', 'w_pos', 'tsla_rsi_d', 'tsla_rsi_w', 'spy_rsi', 'dist_52w',
        'energy_ratio',
        # New
        'macd_hist_d', 'macd_line_d', 'macd_hist_w',
        'bb_pctb_d', 'bb_width_d', 'bb_pctb_w',
        'stoch_k', 'stoch_d',
        'atr_pct', 'rel_volume', 'obv_trend', 'ema_cross',
    ]
    bool_factors = ['d_turning', 'w_turning', 'rsi_w_above_sma', 'd_at_bottom',
                    'stoch_cross', 'ema9_above_21']

    print(f"\n  {'Factor':<20} {'Winners':>10} {'Losers':>10} {'All':>10} {'W-L':>8}  Signal")
    print(f"  {'─'*80}")

    for f in cont_factors:
        if f not in df.columns:
            continue
        w_avg = winners[f].mean() if len(winners) > 0 else np.nan
        l_avg = losers[f].mean() if len(losers) > 0 else np.nan
        a_avg = df[f].mean()
        diff = w_avg - l_avg if not np.isnan(w_avg) and not np.isnan(l_avg) else 0
        arrow = '<<<' if abs(diff) > 0.1 * (abs(a_avg) + 0.001) else ''
        print(f"  {f:<20} {w_avg:>10.3f} {l_avg:>10.3f} {a_avg:>10.3f} {diff:>+7.3f}  {arrow}")

    print()
    for f in bool_factors:
        if f not in df.columns:
            continue
        w_pct = winners[f].mean() * 100 if len(winners) > 0 else 0
        l_pct = losers[f].mean() * 100 if len(losers) > 0 else 0
        a_pct = df[f].mean() * 100
        diff = w_pct - l_pct
        arrow = '<<<' if abs(diff) > 15 else ''
        print(f"  {f:<20} {w_pct:>9.0f}% {l_pct:>9.0f}% {a_pct:>9.0f}% {diff:>+6.0f}pp  {arrow}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 3: New indicator breakdowns
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("SECTION 3: NEW INDICATOR BREAKDOWNS (5d forward returns)")
    print(f"{'='*120}")

    def print_bucket_analysis(title, col, buckets):
        print(f"\n  {title}:")
        for lo, hi, label in buckets:
            bucket = df[(df[col] >= lo) & (df[col] < hi)]
            if len(bucket) == 0:
                continue
            fwd5 = bucket['fwd_5d'].dropna()
            wr = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
            print(f"    {label:<25}:  n={len(bucket):>3}  "
                  f"5d avg={fwd5.mean():+.1f}%  WR={wr:.0f}%  "
                  f"peak={bucket['peak_5d'].mean():+.1f}%  dd={bucket['dd_5d'].mean():+.1f}%")

    # MACD histogram (daily)
    print_bucket_analysis("MACD Histogram (daily)", 'macd_hist_d', [
        (-999, -3, 'Deep negative (<-3)'),
        (-3, -1, 'Negative (-3 to -1)'),
        (-1, 0, 'Slightly negative (-1 to 0)'),
        (0, 999, 'Positive (recovering)'),
    ])

    # MACD histogram (weekly)
    print_bucket_analysis("MACD Histogram (weekly)", 'macd_hist_w', [
        (-999, -5, 'Deep negative (<-5)'),
        (-5, -1, 'Negative (-5 to -1)'),
        (-1, 0, 'Slightly negative (-1 to 0)'),
        (0, 999, 'Positive'),
    ])

    # Bollinger %B (daily)
    print_bucket_analysis("Bollinger %B (daily)", 'bb_pctb_d', [
        (-999, 0, 'Below lower band (<0)'),
        (0, 0.15, 'Near bottom (0-0.15)'),
        (0.15, 0.35, 'Low (0.15-0.35)'),
        (0.35, 999, 'Mid/High (>0.35)'),
    ])

    # Bollinger %B (weekly)
    print_bucket_analysis("Bollinger %B (weekly)", 'bb_pctb_w', [
        (-999, 0.15, 'Near bottom (<0.15)'),
        (0.15, 0.35, 'Low (0.15-0.35)'),
        (0.35, 0.60, 'Mid (0.35-0.60)'),
        (0.60, 999, 'High (>0.60)'),
    ])

    # BB Width (volatility)
    print_bucket_analysis("BB Width / Close (volatility)", 'bb_width_d', [
        (0, 0.04, 'Low vol (<4%)'),
        (0.04, 0.08, 'Normal vol (4-8%)'),
        (0.08, 0.12, 'High vol (8-12%)'),
        (0.12, 999, 'Extreme vol (>12%)'),
    ])

    # Stochastic %K
    print_bucket_analysis("Stochastic %K (daily)", 'stoch_k', [
        (0, 10, '<10 (extreme oversold)'),
        (10, 20, '10-20 (oversold)'),
        (20, 35, '20-35 (weak)'),
        (35, 100, '>35 (recovering)'),
    ])

    # Stochastic cross
    print(f"\n  Stochastic %K vs %D cross:")
    for label, val in [('K > D (bullish cross)', True), ('K < D (bearish)', False)]:
        bucket = df[df['stoch_cross'] == val]
        if len(bucket) == 0:
            continue
        fwd5 = bucket['fwd_5d'].dropna()
        wr = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
        print(f"    {label:<25}:  n={len(bucket):>3}  "
              f"5d avg={fwd5.mean():+.1f}%  WR={wr:.0f}%  "
              f"peak={bucket['peak_5d'].mean():+.1f}%  dd={bucket['dd_5d'].mean():+.1f}%")

    # ATR (volatility regime)
    print_bucket_analysis("ATR / Close (volatility regime)", 'atr_pct', [
        (0, 0.025, 'Low (<2.5%)'),
        (0.025, 0.04, 'Normal (2.5-4%)'),
        (0.04, 0.06, 'High (4-6%)'),
        (0.06, 999, 'Extreme (>6%)'),
    ])

    # Relative volume
    print_bucket_analysis("Relative Volume (vs 20d avg)", 'rel_volume', [
        (0, 0.7, 'Low (<0.7x)'),
        (0.7, 1.0, 'Below avg (0.7-1.0x)'),
        (1.0, 1.5, 'Above avg (1.0-1.5x)'),
        (1.5, 999, 'High (>1.5x)'),
    ])

    # OBV trend
    print_bucket_analysis("OBV Trend (20d momentum)", 'obv_trend', [
        (-999, -0.05, 'Strong outflow (<-5%)'),
        (-0.05, 0, 'Mild outflow (-5% to 0)'),
        (0, 0.05, 'Mild inflow (0 to 5%)'),
        (0.05, 999, 'Strong inflow (>5%)'),
    ])

    # EMA 9/21 cross
    print(f"\n  EMA 9 vs EMA 21:")
    for label, val in [('EMA9 > EMA21 (bullish)', True), ('EMA9 < EMA21 (bearish)', False)]:
        bucket = df[df['ema9_above_21'] == val]
        if len(bucket) == 0:
            continue
        fwd5 = bucket['fwd_5d'].dropna()
        wr = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
        print(f"    {label:<25}:  n={len(bucket):>3}  "
              f"5d avg={fwd5.mean():+.1f}%  WR={wr:.0f}%  "
              f"peak={bucket['peak_5d'].mean():+.1f}%  dd={bucket['dd_5d'].mean():+.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 4: Channel window sensitivity
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("SECTION 4: CHANNEL WINDOW SENSITIVITY (daily TF)")
    print(f"{'='*120}")

    for w in [40, 50, 60, 70, 80]:
        col = f'pos_{w}'
        if col not in df.columns:
            continue
        valid = df[df[col].notna()]
        if len(valid) == 0:
            continue
        oversold = valid[valid[col] < 0.35]
        recovering = valid[(valid[col] >= 0.15) & (valid[col] < 0.35)]
        at_bottom = valid[valid[col] < 0.15]

        r2_col = f'r2_{w}'
        avg_r2 = valid[r2_col].mean() if r2_col in valid.columns else 0

        print(f"\n  Window={w} (avg R²={avg_r2:.3f}):")
        print(f"    Oversold (<0.35):   n={len(oversold):>3}  "
              f"5d avg={oversold['fwd_5d'].mean():+.1f}%  "
              f"WR={(oversold['fwd_5d']>0).mean()*100:.0f}%")
        if len(at_bottom) > 0:
            print(f"    At bottom (<0.15):  n={len(at_bottom):>3}  "
                  f"5d avg={at_bottom['fwd_5d'].mean():+.1f}%  "
                  f"WR={(at_bottom['fwd_5d']>0).mean()*100:.0f}%")
        if len(recovering) > 0:
            print(f"    Recovering (0.15-0.35): n={len(recovering):>3}  "
                  f"5d avg={recovering['fwd_5d'].mean():+.1f}%  "
                  f"WR={(recovering['fwd_5d']>0).mean()*100:.0f}%")

    # Compare: does window choice change which events are "oversold"?
    print(f"\n  Cross-window agreement:")
    for w in [40, 50, 70, 80]:
        col = f'pos_{w}'
        ref = 'pos_60'
        if col not in df.columns or ref not in df.columns:
            continue
        both_os = ((df[col] < 0.35) & (df[ref] < 0.35)).sum()
        only_ref = ((df[col] >= 0.35) & (df[ref] < 0.35)).sum()
        only_w = ((df[col] < 0.35) & (df[ref] >= 0.35)).sum()
        print(f"    W={w} vs W=60:  both oversold={both_os}  "
              f"only W60={only_ref}  only W{w}={only_w}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 5: Multi-factor combination analysis
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("SECTION 5: MULTI-FACTOR COMBINATIONS (5d forward returns)")
    print(f"{'='*120}")

    combos = [
        # Original factors
        ('w_turning + rsi_w>sma',
         lambda r: r['w_turning'] and r['rsi_w_above_sma']),
        ('w_turning + tsla_rsi_w 40-50',
         lambda r: r['w_turning'] and 40 <= r['tsla_rsi_w'] < 50),
        ('w_turning + d_turning',
         lambda r: r['w_turning'] and r['d_turning']),
        ('d_turning + tsla_rsi_d < 30',
         lambda r: r['d_turning'] and r['tsla_rsi_d'] < 30),
        ('w_turning + spy_rsi > 50',
         lambda r: r['w_turning'] and r['spy_rsi'] > 50),

        # New indicator combos
        ('w_turning + macd_hist_d > 0',
         lambda r: r['w_turning'] and r['macd_hist_d'] > 0),
        ('w_turning + stoch_cross (K>D)',
         lambda r: r['w_turning'] and r['stoch_cross']),
        ('w_turning + bb_pctb_d < 0.15',
         lambda r: r['w_turning'] and r['bb_pctb_d'] < 0.15),
        ('stoch_k < 20 + w_turning',
         lambda r: r['stoch_k'] < 20 and r['w_turning']),
        ('macd_hist_d turning + (loosening)',
         lambda r: -1 < r['macd_hist_d'] < 0 and r['w_turning']),

        # Three-factor combos
        ('w_turning + stoch_cross + rsi_d<35',
         lambda r: r['w_turning'] and r['stoch_cross'] and r['tsla_rsi_d'] < 35),
        ('w_turning + macd_hist>0 + rsi_w>sma',
         lambda r: r['w_turning'] and r['macd_hist_d'] > 0 and r['rsi_w_above_sma']),
        ('d_turning + w_turning + stoch_k<25',
         lambda r: r['d_turning'] and r['w_turning'] and r['stoch_k'] < 25),
        ('w_turning + bb_pctb_d<0.2 + spy_rsi>45',
         lambda r: r['w_turning'] and r['bb_pctb_d'] < 0.2 and r['spy_rsi'] > 45),
        ('w_turning + obv_trend>0 + rsi_d<35',
         lambda r: r['w_turning'] and r['obv_trend'] > 0 and r['tsla_rsi_d'] < 35),

        # Volume-based
        ('rel_volume > 1.5 + d_turning',
         lambda r: r['rel_volume'] > 1.5 and r['d_turning']),
        ('obv_trend > 0 + w_turning',
         lambda r: r['obv_trend'] > 0 and r['w_turning']),

        # Volatility-based
        ('atr_pct > 0.04 + w_turning',
         lambda r: r['atr_pct'] > 0.04 and r['w_turning']),
        ('bb_width > 0.08 + w_turning',
         lambda r: r['bb_width_d'] > 0.08 and r['w_turning']),

        # Kitchen sink (best factors)
        ('BEST: w_turn + stoch_cross + rsi_w>sma + spy>45',
         lambda r: r['w_turning'] and r['stoch_cross'] and r['rsi_w_above_sma'] and r['spy_rsi'] > 45),
        ('BEST2: w_turn + macd_hist>-1 + rsi_d<35 + spy>40',
         lambda r: r['w_turning'] and r['macd_hist_d'] > -1 and r['tsla_rsi_d'] < 35 and r['spy_rsi'] > 40),
    ]

    print(f"\n  {'Combination':<50} {'n':>4} {'WR':>5} {'5d avg':>8} {'5d med':>8} {'peak5':>7} {'dd5':>7}")
    print(f"  {'─'*95}")

    combo_results = []
    for label, fn in combos:
        mask = df.apply(fn, axis=1)
        subset = df[mask]
        n_s = len(subset)
        if n_s == 0:
            print(f"  {label:<50} {'0':>4}")
            continue
        fwd5 = subset['fwd_5d'].dropna()
        wr = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
        avg = fwd5.mean()
        med = fwd5.median()
        peak = subset['peak_5d'].dropna().mean()
        dd = subset['dd_5d'].dropna().mean()
        print(f"  {label:<50} {n_s:>4} {wr:>4.0f}% {avg:>+7.1f}% {med:>+7.1f}% "
              f"{peak:>+6.1f}% {dd:>+6.1f}%")
        combo_results.append({'label': label, 'n': n_s, 'wr': wr, 'avg': avg})

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 6: Correlation matrix (key factors vs 5d return)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("SECTION 6: FACTOR CORRELATION WITH 5d FORWARD RETURN")
    print(f"{'='*120}")

    corr_factors = [
        'd_pos', 'w_pos', 'tsla_rsi_d', 'tsla_rsi_w', 'spy_rsi', 'dist_52w',
        'energy_ratio', 'macd_hist_d', 'macd_line_d', 'macd_hist_w',
        'bb_pctb_d', 'bb_width_d', 'bb_pctb_w',
        'stoch_k', 'stoch_d', 'atr_pct', 'rel_volume', 'obv_trend', 'ema_cross',
    ]

    print(f"\n  {'Factor':<20} {'Corr w/ 5d':>12} {'Corr w/ 10d':>12} {'Direction':>12}")
    print(f"  {'─'*60}")

    corrs = []
    for f in corr_factors:
        if f not in df.columns:
            continue
        valid = df[[f, 'fwd_5d', 'fwd_10d']].dropna()
        if len(valid) < 10:
            continue
        c5 = valid[f].corr(valid['fwd_5d'])
        c10 = valid[f].corr(valid['fwd_10d'])
        direction = 'HIGHER=better' if c5 > 0 else 'LOWER=better'
        marker = ' ***' if abs(c5) > 0.15 else (' **' if abs(c5) > 0.10 else '')
        print(f"  {f:<20} {c5:>+11.3f} {c10:>+11.3f}  {direction}{marker}")
        corrs.append((f, c5))

    # Sort by absolute correlation
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  TOP 5 PREDICTORS (by |correlation|):")
    for f, c in corrs[:5]:
        print(f"    {f:<20}  corr={c:+.3f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 7: Individual events table (with new indicators)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("SECTION 7: INDIVIDUAL EVENTS (most recent 30)")
    print(f"{'='*120}")

    recent = df.tail(30)
    print(f"\n  {'Date':<11} {'Close':>7} {'D_pos':>5} {'W_pos':>5} "
          f"{'D_RSI':>5} {'W_RSI':>5} {'StK':>5} {'MACD_H':>7} "
          f"{'BB%B':>5} {'ATR%':>5} {'RelV':>5} "
          f"{'D_trn':>5} {'W_trn':>5} "
          f"{'5d%':>6} {'10d%':>6} {'Peak5':>6}")
    print('  ' + '─' * 115)

    for _, e in recent.iterrows():
        print(f"  {e['date_str']:<11} {e['close']:>7.1f} "
              f"{e['d_pos']:>5.2f} {e['w_pos']:>5.2f} "
              f"{e['tsla_rsi_d']:>5.1f} {e['tsla_rsi_w']:>5.1f} "
              f"{e['stoch_k']:>5.1f} {e['macd_hist_d']:>+6.1f} "
              f"{e['bb_pctb_d']:>5.2f} {e['atr_pct']:>5.3f} {e['rel_volume']:>5.2f} "
              f"{'Y' if e['d_turning'] else 'N':>5} {'Y' if e['w_turning'] else 'N':>5} "
              f"{e['fwd_5d']:>+5.1f}% {e['fwd_10d']:>+5.1f}% {e['peak_5d']:>+5.1f}%")

    # ═══════════════════════════════════════════════════════════════════════════
    # SECTION 8: Existing factor breakdowns (enhanced)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*120}")
    print("SECTION 8: EXISTING FACTOR BREAKDOWNS (enhanced)")
    print(f"{'='*120}")

    # RSI regimes
    print_bucket_analysis("TSLA Daily RSI", 'tsla_rsi_d', [
        (0, 25, '<25 (extreme)'),
        (25, 30, '25-30'),
        (30, 35, '30-35'),
        (35, 45, '35-45 (weak)'),
        (45, 100, '>45'),
    ])

    print_bucket_analysis("TSLA Weekly RSI", 'tsla_rsi_w', [
        (0, 30, '<30 (extreme)'),
        (30, 35, '30-35'),
        (35, 40, '35-40'),
        (40, 50, '40-50 (sweet spot?)'),
        (50, 100, '>50'),
    ])

    # Turning combinations
    print(f"\n  Momentum turning combinations:")
    for dt, wt, label in [(True, True, 'Both turning'),
                          (True, False, 'Daily only'),
                          (False, True, 'Weekly only'),
                          (False, False, 'Neither')]:
        bucket = df[(df['d_turning'] == dt) & (df['w_turning'] == wt)]
        if len(bucket) == 0:
            continue
        fwd5 = bucket['fwd_5d'].dropna()
        wr = (fwd5 > 0).mean() * 100 if len(fwd5) > 0 else 0
        print(f"    {label:<20}:  n={len(bucket):>3}  "
              f"5d avg={fwd5.mean():+.1f}%  WR={wr:.0f}%  "
              f"peak={bucket['peak_5d'].mean():+.1f}%  dd={bucket['dd_5d'].mean():+.1f}%")

    # SPY RSI regimes
    print_bucket_analysis("SPY RSI (market regime)", 'spy_rsi', [
        (0, 30, '<30 (market panic)'),
        (30, 45, '30-45 (market weak)'),
        (45, 55, '45-55 (neutral)'),
        (55, 70, '55-70 (market strong)'),
        (70, 100, '>70 (market overbought)'),
    ])

    print_bucket_analysis("Distance from 52-week SMA", 'dist_52w', [
        (-999, -0.30, '<-30% (extreme)'),
        (-0.30, -0.20, '-30% to -20%'),
        (-0.20, -0.10, '-20% to -10%'),
        (-0.10, 0.0, '-10% to 0%'),
        (0.0, 999, 'Above SMA'),
    ])

    print(f"\n{'='*120}")
    print("STUDY COMPLETE")
    print(f"{'='*120}")


if __name__ == '__main__':
    main()
