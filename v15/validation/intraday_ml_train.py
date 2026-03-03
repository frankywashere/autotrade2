#!/usr/bin/env python3
"""
Intraday ML Model Training — LightGBM classifier for intraday signal quality.

Extracts features per candidate intraday trade signal, labels as win/loss based
on actual simulation outcome, trains LightGBM binary classifier.

Validation: Walk-forward (5yr train → 1yr test) + holdout (≤2021 → 2022-2025) + 2026 OOS.

Usage:
    python -m v15.validation.intraday_ml_train
    python -m v15.validation.intraday_ml_train --tsla C:\\AI\\x14\\data\\TSLAMin.txt
"""

import os
import sys
import time
import pickle
import argparse
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.validation.intraday_v14b_janfeb import (
    load_1min, build_features, precompute_all,
    channel_position, channel_slope, compute_rsi, compute_bvc,
    sig_union_enh,
    SLIPPAGE_PCT, COMM_PER_SHARE,
)


# ---------------------------------------------------------------------------
# Feature extraction per candidate signal
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # 5-min channel features
    'cp_5m', 'rsi_5m', 'bvc_5m', 'vwap_dist', 'vol_ratio', 'vwap_slope',
    'spread_pct', 'rsi_slope', 'gap_pct',
    # Higher TF channel positions
    'cp_15m', 'cp_30m', 'cp_1h', 'cp_4h', 'cp_daily',
    # Higher TF slopes
    'slope_1h', 'slope_4h', 'slope_daily',
    # Cross-TF divergences
    'div_daily_5m', 'div_1h_5m', 'div_4h_5m',
    'div_weighted',  # weighted avg of higher TFs - 5m
    # Time features
    'hour', 'minute', 'minutes_to_close',
    'day_of_week',
    # Volatility context
    'atr_5m_pct',  # 5-min ATR as % of price
    'range_today_pct',  # today's range so far as % of price
    'volume_today_ratio',  # today's volume vs 20-day avg
    # Signal quality
    'confidence',
    'stop_pct', 'tp_pct',
    # Momentum
    'return_5bar', 'return_20bar',  # trailing 5-min returns
    'rsi_1h', 'rsi_daily',
    # Pattern features
    'bars_since_last_trade',  # cooldown pressure
    'daily_trade_count',  # how many trades today
    'consecutive_wins',  # winning streak
    'consecutive_losses',  # losing streak
]

NUM_FEATURES = len(FEATURE_NAMES)


def extract_features(i, f5m, ctx, features, signal_result, trade_state):
    """Extract feature vector for a candidate signal at bar i.

    Returns numpy array of shape (NUM_FEATURES,) or None if data insufficient.
    """
    if signal_result is None:
        return None

    _, conf, stop_pct, tp_pct = signal_result

    close = f5m['close'].values
    high = f5m['high'].values
    low = f5m['low'].values
    volume = f5m['volume'].values

    if i < 60:  # need warmup
        return None

    feat = np.full(NUM_FEATURES, np.nan)
    idx = 0

    # 5-min channel features
    feat[idx] = f5m['chan_pos'].iloc[i] if 'chan_pos' in f5m.columns else np.nan; idx += 1
    feat[idx] = f5m['rsi'].iloc[i] if 'rsi' in f5m.columns else np.nan; idx += 1
    feat[idx] = f5m['bvc'].iloc[i] if 'bvc' in f5m.columns else np.nan; idx += 1
    feat[idx] = f5m['vwap_dist'].iloc[i] if 'vwap_dist' in f5m.columns else np.nan; idx += 1
    feat[idx] = f5m['vol_ratio'].iloc[i] if 'vol_ratio' in f5m.columns else np.nan; idx += 1
    feat[idx] = f5m['vwap_slope'].iloc[i] if 'vwap_slope' in f5m.columns else np.nan; idx += 1
    feat[idx] = f5m['spread_pct'].iloc[i] if 'spread_pct' in f5m.columns else np.nan; idx += 1
    feat[idx] = f5m['rsi_slope'].iloc[i] if 'rsi_slope' in f5m.columns else np.nan; idx += 1
    feat[idx] = f5m['gap_pct'].iloc[i] if 'gap_pct' in f5m.columns else np.nan; idx += 1

    # Higher TF channel positions
    feat[idx] = ctx.get('15m_cp', np.full(1, np.nan))[i] if isinstance(ctx.get('15m_cp'), np.ndarray) and i < len(ctx['15m_cp']) else np.nan; idx += 1
    feat[idx] = ctx.get('30m_cp', np.full(1, np.nan))[i] if isinstance(ctx.get('30m_cp'), np.ndarray) and i < len(ctx['30m_cp']) else np.nan; idx += 1
    feat[idx] = ctx.get('1h_cp', np.full(1, np.nan))[i] if isinstance(ctx.get('1h_cp'), np.ndarray) and i < len(ctx['1h_cp']) else np.nan; idx += 1
    feat[idx] = ctx.get('4h_cp', np.full(1, np.nan))[i] if isinstance(ctx.get('4h_cp'), np.ndarray) and i < len(ctx['4h_cp']) else np.nan; idx += 1
    feat[idx] = ctx['daily_cp']; idx += 1

    # Higher TF slopes
    feat[idx] = ctx.get('1h_slope', np.full(1, np.nan))[i] if isinstance(ctx.get('1h_slope'), np.ndarray) and i < len(ctx['1h_slope']) else np.nan; idx += 1
    feat[idx] = ctx.get('4h_slope', np.full(1, np.nan))[i] if isinstance(ctx.get('4h_slope'), np.ndarray) and i < len(ctx['4h_slope']) else np.nan; idx += 1
    feat[idx] = ctx['daily_slope']; idx += 1

    # Cross-TF divergences
    cp5 = f5m['chan_pos'].iloc[i] if 'chan_pos' in f5m.columns else np.nan
    dcp = ctx['daily_cp']
    hcp = ctx.get('1h_cp', np.full(1, np.nan))
    h1_val = hcp[i] if isinstance(hcp, np.ndarray) and i < len(hcp) else np.nan
    h4cp = ctx.get('4h_cp', np.full(1, np.nan))
    h4_val = h4cp[i] if isinstance(h4cp, np.ndarray) and i < len(h4cp) else np.nan

    feat[idx] = dcp - cp5 if not np.isnan(dcp) and not np.isnan(cp5) else np.nan; idx += 1
    feat[idx] = h1_val - cp5 if not np.isnan(h1_val) and not np.isnan(cp5) else np.nan; idx += 1
    feat[idx] = h4_val - cp5 if not np.isnan(h4_val) and not np.isnan(cp5) else np.nan; idx += 1
    # Weighted avg divergence
    vals = [v for v in [dcp, h4_val, h1_val] if not np.isnan(v)]
    if vals and not np.isnan(cp5):
        weighted_avg = sum(v * w for v, w in zip(vals, [0.35, 0.35, 0.30][:len(vals)])) / sum([0.35, 0.35, 0.30][:len(vals)])
        feat[idx] = weighted_avg - cp5
    idx += 1

    # Time features
    bt = f5m.index[i]
    feat[idx] = bt.hour; idx += 1
    feat[idx] = bt.minute; idx += 1
    market_close = bt.replace(hour=16, minute=0, second=0)
    feat[idx] = (market_close - bt).total_seconds() / 60.0; idx += 1
    feat[idx] = bt.weekday(); idx += 1

    # Volatility context
    if i >= 20:
        hl = high[i-20:i] - low[i-20:i]
        hc = np.abs(high[i-20:i] - close[i-21:i-1])
        lc = np.abs(low[i-20:i] - close[i-21:i-1])
        tr = np.maximum(np.maximum(hl, hc), lc)
        atr = np.mean(tr)
        feat[idx] = (atr / close[i] * 100.0) if close[i] > 0 else np.nan
    idx += 1

    # Today's range as % of price
    bar_dates = [t.date() for t in f5m.index[max(0,i-100):i+1]]
    today = bt.date()
    today_bars = [j for j, d in enumerate(bar_dates) if d == today]
    if today_bars:
        start = max(0, i-100) + today_bars[0]
        end = max(0, i-100) + today_bars[-1] + 1
        today_high = np.max(high[start:end])
        today_low = np.min(low[start:end])
        feat[idx] = ((today_high - today_low) / close[i] * 100.0) if close[i] > 0 else np.nan
    idx += 1

    # Volume today vs 20-day avg
    if i >= 100 and today_bars:
        today_vol = float(np.sum(volume[max(0, i - len(today_bars)):i + 1]))
        avg_daily_vol = float(np.mean(volume[i-100:i])) * 78  # ~78 bars per day
        feat[idx] = today_vol / max(1.0, avg_daily_vol)
    idx += 1

    # Signal quality
    feat[idx] = conf; idx += 1
    feat[idx] = stop_pct; idx += 1
    feat[idx] = tp_pct; idx += 1

    # Momentum
    if i >= 5:
        feat[idx] = (close[i] - close[i-5]) / close[i-5] * 100.0 if close[i-5] > 0 else np.nan
    idx += 1
    if i >= 20:
        feat[idx] = (close[i] - close[i-20]) / close[i-20] * 100.0 if close[i-20] > 0 else np.nan
    idx += 1

    # RSI at higher TFs (from features dict if available)
    feat[idx] = np.nan  # rsi_1h — would need 1h features
    idx += 1
    feat[idx] = np.nan  # rsi_daily — would need daily features
    idx += 1

    # Trade state features
    feat[idx] = trade_state.get('bars_since_last', 999); idx += 1
    feat[idx] = trade_state.get('daily_trades', 0); idx += 1
    feat[idx] = trade_state.get('consec_wins', 0); idx += 1
    feat[idx] = trade_state.get('consec_losses', 0); idx += 1

    assert idx == NUM_FEATURES, f"Feature count mismatch: {idx} != {NUM_FEATURES}"
    return feat


# ---------------------------------------------------------------------------
# Training data generation — simulate + collect features + labels
# ---------------------------------------------------------------------------

def generate_training_data(f5m, features, precomp,
                           start_date=None, end_date=None,
                           tod_start=dt.time(13, 0), tod_end=dt.time(15, 25)):
    """Run simulation collecting features and labels for each signal.

    Returns:
        X: np.ndarray of shape (n_signals, NUM_FEATURES)
        y: np.ndarray of shape (n_signals,) — 1 for win, 0 for loss
        meta: list of dicts with entry_time, pnl, confidence
    """
    (dcp_arr, dslope_arr), htf = precomp

    pw = {'stop': 0.008, 'tp': 0.020, 'd_min': 0.20, 'h1_min': 0.15, 'f5_thresh': 0.35,
          'div_thresh': 0.20, 'vwap_thresh': -0.10, 'min_vol_ratio': 0.8}

    o_arr = f5m['open'].values
    h_arr = f5m['high'].values
    l_arr = f5m['low'].values
    c_arr = f5m['close'].values
    times = f5m.index
    tod_arr = np.array([t.time() for t in times])
    n = len(f5m)

    # Simulate to collect signal features + outcomes
    all_X = []
    all_y = []
    all_meta = []

    ctx = {'daily_cp': 0.0, 'daily_slope': 0.0, **htf}
    in_trade = False
    ep = et = None
    conf = sp = tpp = bp = 0.0
    hb = tt = 0
    cd_ = None
    ps = None  # pending signal
    ps_feat = None  # features for pending signal

    tes = dt.time(9, 35)
    tee = dt.time(15, 30)
    tfe = dt.time(15, 50)

    FLAT_SIZE = 100_000
    tb = 0.006  # trail base
    tp_power = 6  # trail power

    trade_state = {
        'bars_since_last': 999,
        'daily_trades': 0,
        'consec_wins': 0,
        'consec_losses': 0,
    }

    for i in range(n):
        bt = times[i]
        bd = bt.date()
        btod = tod_arr[i]
        o, h, l, c = o_arr[i], h_arr[i], l_arr[i], c_arr[i]

        # Date filter
        if start_date and bd < start_date:
            continue
        if end_date and bd > end_date:
            continue

        if bd != cd_:
            cd_ = bd
            tt = 0
            trade_state['daily_trades'] = 0

        trade_state['bars_since_last'] += 1

        # Execute pending signal
        if ps is not None and not in_trade:
            sc, ss, st = ps
            ps = None
            if btod >= tes and btod <= tee:
                ep = o * (1 + SLIPPAGE_PCT)
                et = bt
                conf = sc
                in_trade = True
                hb = 0
                tt += 1
                sp = ep * (1 - ss)
                tpp = ep * (1 + st)
                bp = ep
                trade_state['daily_trades'] += 1
                trade_state['bars_since_last'] = 0
                # ps_feat was set when signal was generated

        # Check exit if in trade
        if in_trade:
            hb += 1
            xp = xr = None
            bp = max(bp, h)
            trail = tb * (1.0 - conf) ** tp_power
            ts_ = bp * (1 - trail)
            if ts_ > sp:
                sp = ts_
            if l <= sp:
                xp = max(sp, l)
                xr = 'stop' if sp < ep else 'trail'
            elif h >= tpp:
                xp = tpp
                xr = 'tp'
            elif hb >= 78:
                xp = c
                xr = 'timeout'
            elif btod >= tfe:
                xp = c
                xr = 'eod'

            if xp is not None:
                xa = xp * (1 - SLIPPAGE_PCT)
                sh = max(1, int(FLAT_SIZE / ep))
                pnl = (xa - ep) * sh - COMM_PER_SHARE * sh * 2
                win = 1 if pnl > 0 else 0

                # Record sample if we have features
                if ps_feat is not None:
                    all_X.append(ps_feat)
                    all_y.append(win)
                    all_meta.append({
                        'entry_time': et,
                        'exit_time': bt,
                        'pnl': pnl,
                        'confidence': conf,
                        'exit_reason': xr,
                    })

                # Update trade state
                if win:
                    trade_state['consec_wins'] += 1
                    trade_state['consec_losses'] = 0
                else:
                    trade_state['consec_losses'] += 1
                    trade_state['consec_wins'] = 0

                in_trade = False
                ps_feat = None

        # Skip if in trade or at max daily trades
        if in_trade or tt >= 30:
            continue
        if btod < tod_start or btod > tod_end:
            continue

        # Update context
        ctx['daily_cp'] = dcp_arr[i]
        ctx['daily_slope'] = dslope_arr[i]

        # Generate signal
        result = sig_union_enh(i, f5m, ctx, pw)
        if result is not None:
            _, co, s, t = result
            ps = (co, s, t)
            # Extract features NOW (at signal time)
            ps_feat = extract_features(i, f5m, ctx, features, result, trade_state)

    X = np.array(all_X) if all_X else np.empty((0, NUM_FEATURES))
    y = np.array(all_y) if all_y else np.empty(0)

    return X, y, all_meta


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, X_val=None, y_val=None):
    """Train LightGBM binary classifier."""
    import lightgbm as lgb

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'seed': 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    callbacks = [lgb.log_evaluation(100)]

    if X_val is not None and y_val is not None:
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=train_data)
        model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks + [lgb.early_stopping(50)],
        )
    else:
        model = lgb.train(
            params, train_data,
            num_boost_round=300,
            valid_sets=[train_data],
            valid_names=['train'],
            callbacks=callbacks,
        )

    return model


def evaluate_model(model, X, y, meta, label=''):
    """Evaluate model predictions vs actual outcomes."""
    probs = model.predict(X)
    preds = (probs >= 0.5).astype(int)

    n = len(y)
    wins = int(y.sum())
    base_wr = wins / n * 100 if n > 0 else 0

    # ML-filtered results (only take trades where model predicts win)
    ml_mask = preds == 1
    ml_n = ml_mask.sum()
    ml_wins = y[ml_mask].sum() if ml_n > 0 else 0
    ml_wr = ml_wins / ml_n * 100 if ml_n > 0 else 0

    # P&L comparison
    all_pnl = sum(m['pnl'] for m in meta)
    ml_pnl = sum(m['pnl'] for i, m in enumerate(meta) if ml_mask[i])

    # Rejected trades analysis
    rej_mask = preds == 0
    rej_n = rej_mask.sum()
    rej_wins = y[rej_mask].sum() if rej_n > 0 else 0
    rej_wr = rej_wins / rej_n * 100 if rej_n > 0 else 0
    rej_pnl = sum(m['pnl'] for i, m in enumerate(meta) if rej_mask[i])

    print(f"\n  {label}")
    print(f"  {'Metric':<25} {'Baseline':>12} {'ML-Filtered':>12} {'Rejected':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Trades':<25} {n:>12,} {ml_n:>12,} {rej_n:>12,}")
    print(f"  {'Win Rate':<25} {base_wr:>11.1f}% {ml_wr:>11.1f}% {rej_wr:>11.1f}%")
    print(f"  {'Total P&L':<25} ${all_pnl:>+11,.0f} ${ml_pnl:>+11,.0f} ${rej_pnl:>+11,.0f}")
    avg_pnl = all_pnl / n if n > 0 else 0
    ml_avg = ml_pnl / ml_n if ml_n > 0 else 0
    rej_avg = rej_pnl / rej_n if rej_n > 0 else 0
    print(f"  {'Avg P&L':<25} ${avg_pnl:>+11,.0f} ${ml_avg:>+11,.0f} ${rej_avg:>+11,.0f}")

    return {
        'trades': n, 'ml_trades': int(ml_n), 'rejected': int(rej_n),
        'base_wr': round(base_wr, 1), 'ml_wr': round(ml_wr, 1), 'rej_wr': round(rej_wr, 1),
        'base_pnl': round(all_pnl), 'ml_pnl': round(ml_pnl), 'rej_pnl': round(rej_pnl),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Intraday ML Model Training')
    parser.add_argument('--tsla', default='data/TSLAMin.txt', help='Path to TSLAMin.txt')
    args = parser.parse_args()

    tsla_path = args.tsla
    for p in [tsla_path, f'../{tsla_path}', os.path.expanduser('~/Desktop/Coding/x14/data/TSLAMin.txt'),
              r'C:\AI\x14\data\TSLAMin.txt']:
        if os.path.isfile(p):
            tsla_path = p
            break

    print("=" * 70)
    print("  INTRADAY ML MODEL TRAINING")
    print("=" * 70)

    t0 = time.time()

    # Load and build features
    print("\n[1] Loading data and building features...")
    df1m = load_1min(tsla_path)
    features = build_features(df1m)
    f5m = features['5m']
    precomp = precompute_all(features, f5m)
    print(f"  5-min bars: {len(f5m):,}")
    print(f"  Date range: {f5m.index[0].date()} to {f5m.index[-1].date()}")

    # Generate training data per year
    print("\n[2] Generating training data (all signals + outcomes)...")
    all_X_by_year = {}
    all_y_by_year = {}
    all_meta_by_year = {}

    for yr in range(2016, 2027):
        start_d = dt.date(yr, 1, 1)
        end_d = dt.date(yr, 12, 31)
        X, y, meta = generate_training_data(f5m, features, precomp,
                                            start_date=start_d, end_date=end_d)
        all_X_by_year[yr] = X
        all_y_by_year[yr] = y
        all_meta_by_year[yr] = meta
        n = len(y)
        wins = int(y.sum()) if n > 0 else 0
        wr = wins / n * 100 if n > 0 else 0
        pnl = sum(m['pnl'] for m in meta) if meta else 0
        print(f"  {yr}: {n:>6,} signals, WR={wr:.1f}%, P&L=${pnl:+,.0f}")

    total = sum(len(y) for y in all_y_by_year.values())
    total_wins = sum(int(y.sum()) for y in all_y_by_year.values())
    print(f"\n  Total: {total:,} signals, {total_wins:,} wins ({total_wins/total*100:.1f}%)")

    # --- Walk-Forward Validation ---
    print("\n[3] Walk-Forward Validation (5yr train → 1yr test)...")
    print("=" * 70)

    wf_results = []
    for test_yr in range(2021, 2026):
        train_yrs = list(range(test_yr - 5, test_yr))
        X_train = np.vstack([all_X_by_year[yr] for yr in train_yrs if yr in all_X_by_year and len(all_X_by_year[yr]) > 0])
        y_train = np.concatenate([all_y_by_year[yr] for yr in train_yrs if yr in all_y_by_year and len(all_y_by_year[yr]) > 0])

        X_test = all_X_by_year.get(test_yr, np.empty((0, NUM_FEATURES)))
        y_test = all_y_by_year.get(test_yr, np.empty(0))
        meta_test = all_meta_by_year.get(test_yr, [])

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"\n  WF {train_yrs[0]}-{train_yrs[-1]} → {test_yr}: SKIPPED (no data)")
            continue

        print(f"\n  Training on {train_yrs[0]}-{train_yrs[-1]} ({len(X_train):,} samples, {y_train.sum()/len(y_train)*100:.1f}% WR)...")
        model = train_model(X_train, y_train)

        result = evaluate_model(model, X_test, y_test, meta_test,
                               f"WF {train_yrs[0]}-{train_yrs[-1]} → {test_yr}")
        wf_results.append((test_yr, result))

    # --- Holdout Validation ---
    print("\n\n[4] Holdout Validation (≤2021 train → 2022-2025 test)...")
    print("=" * 70)

    train_yrs_ho = list(range(2016, 2022))
    X_train_ho = np.vstack([all_X_by_year[yr] for yr in train_yrs_ho if yr in all_X_by_year and len(all_X_by_year[yr]) > 0])
    y_train_ho = np.concatenate([all_y_by_year[yr] for yr in train_yrs_ho if yr in all_y_by_year and len(all_y_by_year[yr]) > 0])

    test_yrs_ho = list(range(2022, 2026))
    X_test_ho = np.vstack([all_X_by_year[yr] for yr in test_yrs_ho if yr in all_X_by_year and len(all_X_by_year[yr]) > 0])
    y_test_ho = np.concatenate([all_y_by_year[yr] for yr in test_yrs_ho if yr in all_y_by_year and len(all_y_by_year[yr]) > 0])
    meta_test_ho = [m for yr in test_yrs_ho for m in all_meta_by_year.get(yr, [])]

    print(f"  Train: {len(X_train_ho):,} samples ({y_train_ho.sum()/len(y_train_ho)*100:.1f}% WR)")
    print(f"  Test: {len(X_test_ho):,} samples ({y_test_ho.sum()/len(y_test_ho)*100:.1f}% WR)")

    model_ho = train_model(X_train_ho, y_train_ho, X_test_ho, y_test_ho)
    ho_result = evaluate_model(model_ho, X_test_ho, y_test_ho, meta_test_ho,
                              "Holdout: ≤2021 → 2022-2025")

    # In-sample check
    evaluate_model(model_ho, X_train_ho, y_train_ho,
                  [m for yr in train_yrs_ho for m in all_meta_by_year.get(yr, [])],
                  "Holdout: In-Sample (≤2021)")

    # --- 2026 OOS ---
    print("\n\n[5] 2026 Out-of-Sample...")
    print("=" * 70)

    X_2026 = all_X_by_year.get(2026, np.empty((0, NUM_FEATURES)))
    y_2026 = all_y_by_year.get(2026, np.empty(0))
    meta_2026 = all_meta_by_year.get(2026, [])

    if len(X_2026) > 0:
        oos_result = evaluate_model(model_ho, X_2026, y_2026, meta_2026, "2026 OOS")
    else:
        print("  No 2026 data available")

    # --- Train final production model on ALL data ---
    print("\n\n[6] Training final production model (all data through 2025)...")
    print("=" * 70)

    all_yrs = list(range(2016, 2026))  # Exclude 2026 for OOS
    X_all = np.vstack([all_X_by_year[yr] for yr in all_yrs if yr in all_X_by_year and len(all_X_by_year[yr]) > 0])
    y_all = np.concatenate([all_y_by_year[yr] for yr in all_yrs if yr in all_y_by_year and len(all_y_by_year[yr]) > 0])

    # Replace NaN with -999 for LightGBM
    X_all_clean = np.nan_to_num(X_all, nan=-999.0)

    final_model = train_model(X_all_clean, y_all)

    # Feature importance
    importance = final_model.feature_importance(importance_type='gain')
    sorted_idx = np.argsort(importance)[::-1]
    print(f"\n  Top 15 Features (by gain):")
    for rank, idx in enumerate(sorted_idx[:15]):
        print(f"    {rank+1:>2}. {FEATURE_NAMES[idx]:<25} {importance[idx]:>10,.0f}")

    # Save model
    model_dir = Path('surfer_models')
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / 'intraday_ml_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': final_model,
            'feature_names': FEATURE_NAMES,
            'num_features': NUM_FEATURES,
            'train_samples': len(X_all),
            'train_wr': float(y_all.mean() * 100),
            'holdout_result': ho_result,
            'wf_results': wf_results,
        }, f)
    print(f"\n  Model saved to {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")

    # Also save as standalone LightGBM model
    final_model.save_model(str(model_dir / 'intraday_lgb_model.txt'))

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE in {total_time:.0f}s ({total_time/60:.1f}m)")
    print(f"  Features: {NUM_FEATURES}")
    print(f"  Training samples: {len(X_all):,}")
    print(f"  Holdout WR improvement: {ho_result['base_wr']:.1f}% → {ho_result['ml_wr']:.1f}%")
    print(f"  Model: {model_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
