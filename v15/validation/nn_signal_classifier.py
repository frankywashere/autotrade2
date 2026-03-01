#!/usr/bin/env python3
"""
Neural Network Signal Classifier for Channel Surfer trades.

Trains a small PyTorch NN (and a GBT comparison) to predict whether a
CS trade will be a winner or loser, using features available at signal time.

Features:
  - CS signal scores (confidence, position, energy, entropy, confluence, timing, health)
  - Signal type (bounce/break), direction (BUY/SELL)
  - Calendar features (day of week, month)
  - Intraday price action (return, range)
  - Technical indicators from daily_df (RSI-14, ATR-20%, 5d/20d returns)
  - V5 bounce signal (take_bounce, confidence)
  - Per-TF momentum (1h, 4h, daily, weekly): direction and is_turning

Label: 1 if trade PnL > 0, 0 otherwise.

Train: 2016-2021 | Test: 2022-2025
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Setup path
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

CACHE_FILE = Path(__file__).resolve().parent / 'combo_cache' / 'combo_signals_compat.pkl'

# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute RSI at the last bar."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    # EMA-style
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def calc_atr_pct(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                 period: int = 20) -> float:
    """Compute ATR as percentage of price at the last bar."""
    if len(closes) < period + 1:
        return 0.03
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    atr = np.mean(tr[-period:])
    return atr / closes[-1] if closes[-1] > 0 else 0.03


# ---------------------------------------------------------------------------
# Build feature vectors from signals + trades
# ---------------------------------------------------------------------------

def build_dataset(signals, trades, daily_df):
    """
    For each trade, find the signal day (entry_date - 1 trading day typically,
    but we match by finding the signal that triggered the trade) and build
    a feature vector.

    Returns: X (np.ndarray), y (np.ndarray), trade_dates (list), trade_pnls (list)
    """
    from v15.validation.combo_backtest import (
        DaySignals, simulate_trades, _make_cs_all_combo,
        MIN_SIGNAL_CONFIDENCE,
    )

    # Build a date->signal lookup
    sig_by_date = {s.date: s for s in signals}

    # Build a date->index lookup for daily_df
    daily_dates = daily_df.index.tolist()
    daily_date_to_idx = {d: i for i, d in enumerate(daily_dates)}

    # Pre-compute daily arrays for technical indicators
    daily_closes = daily_df['close'].values.astype(float)
    daily_highs = daily_df['high'].values.astype(float)
    daily_lows = daily_df['low'].values.astype(float)
    daily_opens = daily_df['open'].values.astype(float)

    features = []
    labels = []
    trade_dates = []
    trade_pnls = []

    for trade in trades:
        # The signal day is the day BEFORE entry (since entry is at next-day open)
        # Find it: look for the signal date that is the last trading day before entry_date
        entry_date = trade.entry_date
        signal_date = None

        # Search backwards from entry_date
        for s in signals:
            if s.date < entry_date:
                signal_date = s.date
            elif s.date >= entry_date:
                break

        if signal_date is None:
            continue

        sig = sig_by_date.get(signal_date)
        if sig is None:
            continue

        # Get daily_df index for signal date
        daily_idx = daily_date_to_idx.get(signal_date)
        if daily_idx is None:
            # Find closest
            for d in daily_dates:
                if d <= signal_date:
                    daily_idx = daily_date_to_idx[d]
                else:
                    break
        if daily_idx is None or daily_idx < 30:
            continue

        # --- Build feature vector ---
        feat = []

        # 1. CS scores (7 features)
        feat.append(sig.cs_confidence)
        feat.append(sig.cs_position_score)
        feat.append(sig.cs_energy_score)
        feat.append(sig.cs_entropy_score)
        feat.append(sig.cs_confluence_score)
        feat.append(sig.cs_timing_score)
        feat.append(sig.cs_channel_health)

        # 2. Signal type: bounce=0, break=1 (1 feature)
        feat.append(1.0 if sig.cs_signal_type == 'break' else 0.0)

        # 3. Direction: BUY=1, SELL=-1 (1 feature)
        feat.append(1.0 if trade.direction == 'LONG' else -1.0)

        # 4. Calendar features (2 features)
        feat.append(float(signal_date.dayofweek))  # 0=Mon, 4=Fri
        feat.append(float(signal_date.month))       # 1-12

        # 5. Intraday price action (2 features)
        if sig.day_open > 0:
            feat.append(sig.day_close / sig.day_open - 1.0)  # intraday return
        else:
            feat.append(0.0)
        if sig.day_close > 0:
            feat.append((sig.day_high - sig.day_low) / sig.day_close)  # intraday range %
        else:
            feat.append(0.0)

        # 6. Technical indicators from daily_df (4 features)
        closes_slice = daily_closes[:daily_idx + 1]
        highs_slice = daily_highs[:daily_idx + 1]
        lows_slice = daily_lows[:daily_idx + 1]

        # RSI-14
        rsi = calc_rsi(closes_slice, 14)
        feat.append(rsi / 100.0)  # Normalize to 0-1

        # ATR-20 as % of price
        atr_pct = calc_atr_pct(highs_slice, lows_slice, closes_slice, 20)
        feat.append(atr_pct)

        # 5-day return
        if daily_idx >= 5 and daily_closes[daily_idx - 5] > 0:
            ret5 = daily_closes[daily_idx] / daily_closes[daily_idx - 5] - 1.0
        else:
            ret5 = 0.0
        feat.append(ret5)

        # 20-day return
        if daily_idx >= 20 and daily_closes[daily_idx - 20] > 0:
            ret20 = daily_closes[daily_idx] / daily_closes[daily_idx - 20] - 1.0
        else:
            ret20 = 0.0
        feat.append(ret20)

        # 7. V5 bounce features (2 features)
        feat.append(1.0 if sig.v5_take_bounce else 0.0)
        feat.append(sig.v5_confidence)

        # 8. Per-TF momentum (8 features: 4 TFs x 2 each)
        for tf in ['1h', '4h', 'daily', 'weekly']:
            if sig.cs_tf_states and tf in sig.cs_tf_states:
                st = sig.cs_tf_states[tf]
                feat.append(float(st.get('momentum_direction', 0.0)))
                feat.append(1.0 if st.get('momentum_is_turning', False) else 0.0)
            else:
                feat.append(0.0)
                feat.append(0.0)

        features.append(feat)
        labels.append(1 if trade.pnl > 0 else 0)
        trade_dates.append(trade.entry_date)
        trade_pnls.append(trade.pnl)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    return X, y, trade_dates, trade_pnls


FEATURE_NAMES = [
    'cs_confidence', 'cs_position_score', 'cs_energy_score',
    'cs_entropy_score', 'cs_confluence_score', 'cs_timing_score',
    'cs_channel_health',
    'signal_type_break', 'direction_long',
    'day_of_week', 'month',
    'intraday_return', 'intraday_range_pct',
    'rsi_14', 'atr_20_pct', 'return_5d', 'return_20d',
    'v5_take_bounce', 'v5_confidence',
    'mom_1h_dir', 'mom_1h_turning',
    'mom_4h_dir', 'mom_4h_turning',
    'mom_daily_dir', 'mom_daily_turning',
    'mom_weekly_dir', 'mom_weekly_turning',
]


# ---------------------------------------------------------------------------
# PyTorch NN
# ---------------------------------------------------------------------------

def train_nn(X_train, y_train, X_test, y_test, class_weights=None):
    """Train a small neural network classifier.

    Uses binary cross-entropy with sigmoid output, proper train/val split
    from the training data, and a 20% held-out validation fold for early
    stopping so the test set is never leaked.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    # Use CPU for small datasets -- more stable than MPS for tiny batches
    device = torch.device('cpu')
    print(f"  PyTorch device: {device}")

    # Standardize features (fit on train)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    X_tr_np = (X_train - mean) / std
    X_te_np = (X_test - mean) / std

    # Hold out 20% of training data for validation (time-ordered split)
    n_train = len(X_tr_np)
    n_val = max(1, int(n_train * 0.2))
    X_val_np = X_tr_np[-n_val:]
    y_val_np = y_train[-n_val:]
    X_tr_np = X_tr_np[:-n_val]
    y_tr_np = y_train[:-n_val]

    print(f"  NN train/val split: {len(X_tr_np)} train, {n_val} val, {len(X_test)} test")

    # Convert to tensors
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    X_te = torch.tensor(X_te_np, dtype=torch.float32)

    # Compute pos_weight for BCEWithLogitsLoss (handles class imbalance)
    n_pos = y_tr_np.sum()
    n_neg = len(y_tr_np) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    print(f"  pos_weight: {pos_weight.item():.3f}")

    # DataLoader -- full batch for tiny datasets
    batch_size = min(32, len(X_tr))
    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Model: 3-layer with BatchNorm and Dropout, single sigmoid output
    n_features = X_train.shape[1]
    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, 1),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20,
    )

    # Training with early stopping on VALIDATION loss (not test)
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 40
    best_state = None

    X_val_dev = X_val.to(device)
    y_val_dev = y_val.to(device)

    for epoch in range(500):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_X).squeeze(-1)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_y)
        epoch_loss /= len(train_ds)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_dev).squeeze(-1)
            val_loss = criterion(val_out, y_val_dev).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1:3d}: train_loss={epoch_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")

        if patience_counter >= patience_limit:
            print(f"    Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
            break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Get probabilities (sigmoid of logits)
    # For train probs, use the FULL training set (including val fold)
    X_full_tr = torch.tensor((X_train - mean) / std, dtype=torch.float32).to(device)
    X_te_dev = X_te.to(device)

    with torch.no_grad():
        train_probs = torch.sigmoid(model(X_full_tr).squeeze(-1)).cpu().numpy()
        test_probs = torch.sigmoid(model(X_te_dev).squeeze(-1)).cpu().numpy()

    return model, train_probs, test_probs, mean, std


# ---------------------------------------------------------------------------
# Gradient Boosted Tree (sklearn)
# ---------------------------------------------------------------------------

def train_gbt(X_train, y_train, X_test, y_test, class_weights=None):
    """Train a GradientBoostingClassifier as comparison."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # Compute sample weights from class weights
    sample_weights = None
    if class_weights is not None:
        sample_weights = np.array([class_weights[y] for y in y_train])

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=15,
        max_features='sqrt',
        random_state=42,
    )
    model.fit(X_tr, y_train, sample_weight=sample_weights)

    train_probs = model.predict_proba(X_tr)[:, 1]
    test_probs = model.predict_proba(X_te)[:, 1]

    return model, train_probs, test_probs, scaler


def train_rf(X_train, y_train, X_test, y_test, class_weights=None):
    """Train a Random Forest classifier as comparison."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    # class_weight parameter for RF
    cw = None
    if class_weights is not None:
        cw = {0: class_weights[0], 1: class_weights[1]}

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=10,
        max_features='sqrt',
        class_weight=cw,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_tr, y_train)

    train_probs = model.predict_proba(X_tr)[:, 1]
    test_probs = model.predict_proba(X_te)[:, 1]

    return model, train_probs, test_probs, scaler


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(name, y_true, probs, pnls, thresholds=(0.50, 0.55, 0.60, 0.65, 0.70)):
    """Print classification report and threshold analysis."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_true, preds)
    prec_1 = precision_score(y_true, preds, pos_label=1, zero_division=0)
    rec_1 = recall_score(y_true, preds, pos_label=1, zero_division=0)
    f1_1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
    prec_0 = precision_score(y_true, preds, pos_label=0, zero_division=0)
    rec_0 = recall_score(y_true, preds, pos_label=0, zero_division=0)
    f1_0 = f1_score(y_true, preds, pos_label=0, zero_division=0)

    print(f"\n  --- {name} ---")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  Class 1 (WIN):  Precision={prec_1:.3f}  Recall={rec_1:.3f}  F1={f1_1:.3f}")
    print(f"  Class 0 (LOSS): Precision={prec_0:.3f}  Recall={rec_0:.3f}  F1={f1_0:.3f}")

    # Threshold analysis
    pnls = np.array(pnls)
    total_trades = len(y_true)
    total_wins = int(y_true.sum())
    total_pnl = float(pnls.sum())
    base_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    print(f"\n  Baseline: {total_trades} trades | WR={base_wr:.1f}% | PnL=${total_pnl:+,.0f}")
    print(f"  {'Threshold':<10} {'Trades':>7} {'Wins':>6} {'WR%':>6} "
          f"{'PnL':>10} {'AvgPnL':>8} {'Rejected':>8}")
    print(f"  {'-'*58}")

    for thresh in thresholds:
        mask = probs >= thresh
        n_pass = mask.sum()
        if n_pass == 0:
            print(f"  {thresh:<10.2f} {'0':>7} {'---':>6} {'---':>6} "
                  f"{'---':>10} {'---':>8} {total_trades:>8}")
            continue
        wins = int(y_true[mask].sum())
        wr = wins / n_pass * 100
        pnl = float(pnls[mask].sum())
        avg_pnl = pnl / n_pass
        rejected = total_trades - n_pass
        print(f"  {thresh:<10.2f} {n_pass:>7} {wins:>6} {wr:>5.1f}% "
              f"${pnl:>+9,.0f} ${avg_pnl:>+7,.0f} {rejected:>8}")


def feature_importance_analysis(model, feature_names, X_test, y_test):
    """Analyze GBT feature importances."""
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print("\n  --- GBT Feature Importances (top 15) ---")
    for i, idx in enumerate(sorted_idx[:15]):
        print(f"    {i+1:2d}. {feature_names[idx]:<25s} {importances[idx]:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  Neural Network Signal Classifier for Channel Surfer Trades")
    print("=" * 70)

    # Load cache
    print(f"\nLoading cache from {CACHE_FILE}...")
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)

    signals = cache['signals']
    daily_df = cache['daily_df']
    print(f"  Loaded {len(signals)} signal days")
    print(f"  Date range: {signals[0].date.date()} to {signals[-1].date.date()}")

    # Run CS-ALL simulation to get trades
    from v15.validation.combo_backtest import simulate_trades, _make_cs_all_combo

    print("\nRunning CS-ALL trade simulation...")
    combo_fn = _make_cs_all_combo()
    trades = simulate_trades(signals, combo_fn, 'CS-ALL')
    print(f"  {len(trades)} trades generated")

    wins = sum(1 for t in trades if t.pnl > 0)
    total_pnl = sum(t.pnl for t in trades)
    print(f"  Baseline WR: {wins/len(trades)*100:.1f}% | PnL: ${total_pnl:+,.0f}")

    # Build feature vectors
    print("\nBuilding feature vectors...")
    X, y, trade_dates, trade_pnls = build_dataset(signals, trades, daily_df)
    print(f"  Feature matrix: {X.shape} ({len(FEATURE_NAMES)} features)")
    print(f"  Labels: {y.sum()} wins ({y.sum()/len(y)*100:.1f}%), "
          f"{len(y)-y.sum()} losses ({(len(y)-y.sum())/len(y)*100:.1f}%)")

    # Check for NaN/inf
    nan_count = np.isnan(X).sum()
    inf_count = np.isinf(X).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"  WARNING: {nan_count} NaN, {inf_count} inf in features -- replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train/test split by date
    train_mask = np.array([d.year <= 2021 for d in trade_dates])
    test_mask = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    pnls_train = np.array(trade_pnls)[train_mask]
    pnls_test = np.array(trade_pnls)[test_mask]

    print(f"\n  Train: {len(X_train)} trades ({y_train.sum()} wins, "
          f"{len(y_train)-y_train.sum()} losses) [<=2021]")
    print(f"  Test:  {len(X_test)} trades ({y_test.sum()} wins, "
          f"{len(y_test)-y_test.sum()} losses) [>=2022]")

    # Class weights (inversely proportional to class frequency)
    n_samples = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_samples - n_pos
    w_pos = n_samples / (2.0 * n_pos) if n_pos > 0 else 1.0
    w_neg = n_samples / (2.0 * n_neg) if n_neg > 0 else 1.0
    class_weights = {0: w_neg, 1: w_pos}
    print(f"\n  Class weights: loss={w_neg:.3f}, win={w_pos:.3f}")

    # -----------------------------------------------------------------------
    # 1. Neural Network
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRAINING: PyTorch Neural Network")
    print("=" * 70)

    nn_model, nn_train_probs, nn_test_probs, nn_mean, nn_std = train_nn(
        X_train, y_train, X_test, y_test, class_weights
    )

    print("\n  >> TRAIN SET <<")
    evaluate("NN Train", y_train, nn_train_probs, pnls_train)

    print("\n  >> TEST SET (OOS 2022-2025) <<")
    evaluate("NN Test", y_test, nn_test_probs, pnls_test)

    # -----------------------------------------------------------------------
    # 2. Gradient Boosted Trees
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRAINING: Gradient Boosted Trees (sklearn)")
    print("=" * 70)

    gbt_model, gbt_train_probs, gbt_test_probs, gbt_scaler = train_gbt(
        X_train, y_train, X_test, y_test, class_weights
    )

    print("\n  >> TRAIN SET <<")
    evaluate("GBT Train", y_train, gbt_train_probs, pnls_train)

    print("\n  >> TEST SET (OOS 2022-2025) <<")
    evaluate("GBT Test", y_test, gbt_test_probs, pnls_test)

    # Feature importance from GBT
    feature_importance_analysis(gbt_model, FEATURE_NAMES, X_test, y_test)

    # -----------------------------------------------------------------------
    # 3. Random Forest
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  TRAINING: Random Forest (sklearn)")
    print("=" * 70)

    rf_model, rf_train_probs, rf_test_probs, rf_scaler = train_rf(
        X_train, y_train, X_test, y_test, class_weights
    )

    print("\n  >> TRAIN SET <<")
    evaluate("RF Train", y_train, rf_train_probs, pnls_train)

    print("\n  >> TEST SET (OOS 2022-2025) <<")
    evaluate("RF Test", y_test, rf_test_probs, pnls_test)

    # Feature importance from RF
    feature_importance_analysis(rf_model, FEATURE_NAMES, X_test, y_test)

    # -----------------------------------------------------------------------
    # Summary comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY (Test Set, OOS 2022-2025)")
    print("=" * 70)

    from sklearn.metrics import accuracy_score, roc_auc_score

    def _safe_auc(y_true, probs):
        try:
            return roc_auc_score(y_true, probs)
        except ValueError:
            return 0.0

    models_results = {
        'Neural Network': nn_test_probs,
        'Gradient Boosted': gbt_test_probs,
        'Random Forest': rf_test_probs,
    }

    print(f"\n  {'Model':<20} {'Accuracy':>9} {'AUC-ROC':>9}")
    print(f"  {'-'*40}")
    for name, probs in models_results.items():
        acc = accuracy_score(y_test, (probs >= 0.5).astype(int))
        auc = _safe_auc(y_test, probs)
        print(f"  {name:<20} {acc:>8.3f} {auc:>9.3f}")

    # Threshold comparison across all models
    def _stats(mask, ys, pnls):
        n = mask.sum()
        if n == 0:
            return 0, 0.0, 0.0
        w = int(ys[mask].sum())
        wr = w / n * 100
        pnl = float(pnls[mask].sum())
        return n, wr, pnl

    def _fmt(n, wr, pnl):
        if n == 0:
            return f"{'0':>7} {'---':>6} {'---':>11}"
        return f"{n:>7} {wr:>5.1f}% ${pnl:>+10,.0f}"

    print(f"\n  {'Threshold':<10} | {'--- NN ---':^26} | {'--- GBT ---':^26} | {'--- RF ---':^26}")
    print(f"  {'':10} | {'Trades':>7} {'WR%':>6} {'PnL':>11} "
          f"| {'Trades':>7} {'WR%':>6} {'PnL':>11} "
          f"| {'Trades':>7} {'WR%':>6} {'PnL':>11}")
    print(f"  {'-'*93}")

    baseline_n = len(y_test)
    baseline_wr = y_test.sum() / len(y_test) * 100
    baseline_pnl = pnls_test.sum()
    bl = _fmt(baseline_n, baseline_wr, baseline_pnl)
    print(f"  {'Baseline':<10} | {bl} | {bl} | {bl}")

    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        parts = []
        for probs in [nn_test_probs, gbt_test_probs, rf_test_probs]:
            mask = probs >= thresh
            n, wr, pnl = _stats(mask, y_test, pnls_test)
            parts.append(_fmt(n, wr, pnl))
        print(f"  {thresh:<10.2f} | {parts[0]} | {parts[1]} | {parts[2]}")

    # -----------------------------------------------------------------------
    # Ensemble: average of all 3 models
    # -----------------------------------------------------------------------
    print(f"\n  --- Ensemble (average of NN + GBT + RF) ---")
    ensemble_probs = (nn_test_probs + gbt_test_probs + rf_test_probs) / 3.0
    ens_acc = accuracy_score(y_test, (ensemble_probs >= 0.5).astype(int))
    ens_auc = _safe_auc(y_test, ensemble_probs)
    print(f"  Accuracy: {ens_acc:.3f} | AUC-ROC: {ens_auc:.3f}")

    print(f"\n  {'Threshold':<10} {'Trades':>7} {'Wins':>6} {'WR%':>6} "
          f"{'PnL':>11} {'AvgPnL':>9} {'Rejected':>8}")
    print(f"  {'-'*62}")
    print(f"  {'Baseline':<10} {baseline_n:>7} {int(y_test.sum()):>6} {baseline_wr:>5.1f}% "
          f"${baseline_pnl:>+10,.0f} ${baseline_pnl/baseline_n:>+8,.0f} {'0':>8}")

    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = ensemble_probs >= thresh
        n = mask.sum()
        if n == 0:
            print(f"  {thresh:<10.2f} {'0':>7}")
            continue
        w = int(y_test[mask].sum())
        wr = w / n * 100
        pnl = float(pnls_test[mask].sum())
        avg = pnl / n
        rej = baseline_n - n
        print(f"  {thresh:<10.2f} {n:>7} {w:>6} {wr:>5.1f}% "
              f"${pnl:>+10,.0f} ${avg:>+8,.0f} {rej:>8}")

    print("\n  Done.")


if __name__ == '__main__':
    main()
