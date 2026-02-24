"""
v15/validation/break_predictor.py — ML Channel Break Predictor

Trains a LightGBM classifier to predict whether a TSLA channel break will
CONTINUE (upward, making it a tradeable long entry) or FAIL/REVERSE (making
it a trap that should be avoided — or even a short entry signal for the bounce
system's counter-trade).

Key insight from c9 energy analysis:
  High energy_ratio at boundary → FAILED break → violent reversal
  The bounce system PROFITS from failed breaks (trades the reversal).
  This model tries to predict BEFORE the break which outcome is more likely,
  so we can:
    - Skip the bounce trade when break is predicted to CONTINUE (price won't reverse)
    - Size up the bounce trade when break is predicted to FAIL (high-conviction reversal)

Algorithm:
  1. Scan 2015-2024 daily TSLA data for all channel boundary crossings
  2. Classify each break: CONTINUE (+1), FAIL/REVERSE (-1), AMBIGUOUS (0=skip)
  3. Extract 20-feature vector from bar of break (and 3 bars before)
  4. Train LightGBM binary classifier (CONTINUE vs FAIL)
  5. Leave-one-year-out cross-validation (same as signal_quality_model.py)
  6. Save model for use as dashboard signal

Usage:
  python3 -m v15.validation.break_predictor --train
  python3 -m v15.validation.break_predictor --train --output v15/validation/break_predictor_model.pkl
  python3 -m v15.validation.break_predictor --analyze   # show feature importance + CV results
"""

from __future__ import annotations

import argparse
import pickle
import warnings
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize

from v15.data.native_tf import fetch_native_tf
from v15.core.channel import detect_channel

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────────
CHANNEL_WINDOW    = 50      # bars to detect channel on
FEATURE_LOOKBACK  = 5       # bars before break to extract features from
CONTINUE_THRESH   = 0.03    # 3% gain in next 5 bars = CONTINUE
FAIL_THRESH       = -0.02   # 2% loss in next 5 bars = FAIL/REVERSE
LABEL_BARS        = 5       # bars forward to check outcome
MIN_BREAK_PCT     = 0.002   # min break size (price must cross line by 0.2%)


# ── Data structures ───────────────────────────────────────────────────────────
@dataclass
class BreakEvent:
    date:        object
    bar_idx:     int
    direction:   int          # +1 = upper break, -1 = lower break
    break_pct:   float        # how far past channel line (%)
    outcome:     int          # +1 = continue, -1 = fail/reverse, 0 = ambiguous
    forward_ret: float        # actual 5-bar return after break
    features:    np.ndarray = field(default_factory=lambda: np.array([]))
    year:        int = 0


# ── Channel helpers ───────────────────────────────────────────────────────────
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=True).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=True).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift(1)).abs()
    lc = (df['low']  - df['close'].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=True).mean()


def _channel_at(df_slice: pd.DataFrame):
    if len(df_slice) < 10:
        return None
    try:
        ch = detect_channel(df_slice)
        return ch if (ch and ch.valid) else None
    except Exception:
        return None


# ── Feature extraction ────────────────────────────────────────────────────────
FEATURE_NAMES = [
    # Channel geometry
    'channel_pos',          # price position in channel (0=lower, 1=upper, can exceed)
    'break_magnitude',      # how far past channel line (%)
    'channel_width_norm',   # channel width / price (relative width)
    'channel_compression',  # current width / width 10 bars ago (< 1 = coiling)
    # TSLA momentum
    'tsla_rsi',             # RSI 14
    'tsla_1d_ret',          # 1-day return
    'tsla_3d_ret',          # 3-day return
    'tsla_5d_ret',          # 5-day return
    'tsla_vol_ratio',       # today volume / 10d avg volume
    'tsla_atr_norm',        # ATR / price (relative volatility)
    'tsla_bullish_candle',  # close > open (0/1)
    # SPY context
    'spy_1d_ret',           # SPY 1-day return
    'spy_3d_ret',           # SPY 3-day return
    'spy_5d_ret',           # SPY 5-day return
    'spy_above_ma50',       # SPY > 50d MA (0/1)
    'spy_above_ma200',      # SPY > 200d MA (0/1)
    'tsla_spy_lag_5d',      # TSLA 5d ret minus SPY 5d ret (negative = TSLA lagging)
    # VIX regime
    'vix_level',            # raw VIX
    'vix_change_5d',        # VIX change over last 5 days (rising = stress increasing)
    # Temporal
    'day_of_week',          # 0=Mon ... 4=Fri
    'month',                # 1-12
    # Break context
    'n_upper_touches_10d',  # times in top 15% of channel in last 10 bars
    'n_lower_touches_10d',  # times in bottom 15% of channel in last 10 bars
    'bars_since_last_break', # how long since a channel was last broken
]

N_FEATURES = len(FEATURE_NAMES)


def extract_features(i: int,
                     tsla: pd.DataFrame,
                     spy: pd.DataFrame,
                     vix: pd.DataFrame,
                     rsi_tsla: pd.Series,
                     atr_tsla: pd.Series,
                     last_break_bar: int) -> Optional[np.ndarray]:
    """Extract feature vector at bar i (the break bar)."""
    if i < CHANNEL_WINDOW + 5:
        return None

    ch = _channel_at(tsla.iloc[i - CHANNEL_WINDOW:i])
    if ch is None:
        return None

    price = tsla['close'].iloc[i]
    lo    = ch.lower_line[-1]
    hi    = ch.upper_line[-1]
    width = hi - lo
    if width <= 0 or price <= 0:
        return None

    # Channel geometry
    channel_pos       = (price - lo) / width
    break_magnitude   = max(0.0, (price - hi) / price)   # for upper break
    channel_width_norm = width / price

    ch_past = _channel_at(tsla.iloc[i - CHANNEL_WINDOW - 10: i - 10])
    if ch_past is not None:
        w_past = ch_past.upper_line[-1] - ch_past.lower_line[-1]
        compression = (width / w_past) if w_past > 0 else 1.0
    else:
        compression = 1.0

    # TSLA momentum
    tsla_rsi = rsi_tsla.iloc[i]
    if pd.isna(tsla_rsi):
        tsla_rsi = 50.0

    def safe_ret(series, n):
        if i < n:
            return 0.0
        v0 = series.iloc[i - n]
        v1 = series.iloc[i]
        return (v1 / v0 - 1.0) if v0 > 0 else 0.0

    tsla_1d = safe_ret(tsla['close'], 1)
    tsla_3d = safe_ret(tsla['close'], 3)
    tsla_5d = safe_ret(tsla['close'], 5)

    avg_vol = tsla['volume'].iloc[i - 10:i].mean()
    vol_ratio = (tsla['volume'].iloc[i] / avg_vol) if avg_vol > 0 else 1.0

    atr_val   = atr_tsla.iloc[i]
    atr_norm  = (atr_val / price) if price > 0 else 0.01

    bullish_candle = 1.0 if tsla['close'].iloc[i] > tsla['open'].iloc[i] else 0.0

    # SPY context
    spy_1d = safe_ret(spy['close'], 1)
    spy_3d = safe_ret(spy['close'], 3)
    spy_5d = safe_ret(spy['close'], 5)

    spy_ma50  = spy['close'].iloc[i - 50:i].mean() if i >= 50 else spy['close'].iloc[:i].mean()
    spy_ma200 = spy['close'].iloc[i - 200:i].mean() if i >= 200 else spy['close'].iloc[:i].mean()
    spy_above_ma50  = 1.0 if spy['close'].iloc[i] > spy_ma50  else 0.0
    spy_above_ma200 = 1.0 if spy['close'].iloc[i] > spy_ma200 else 0.0
    tsla_spy_lag = tsla_5d - spy_5d

    # VIX
    vix_now    = vix['close'].iloc[i]
    vix_5d_ago = vix['close'].iloc[i - 5] if i >= 5 else vix_now
    vix_change = vix_now - vix_5d_ago

    # Temporal
    dow   = float(tsla.index[i].dayofweek)
    month = float(tsla.index[i].month)

    # Touches
    upper_touches = 0
    lower_touches = 0
    for j in range(max(0, i - 10), i):
        p_j = tsla['close'].iloc[j]
        pos_j = (p_j - lo) / width if width > 0 else 0.5
        if pos_j > 0.85:
            upper_touches += 1
        if pos_j < 0.15:
            lower_touches += 1

    bars_since = float(i - last_break_bar) if last_break_bar >= 0 else float(i)
    bars_since = min(bars_since, 100.0)   # cap at 100

    feat = np.array([
        channel_pos, break_magnitude, channel_width_norm, compression,
        tsla_rsi, tsla_1d, tsla_3d, tsla_5d, vol_ratio, atr_norm, bullish_candle,
        spy_1d, spy_3d, spy_5d, spy_above_ma50, spy_above_ma200, tsla_spy_lag,
        vix_now, vix_change,
        dow, month,
        float(upper_touches), float(lower_touches),
        bars_since,
    ], dtype=np.float32)

    if not np.all(np.isfinite(feat)):
        feat = np.nan_to_num(feat, nan=0.0, posinf=5.0, neginf=-5.0)

    return feat


# ── Break detection + labeling ────────────────────────────────────────────────
def scan_breaks(tsla: pd.DataFrame,
                spy: pd.DataFrame,
                vix: pd.DataFrame,
                start_year: int = 2015,
                end_year: int   = 2024,
                verbose: bool   = True) -> List[BreakEvent]:
    """Scan for all channel boundary crossings and label outcomes."""
    common = tsla.index.intersection(spy.index).intersection(vix.index)
    tsla = tsla.loc[common]
    spy  = spy.loc[common]
    vix  = vix.loc[common]

    n         = len(tsla)
    rsi_tsla  = _rsi(tsla['close'], 14)
    atr_tsla  = _atr(tsla, 14)
    events: List[BreakEvent] = []
    last_break_bar = -1

    for i in range(CHANNEL_WINDOW + 5, n - LABEL_BARS - 1):
        bar_year = tsla.index[i].year
        if bar_year < start_year or bar_year > end_year:
            continue

        ch = _channel_at(tsla.iloc[i - CHANNEL_WINDOW:i])
        if ch is None:
            continue

        price = tsla['close'].iloc[i]
        lo    = ch.lower_line[-1]
        hi    = ch.upper_line[-1]
        width = hi - lo
        if width <= 0:
            continue

        # Check for upper break
        prev_price = tsla['close'].iloc[i - 1]
        upper_break = price > hi * (1 + MIN_BREAK_PCT) and prev_price <= hi
        lower_break = price < lo * (1 - MIN_BREAK_PCT) and prev_price >= lo

        if not (upper_break or lower_break):
            continue

        direction  = 1 if upper_break else -1
        break_pct  = (price - hi) / hi if upper_break else (lo - price) / lo

        # Label: what happened in next LABEL_BARS bars?
        future_ret = (tsla['close'].iloc[i + LABEL_BARS] / tsla['close'].iloc[i]) - 1.0

        # For upper break: continue = price stays up, fail = reversal down
        # For lower break: continue = price stays down, fail = reversal up
        labeled_ret = future_ret * direction

        if labeled_ret >= CONTINUE_THRESH:
            outcome = 1    # CONTINUE
        elif labeled_ret <= FAIL_THRESH:
            outcome = -1   # FAIL/REVERSE
        else:
            outcome = 0    # AMBIGUOUS — skip in training

        feats = extract_features(i, tsla, spy, vix, rsi_tsla, atr_tsla, last_break_bar)

        event = BreakEvent(
            date=tsla.index[i],
            bar_idx=i,
            direction=direction,
            break_pct=break_pct,
            outcome=outcome,
            forward_ret=future_ret,
            features=feats if feats is not None else np.zeros(N_FEATURES),
            year=bar_year,
        )
        events.append(event)
        last_break_bar = i

    if verbose:
        total    = len(events)
        upper    = sum(1 for e in events if e.direction ==  1)
        lower    = sum(1 for e in events if e.direction == -1)
        cont     = sum(1 for e in events if e.outcome ==  1)
        fail     = sum(1 for e in events if e.outcome == -1)
        ambig    = sum(1 for e in events if e.outcome ==  0)
        print(f"Breaks found: {total} ({upper} upper, {lower} lower)")
        print(f"Labels: {cont} CONTINUE ({cont/total:.0%}), "
              f"{fail} FAIL ({fail/total:.0%}), "
              f"{ambig} AMBIGUOUS ({ambig/total:.0%})")

    return events


# ── Training ──────────────────────────────────────────────────────────────────
def train_model(events: List[BreakEvent],
                direction_filter: int = 1,
                verbose: bool = True):
    """Train LightGBM on break events. direction_filter=1 for upper breaks only."""
    # Filter to chosen direction and non-ambiguous
    ev = [e for e in events
          if e.direction == direction_filter
          and e.outcome != 0
          and e.features is not None
          and len(e.features) == N_FEATURES]

    if len(ev) < 30:
        print(f"Too few events ({len(ev)}) to train reliably.")
        return None, None

    X = np.array([e.features for e in ev])
    # Binary: 1 = CONTINUE, 0 = FAIL
    y = np.array([1 if e.outcome == 1 else 0 for e in ev])
    years = np.array([e.year for e in ev])

    if verbose:
        print(f"\nTraining on {len(ev)} {'+' if direction_filter==1 else '-'} breaks: "
              f"{y.sum()} CONTINUE, {(1-y).sum()} FAIL")

    # Leave-one-year-out CV
    unique_years = sorted(set(years))
    oof_preds = np.zeros(len(ev))
    oof_labels = np.zeros(len(ev))
    fold_aucs = []

    for yr in unique_years:
        train_mask = years != yr
        test_mask  = years == yr
        if test_mask.sum() < 3:
            continue

        clf = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        clf.fit(X[train_mask], y[train_mask])
        preds = clf.predict_proba(X[test_mask])[:, 1]
        oof_preds[test_mask]  = preds
        oof_labels[test_mask] = y[test_mask]

        if test_mask.sum() >= 5 and len(set(y[test_mask])) > 1:
            auc = roc_auc_score(y[test_mask], preds)
            fold_aucs.append(auc)
            if verbose:
                print(f"  {yr}: n={test_mask.sum():3d}, AUC={auc:.3f}, "
                      f"CONTINUE={y[test_mask].sum()}, FAIL={(1-y[test_mask]).sum()}")

    if len(fold_aucs) > 0:
        mean_auc = np.mean(fold_aucs)
        if verbose:
            print(f"\nLeave-one-year-out AUC: {mean_auc:.3f} (mean of {len(fold_aucs)} folds)")
            mask = oof_labels >= 0
            if len(set(oof_labels[mask])) > 1:
                overall_auc = roc_auc_score(oof_labels[mask], oof_preds[mask])
                print(f"OOF overall AUC: {overall_auc:.3f}")

    # Train final model on all data
    final_clf = LGBMClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        num_leaves=15, min_child_samples=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1,
    )
    final_clf.fit(X, y)

    # Calibrate
    calibrated = CalibratedClassifierCV(final_clf, cv=3, method='isotonic')
    calibrated.fit(X, y)

    if verbose:
        # Feature importance
        imp = final_clf.feature_importances_
        ranked = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])
        print("\nTop 10 features:")
        for name, score in ranked[:10]:
            print(f"  {name:<30s} {score:6.1f}")

    return calibrated, fold_aucs


# ── Analysis ──────────────────────────────────────────────────────────────────
def analyze_breaks(events: List[BreakEvent]) -> None:
    """Print a statistical breakdown of breaks by outcome."""
    df = pd.DataFrame([{
        'year':        e.year,
        'direction':   'UP' if e.direction == 1 else 'DOWN',
        'outcome':     {1: 'CONTINUE', -1: 'FAIL', 0: 'AMBIGUOUS'}[e.outcome],
        'forward_ret': e.forward_ret * 100,
        'break_pct':   e.break_pct * 100,
    } for e in events])

    print("\n=== Break Outcomes by Year ===")
    pivot = pd.crosstab(df['year'], [df['direction'], df['outcome']])
    print(pivot.to_string())

    print("\n=== Average Forward Return by Outcome (upper breaks only) ===")
    up = df[df['direction'] == 'UP']
    print(up.groupby('outcome')['forward_ret'].agg(['mean', 'count', 'std']).round(2).to_string())

    print("\n=== Average Forward Return by Outcome (lower breaks only) ===")
    dn = df[df['direction'] == 'DOWN']
    print(dn.groupby('outcome')['forward_ret'].agg(['mean', 'count', 'std']).round(2).to_string())


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='ML Channel Break Predictor')
    parser.add_argument('--train',   action='store_true', help='Train and save model')
    parser.add_argument('--analyze', action='store_true', help='Analyze break statistics without training')
    parser.add_argument('--output',  default='v15/validation/break_predictor_model.pkl',
                        help='Output path for trained model pkl')
    parser.add_argument('--start-year', type=int, default=2015)
    parser.add_argument('--end-year',   type=int, default=2024)
    args = parser.parse_args()

    print("Loading daily data...")
    fetch_start = f'{args.start_year - 1}-01-01'
    fetch_end   = f'{args.end_year}-12-31'

    def norm(df):
        df = df.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').normalize().tz_localize(None)
        return df

    tsla = norm(fetch_native_tf('TSLA', 'daily', fetch_start, fetch_end))
    spy  = norm(fetch_native_tf('SPY',  'daily', fetch_start, fetch_end))
    vix  = norm(fetch_native_tf('^VIX', 'daily', fetch_start, fetch_end))
    print(f"TSLA: {len(tsla)} bars | SPY: {len(spy)} bars | VIX: {len(vix)} bars")

    events = scan_breaks(tsla, spy, vix,
                         start_year=args.start_year,
                         end_year=args.end_year)

    if args.analyze or not args.train:
        analyze_breaks(events)

    if args.train:
        print("\n" + "="*60)
        print("Training UPPER BREAK predictor (CONTINUE vs FAIL)...")
        print("="*60)
        model_up, aucs_up = train_model(events, direction_filter=1, verbose=True)

        print("\n" + "="*60)
        print("Training LOWER BREAK predictor (CONTINUE vs FAIL)...")
        print("="*60)
        model_dn, aucs_dn = train_model(events, direction_filter=-1, verbose=True)

        if model_up is not None or model_dn is not None:
            payload = {
                'model_upper': model_up,
                'model_lower': model_dn,
                'feature_names': FEATURE_NAMES,
                'n_features': N_FEATURES,
                'channel_window': CHANNEL_WINDOW,
                'continue_thresh': CONTINUE_THRESH,
                'fail_thresh': FAIL_THRESH,
                'label_bars': LABEL_BARS,
                'aucs_upper': aucs_up,
                'aucs_lower': aucs_dn,
            }
            with open(args.output, 'wb') as f:
                pickle.dump(payload, f)
            print(f"\nModel saved → {args.output}")
        else:
            print("Training failed — no model saved.")


if __name__ == '__main__':
    main()


# ── Hourly Break Predictor ────────────────────────────────────────────────────
def run_hourly(start_year: int = 2024, end_year: int = 2025, output: str = None):
    """Train break predictor on 1h yfinance data for more events (~10x daily)."""
    from datetime import date, timedelta
    from v15.validation.swing_backtest import _strip_tz_intraday, _align_daily_to_hourly

    print("\nLoading 1h data (yfinance, ~2yr)...")
    hourly_end   = date.today().strftime('%Y-%m-%d')
    hourly_start = (date.today() - timedelta(days=728)).strftime('%Y-%m-%d')

    def strip(df):
        df = df.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
        return df

    tsla_h = strip(fetch_native_tf('TSLA', '1h', hourly_start, hourly_end))
    spy_h  = strip(fetch_native_tf('SPY',  '1h', hourly_start, hourly_end))
    common = tsla_h.index.intersection(spy_h.index)
    tsla_h = tsla_h.loc[common]
    spy_h  = spy_h.loc[common]

    def norm(df):
        df = df.copy()
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').normalize().tz_localize(None)
        return df

    vix_d = norm(fetch_native_tf('^VIX', 'daily', hourly_start, hourly_end))
    vix_h = _align_daily_to_hourly(vix_d, common)
    print(f"TSLA 1h: {len(tsla_h)} bars | SPY 1h: {len(spy_h)} bars")

    # Use smaller channel window for hourly (50h ≈ 1 trading week)
    global CHANNEL_WINDOW, LABEL_BARS, CONTINUE_THRESH, FAIL_THRESH
    orig = (CHANNEL_WINDOW, LABEL_BARS, CONTINUE_THRESH, FAIL_THRESH)
    CHANNEL_WINDOW   = 33   # ~1 trading week of 1h bars
    LABEL_BARS       = 8    # 8 hourly bars forward (~1 trading day)
    CONTINUE_THRESH  = 0.015  # 1.5% in 8h = CONTINUE
    FAIL_THRESH      = -0.01  # -1% in 8h = FAIL

    events = scan_breaks(tsla_h, spy_h, vix_h,
                         start_year=start_year, end_year=end_year + 1)
    analyze_breaks(events)

    if len(events) >= 30:
        print("\n" + "="*60)
        print("Training UPPER BREAK predictor (1h bars)...")
        print("="*60)
        model_up, aucs = train_model(events, direction_filter=1, verbose=True)
        if output and model_up:
            payload = {
                'model_upper': model_up, 'model_lower': None,
                'feature_names': FEATURE_NAMES, 'n_features': N_FEATURES,
                'channel_window': CHANNEL_WINDOW, 'bar_size': '1h',
                'aucs_upper': aucs,
            }
            with open(output, 'wb') as f:
                pickle.dump(payload, f)
            print(f"Saved → {output}")

    # Restore globals
    CHANNEL_WINDOW, LABEL_BARS, CONTINUE_THRESH, FAIL_THRESH = orig
