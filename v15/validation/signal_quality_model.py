#!/usr/bin/env python3
"""
Signal Quality Model — Predicts per-trade win probability and expected P&L.

Trains a GBT model on the 169-dim ML feature vectors captured from backtests,
plus signal-meta features (9 base or 21 extended).

Uses leave-one-year-out cross-validation on the 10-year backtest dataset
(~10K trades) for honest performance estimation.

Supports:
  - Extended features (stop_pct, tp_pct, R:R, TF one-hots, interactions)
  - Isotonic calibration for well-calibrated win probabilities
  - Custom hyperparameters (from Optuna tuning)

Usage:
    python3 -m v15.validation.signal_quality_model \
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt \
        --output v15/validation/signal_quality_model.pkl
"""

import argparse
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TradeSnapshot:
    """One trade's feature vector + outcome for training."""
    features: np.ndarray   # 178-dim (base) or 190-dim (extended)
    win: int               # 1 if pnl > 0
    pnl_pct: float         # P&L percentage
    year: int              # For CV fold assignment


SIGNAL_META_NAMES = [
    'sig_type_bounce',       # 0/1
    'sig_direction_buy',     # 0/1
    'sig_position_score',
    'sig_energy_score',
    'sig_entropy_score',
    'sig_confluence_score',
    'sig_timing_score',
    'sig_channel_health',
    'sig_confidence',
]

SIGNAL_META_EXTENDED_NAMES = SIGNAL_META_NAMES + [
    'sig_stop_pct',
    'sig_tp_pct',
    'sig_rr_ratio',
    'sig_primary_tf_5min',
    'sig_primary_tf_1h',
    'sig_primary_tf_4h',
    'sig_primary_tf_daily',
    'sig_primary_tf_weekly',
    'sig_vix_x_position',
    'sig_confluence_x_energy',
    'sig_is_overnight',
]

# TF names used for one-hot encoding of primary_tf
_TF_ONEHOT_KEYS = ['5min', '1h', '4h', 'daily', 'weekly']


def _append_signal_meta(
    base_features: np.ndarray,
    trade,
    sig_data: dict,
    extended: bool = False,
) -> np.ndarray:
    """Append signal-meta features to the 169-dim base vector.

    Args:
        extended: If True, append 21 features (9 base + 12 new).
                  If False, append original 9 features for backward compat.
    """
    names = SIGNAL_META_EXTENDED_NAMES if extended else SIGNAL_META_NAMES
    meta = np.zeros(len(names), dtype=np.float32)

    # --- Original 9 features (indices 0-8) ---
    meta[0] = 1.0 if trade.signal_type == 'bounce' else 0.0
    meta[1] = 1.0 if trade.direction == 'BUY' else 0.0
    if sig_data:
        meta[2] = sig_data.get('position_score', 0.0)
        meta[3] = sig_data.get('energy_score', 0.0)
        meta[4] = sig_data.get('entropy_score', 0.0)
        meta[5] = sig_data.get('confluence_score', 0.0)
        meta[6] = sig_data.get('timing_score', 0.0)
        meta[7] = sig_data.get('channel_health', 0.0)
        meta[8] = sig_data.get('confidence', 0.0)

    if extended:
        # --- New features (indices 9-20) ---
        stop_pct = getattr(trade, 'stop_pct', 0.0) or 0.0
        tp_pct = getattr(trade, 'tp_pct', 0.0) or 0.0
        meta[9] = stop_pct
        meta[10] = tp_pct
        meta[11] = stop_pct / tp_pct if tp_pct > 1e-8 else 1.0  # R:R ratio

        # Primary TF one-hot
        primary_tf = getattr(trade, 'primary_tf', '') or ''
        for j, tf_key in enumerate(_TF_ONEHOT_KEYS):
            meta[12 + j] = 1.0 if primary_tf == tf_key else 0.0

        # Interaction features
        pos_score = sig_data.get('position_score', 0.0) if sig_data else 0.0
        confluence = sig_data.get('confluence_score', 0.0) if sig_data else 0.0
        energy = sig_data.get('energy_score', 0.0) if sig_data else 0.0

        # VIX level from base features — look up dynamically to handle feature count changes
        try:
            from v15.core.surfer_ml import get_feature_names as _gfn
            _vix_idx = _gfn().index('vix_level')
        except (ValueError, ImportError):
            _vix_idx = 167  # fallback: 120 per-TF + 18 cross + 14 ctx + 12 temporal + 3 corr
        vix_level = float(base_features[_vix_idx]) if len(base_features) > _vix_idx else 0.0
        meta[17] = vix_level * pos_score  # sig_vix_x_position

        meta[18] = confluence * energy  # sig_confluence_x_energy

        # Overnight flag: entry hour >= 15 (3pm ET)
        entry_time = getattr(trade, 'entry_time', '') or ''
        is_overnight = 0.0
        if entry_time:
            try:
                hour = int(entry_time.split('T')[1].split(':')[0]) if 'T' in entry_time else 0
                is_overnight = 1.0 if hour >= 15 else 0.0
            except (IndexError, ValueError):
                pass
        meta[19] = is_overnight

    return np.concatenate([base_features, meta])


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    tsla_path: str,
    spy_path: str = None,
    years: List[int] = None,
    eval_interval: int = 6,
    min_confidence: float = 0.45,
    capital: float = 100_000.0,
    verbose: bool = True,
    extended_features: bool = True,
    bounce_cap: float = 12.0,
    max_trade_usd: float = 1_000_000.0,
) -> List[TradeSnapshot]:
    """
    Run year-by-year backtests with capture_features=True, no ML model.
    Returns list of TradeSnapshot with feature vectors and outcomes.
    bounce_cap and max_trade_usd should match the arch being trained on.
    """
    from v15.core.historical_data import prepare_backtest_data, prepare_year_data
    from v15.core.surfer_backtest import run_backtest

    if years is None:
        years = list(range(2015, 2025))

    if verbose:
        print(f"\n{'='*60}")
        print("BUILDING SIGNAL QUALITY DATASET")
        print(f"{'='*60}")
        print(f"Years: {years[0]}-{years[-1]}")

    # Load all data once
    t0 = time.time()
    full_data = prepare_backtest_data(tsla_path, spy_path)

    # Also load VIX for correlation features
    vix_df = None
    try:
        from v15.validation.vix_loader import load_vix_daily
        vix_df = load_vix_daily(start='2014-01-01', end='2025-12-31')
    except Exception as e:
        if verbose:
            print(f"  VIX load failed: {e} (correlation features will be zeros)")

    if verbose:
        print(f"  Data loaded in {time.time() - t0:.1f}s")

    snapshots: List[TradeSnapshot] = []

    for year in years:
        year_data = prepare_year_data(full_data, year)
        if year_data is None:
            if verbose:
                print(f"  {year}: no data, skipping")
            continue

        tsla_5min = year_data['tsla_5min']
        if len(tsla_5min) < 200:
            if verbose:
                print(f"  {year}: too few bars ({len(tsla_5min)}), skipping")
            continue

        t_year = time.time()
        result = run_backtest(
            days=0,
            eval_interval=eval_interval,
            max_hold_bars=60,
            position_size=capital / 10,
            min_confidence=min_confidence,
            use_multi_tf=True,
            tsla_df=tsla_5min,
            higher_tf_dict=year_data['higher_tf_data'],
            spy_df_input=year_data.get('spy_5min'),
            vix_df_input=vix_df,
            realistic=True,
            slippage_bps=3.0,
            commission_per_share=0.005,
            max_leverage=4.0,
            initial_capital=capital,
            bounce_cap=bounce_cap,
            max_trade_usd=max_trade_usd,
            capture_features=True,
        )

        # capture_features=True returns 5-tuple
        metrics, trades, equity_curve, trade_features, trade_signals = result

        n_with_features = 0
        for i, trade in enumerate(trades):
            feat = trade_features[i] if i < len(trade_features) else None
            sig = trade_signals[i] if i < len(trade_signals) else None
            if feat is None:
                continue

            full_feat = _append_signal_meta(feat, trade, sig or {}, extended=extended_features)
            snapshots.append(TradeSnapshot(
                features=full_feat,
                win=1 if trade.pnl > 0 else 0,
                pnl_pct=trade.pnl_pct,
                year=year,
            ))
            n_with_features += 1

        elapsed = time.time() - t_year
        win_rate = metrics.win_rate if metrics.total_trades > 0 else 0
        if verbose:
            print(f"  {year}: {metrics.total_trades} trades, "
                  f"{n_with_features} with features, "
                  f"WR={win_rate:.0%}, "
                  f"PF={metrics.profit_factor:.2f}, "
                  f"({elapsed:.1f}s)")

    if verbose:
        print(f"\nTotal snapshots: {len(snapshots)}")
        if snapshots:
            wins = sum(s.win for s in snapshots)
            print(f"  Win rate: {wins/len(snapshots):.1%}")
            print(f"  Feature dim: {len(snapshots[0].features)}")

    return snapshots


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SignalQualityModel:
    """
    Two sub-models:
      - win_classifier: P(win | features) — binary classifier
      - pnl_regressor: E[pnl_pct | features] — Huber regressor

    Supports isotonic calibration and custom hyperparameters.
    """

    def __init__(self, params: dict = None, loss_weight_scale: float = 0.0):
        self.win_model = None
        self.pnl_model = None
        self.calibrator = None  # IsotonicRegression for calibrated win_prob
        self.feature_names: List[str] = []
        self.cv_metrics: Dict = {}
        self.feature_importance: Dict = {}
        self.params = params  # Custom LightGBM hyperparameters
        self.loss_weight_scale = loss_weight_scale  # Weight losers by |pnl_pct| magnitude

    def train(self, snapshots: List[TradeSnapshot], verbose: bool = True):
        """Train with leave-one-year-out CV, then retrain on all data."""
        X = np.array([s.features for s in snapshots], dtype=np.float32)
        y_win = np.array([s.win for s in snapshots], dtype=np.int32)
        y_pnl = np.array([s.pnl_pct for s in snapshots], dtype=np.float32)
        years = np.array([s.year for s in snapshots])

        # Build feature names — auto-detect extended vs base from feature dim
        from v15.core.surfer_ml import get_feature_names
        base_names = get_feature_names()
        n_base = len(base_names)
        n_feat = X.shape[1]
        n_meta = n_feat - n_base
        if n_meta == len(SIGNAL_META_EXTENDED_NAMES):
            self.feature_names = base_names + SIGNAL_META_EXTENDED_NAMES
        else:
            self.feature_names = base_names + SIGNAL_META_NAMES

        unique_years = sorted(int(y) for y in set(years))
        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING SIGNAL QUALITY MODEL")
            print(f"{'='*60}")
            print(f"  Samples: {len(snapshots)}, Features: {X.shape[1]}")
            print(f"  Years: {unique_years}")
            print(f"  Win rate: {y_win.mean():.1%}")
            if self.params:
                print(f"  Custom params: yes ({len(self.params)} overrides)")

        # --- Leave-one-year-out CV ---
        cv_win_probs = np.zeros(len(snapshots))
        cv_pnl_preds = np.zeros(len(snapshots))
        per_year_metrics = {}

        # Calibrated OOS predictions (nested calibration — no leak)
        cv_cal_win_probs = np.zeros(len(snapshots))

        for held_year in unique_years:
            train_mask = years != held_year
            test_mask = years == held_year

            X_train, X_test = X[train_mask], X[test_mask]
            y_win_train, y_win_test = y_win[train_mask], y_win[test_mask]
            y_pnl_train, y_pnl_test = y_pnl[train_mask], y_pnl[test_mask]

            if len(X_test) < 10:
                continue

            # Split train into model-train + calibration (use earliest year as cal)
            train_years = sorted(set(years[train_mask].astype(int)))
            cal_year = train_years[0]  # Hold out earliest train year for calibration
            cal_mask_in_train = years[train_mask] == cal_year
            model_mask_in_train = ~cal_mask_in_train

            X_model = X_train[model_mask_in_train]
            y_win_model = y_win_train[model_mask_in_train]
            y_pnl_model = y_pnl_train[model_mask_in_train]
            X_cal = X_train[cal_mask_in_train]
            y_win_cal = y_win_train[cal_mask_in_train]

            sw_model = self._compute_sample_weights(y_win_model, y_pnl_model)
            win_model, pnl_model = self._fit_models(X_model, y_win_model, y_pnl_model,
                                                     sample_weight=sw_model)

            # OOS predictions (wrap in DataFrame to match fit)
            X_test_df = self._make_df(X_test)
            win_probs = win_model.predict_proba(X_test_df)[:, 1]
            pnl_preds = pnl_model.predict(X_test_df)

            cv_win_probs[test_mask] = win_probs
            cv_pnl_preds[test_mask] = pnl_preds

            # Nested calibration: fit on cal fold, apply to test fold
            if len(X_cal) >= 30:
                from sklearn.isotonic import IsotonicRegression
                X_cal_df = self._make_df(X_cal)
                cal_probs = win_model.predict_proba(X_cal_df)[:, 1]
                fold_calibrator = IsotonicRegression(
                    y_min=0.0, y_max=1.0, out_of_bounds='clip'
                )
                fold_calibrator.fit(cal_probs, y_win_cal)
                cv_cal_win_probs[test_mask] = fold_calibrator.predict(win_probs)
            else:
                cv_cal_win_probs[test_mask] = win_probs

            # Per-year metrics
            from sklearn.metrics import roc_auc_score, brier_score_loss
            try:
                auc = roc_auc_score(y_win_test, win_probs)
            except ValueError:
                auc = 0.5
            brier = brier_score_loss(y_win_test, win_probs)
            pnl_mae = np.mean(np.abs(pnl_preds - y_pnl_test))
            wr = y_win_test.mean()

            per_year_metrics[held_year] = {
                'n_trades': int(len(X_test)),
                'win_rate': float(wr),
                'auc': float(auc),
                'brier': float(brier),
                'pnl_mae': float(pnl_mae),
            }

            if verbose:
                print(f"  {held_year}: n={len(X_test):4d}, "
                      f"WR={wr:.0%}, AUC={auc:.3f}, "
                      f"Brier={brier:.3f}, PnL MAE={pnl_mae:.4f}")

        # Overall CV metrics
        from sklearn.metrics import roc_auc_score, brier_score_loss
        valid = cv_win_probs > 0  # skip years with no predictions
        if valid.sum() > 0:
            overall_auc = roc_auc_score(y_win[valid], cv_win_probs[valid])
            overall_brier = brier_score_loss(y_win[valid], cv_win_probs[valid])
            overall_pnl_mae = np.mean(np.abs(cv_pnl_preds[valid] - y_pnl[valid]))
        else:
            overall_auc = 0.5
            overall_brier = 0.25
            overall_pnl_mae = 0.0

        # --- Isotonic calibration (nested — honest OOS Brier) ---
        cal_valid = cv_cal_win_probs > 0
        if cal_valid.sum() > 50:
            calibrated_brier = brier_score_loss(y_win[cal_valid], cv_cal_win_probs[cal_valid])
            if verbose:
                print(f"\n  Nested calibration: "
                      f"Brier {overall_brier:.4f} → {calibrated_brier:.4f} (honest OOS)")
        else:
            calibrated_brier = overall_brier

        # Fit final calibrator on ALL OOS predictions for production use
        if valid.sum() > 50:
            from sklearn.isotonic import IsotonicRegression
            self.calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds='clip'
            )
            self.calibrator.fit(cv_win_probs[valid], y_win[valid])


        # --- Bootstrap 95% confidence intervals ---
        auc_ci = [None, None]
        brier_ci = [None, None]
        if valid.sum() > 100:
            rng = np.random.RandomState(42)
            n = int(valid.sum())
            _y = y_win[valid]
            _p = cv_win_probs[valid]
            boot_aucs, boot_briers = [], []
            for _ in range(5000):
                idx = rng.choice(n, n, replace=True)
                if len(set(_y[idx])) < 2:
                    continue
                boot_aucs.append(roc_auc_score(_y[idx], _p[idx]))
                boot_briers.append(brier_score_loss(_y[idx], _p[idx]))
            if boot_aucs:
                auc_ci = [float(np.percentile(boot_aucs, 2.5)),
                          float(np.percentile(boot_aucs, 97.5))]
                brier_ci = [float(np.percentile(boot_briers, 2.5)),
                            float(np.percentile(boot_briers, 97.5))]

        self.cv_metrics = {
            'overall_auc': float(overall_auc),
            'overall_auc_ci': auc_ci,
            'overall_brier': float(overall_brier),
            'overall_brier_ci': brier_ci,
            'calibrated_brier': float(calibrated_brier),
            'overall_pnl_mae': float(overall_pnl_mae),
            'per_year': per_year_metrics,
        }

        if verbose:
            auc_ci_str = (f" [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]"
                          if auc_ci[0] is not None else "")
            brier_ci_str = (f" [{brier_ci[0]:.3f}, {brier_ci[1]:.3f}]"
                            if brier_ci[0] is not None else "")
            print(f"\n  Overall CV: AUC={overall_auc:.3f}{auc_ci_str}, "
                  f"Brier={overall_brier:.3f}{brier_ci_str} "
                  f"(calibrated: {calibrated_brier:.3f}), "
                  f"PnL MAE={overall_pnl_mae:.4f}")

        # --- Retrain on ALL data for production ---
        if verbose:
            print(f"\n  Retraining on all {len(snapshots)} samples for production model...")
        sw_all = self._compute_sample_weights(y_win, y_pnl)
        self.win_model, self.pnl_model = self._fit_models(X, y_win, y_pnl,
                                                           sample_weight=sw_all)

        # Feature importance
        self._compute_feature_importance()

        if verbose:
            self._print_top_features(20)

    def _make_df(self, X):
        """Wrap array in DataFrame with feature names to suppress LightGBM warnings."""
        import pandas as pd
        names = self.feature_names if len(self.feature_names) == X.shape[-1] else None
        if X.ndim == 1:
            return pd.DataFrame([X], columns=names)
        return pd.DataFrame(X, columns=names)

    def _compute_sample_weights(self, y_win, y_pnl):
        """Compute sample weights for loss-magnitude weighting.

        Winners get weight=1.0. Losers get weight = 1 + scale * |pnl_pct|.
        If loss_weight_scale=0 (default), returns None (uniform weights).
        """
        if self.loss_weight_scale <= 0:
            return None
        weights = np.ones(len(y_win), dtype=np.float32)
        loser_mask = y_win == 0
        weights[loser_mask] = 1.0 + self.loss_weight_scale * np.abs(y_pnl[loser_mask])
        return weights

    def _fit_models(self, X, y_win, y_pnl, sample_weight=None):
        """Fit win classifier and pnl regressor."""
        X_df = self._make_df(X)
        try:
            import lightgbm as lgb

            # Default params, overridden by self.params if set
            default_params = dict(
                n_estimators=500,
                num_leaves=31,
                learning_rate=0.03,
                min_child_samples=50,
                feature_fraction=0.7,
                bagging_fraction=0.7,
                bagging_freq=5,
            )
            if self.params:
                default_params.update(self.params)

            win_model = lgb.LGBMClassifier(
                is_unbalance=True,
                verbose=-1,
                n_jobs=-1,
                **default_params,
            )
            win_model.fit(X_df, y_win, sample_weight=sample_weight)

            pnl_model = lgb.LGBMRegressor(
                objective='huber',
                verbose=-1,
                n_jobs=-1,
                **default_params,
            )
            pnl_model.fit(X_df, y_pnl, sample_weight=sample_weight)

        except ImportError:
            print("  LightGBM not available, falling back to sklearn GBT")
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

            win_model = GradientBoostingClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                min_samples_leaf=50,
                subsample=0.7,
                max_features=0.7,
            )
            win_model.fit(X, y_win)

            pnl_model = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                min_samples_leaf=50,
                subsample=0.7,
                max_features=0.7,
                loss='huber',
            )
            pnl_model.fit(X, y_pnl)

        return win_model, pnl_model

    def _compute_feature_importance(self):
        """Extract feature importance from trained models."""
        if self.win_model is None:
            return
        try:
            win_imp = self.win_model.feature_importances_
            pnl_imp = self.pnl_model.feature_importances_
        except AttributeError:
            return

        names = self.feature_names
        if len(names) != len(win_imp):
            names = [f'f{i}' for i in range(len(win_imp))]

        # Sort by win importance
        sorted_idx = np.argsort(win_imp)[::-1]
        self.feature_importance = {
            'win_top': [(names[i], float(win_imp[i])) for i in sorted_idx[:30]],
            'pnl_top': [(names[i], float(pnl_imp[i]))
                        for i in np.argsort(pnl_imp)[::-1][:30]],
        }

    def _print_top_features(self, n: int = 20):
        """Print top feature importances."""
        if not self.feature_importance:
            return
        print(f"\n  Top {n} features (win classifier):")
        for name, imp in self.feature_importance['win_top'][:n]:
            print(f"    {name:<40s} {imp:.0f}")
        print(f"\n  Top {n} features (pnl regressor):")
        for name, imp in self.feature_importance['pnl_top'][:n]:
            print(f"    {name:<40s} {imp:.0f}")

    def predict(self, feature_vec: np.ndarray) -> dict:
        """
        Predict signal quality for a single feature vector.

        Args:
            feature_vec: 178-dim or 190-dim array (169 base + 9/21 signal meta)

        Returns:
            dict with win_prob, expected_pnl_pct, quality_score, risk_rating
        """
        if self.win_model is None:
            return {'win_prob': 0.5, 'expected_pnl_pct': 0.0,
                    'quality_score': 50.0, 'risk_rating': 'MEDIUM'}

        X = self._make_df(feature_vec.reshape(1, -1))
        win_prob = float(self.win_model.predict_proba(X)[0, 1])
        expected_pnl = float(self.pnl_model.predict(X)[0])

        # Apply isotonic calibration if available
        if self.calibrator is not None:
            win_prob = float(self.calibrator.predict([win_prob])[0])

        # Quality score: weighted combination (0-100)
        # win_prob contributes 70%, pnl direction 30%
        pnl_signal = np.clip(expected_pnl * 500 + 0.5, 0, 1)  # Scale pnl to 0-1
        quality_score = float(np.clip(win_prob * 70 + pnl_signal * 30, 0, 100))

        # Risk rating
        if win_prob >= 0.75 and expected_pnl > 0:
            risk_rating = 'LOW'
        elif win_prob < 0.55 or expected_pnl < -0.002:
            risk_rating = 'HIGH'
        else:
            risk_rating = 'MEDIUM'

        return {
            'win_prob': win_prob,
            'expected_pnl_pct': expected_pnl,
            'quality_score': quality_score,
            'risk_rating': risk_rating,
        }

    def save(self, path: str):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'win_model': self.win_model,
                'pnl_model': self.pnl_model,
                'calibrator': self.calibrator,
                'feature_names': self.feature_names,
                'cv_metrics': self.cv_metrics,
                'feature_importance': self.feature_importance,
                'params': self.params,
            }, f)
        print(f"  Saved signal quality model to {path}")

    @classmethod
    def load(cls, path: str) -> 'SignalQualityModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls(params=data.get('params'))
        model.win_model = data['win_model']
        model.pnl_model = data['pnl_model']
        model.calibrator = data.get('calibrator')
        model.feature_names = data.get('feature_names', [])
        model.cv_metrics = data.get('cv_metrics', {})
        model.feature_importance = data.get('feature_importance', {})
        return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train Signal Quality Model (win probability + expected PnL)')
    parser.add_argument('--tsla', type=str, default='data/TSLAMin.txt',
                        help='Path to TSLA minute data')
    parser.add_argument('--spy', type=str, default=None,
                        help='Path to SPY minute data (optional)')
    parser.add_argument('--output', type=str,
                        default='v15/validation/signal_quality_model.pkl',
                        help='Output path for trained model')
    parser.add_argument('--years', type=str, default='2015-2024',
                        help='Year range (e.g., 2015-2024)')
    parser.add_argument('--eval-interval', type=int, default=6,
                        help='Bars between evaluations')
    parser.add_argument('--min-confidence', type=float, default=0.45,
                        help='Minimum signal confidence')
    parser.add_argument('--capital', type=float, default=100_000.0,
                        help='Initial capital')
    parser.add_argument('--tuned-params', type=str, default=None,
                        help='Path to tuned_params.json from Optuna')
    parser.add_argument('--no-extended', action='store_true',
                        help='Disable extended features (use original 9 meta)')
    parser.add_argument('--bounce-cap', type=float, default=12.0,
                        help='Max exposure cap multiplier for bounce signals (default: 12.0, c9/Arch418)')
    parser.add_argument('--max-trade-usd', type=float, default=1_000_000.0,
                        help='Hard dollar cap per trade (default: 1000000, Arch418)')
    parser.add_argument('--loss-weight', type=float, default=0.0,
                        help='Scale factor for loss-magnitude sample weighting (0=off, try 200-500)')
    args = parser.parse_args()

    # Parse year range
    parts = args.years.split('-')
    start_year = int(parts[0])
    end_year = int(parts[1]) if len(parts) > 1 else start_year
    years = list(range(start_year, end_year + 1))

    # Load tuned params if provided
    custom_params = None
    if args.tuned_params:
        if not os.path.isfile(args.tuned_params):
            print(f"\nERROR: --tuned-params file not found: {args.tuned_params}")
            print("  Run optuna_tune.py first to generate it, or omit the flag.")
            sys.exit(1)
        with open(args.tuned_params) as f:
            data = json.load(f)
        custom_params = data.get('best_params', data)
        if not custom_params:
            print(f"\nERROR: No 'best_params' found in {args.tuned_params}")
            sys.exit(1)
        print(f"  Loaded tuned params from {args.tuned_params}")

    extended = not args.no_extended

    # Build dataset
    snapshots = build_dataset(
        tsla_path=args.tsla,
        spy_path=args.spy,
        years=years,
        eval_interval=args.eval_interval,
        min_confidence=args.min_confidence,
        capital=args.capital,
        extended_features=extended,
        bounce_cap=args.bounce_cap,
        max_trade_usd=args.max_trade_usd,
    )

    if len(snapshots) < 100:
        print(f"\nERROR: Only {len(snapshots)} snapshots — need at least 100 for training")
        sys.exit(1)

    # Train model
    if args.loss_weight > 0:
        print(f"  Loss-magnitude weighting: scale={args.loss_weight}")
    model = SignalQualityModel(params=custom_params, loss_weight_scale=args.loss_weight)
    model.train(snapshots, verbose=True)

    # Save
    model.save(args.output)

    # Summary
    print(f"\n{'='*60}")
    print("SIGNAL QUALITY MODEL — SUMMARY")
    print(f"{'='*60}")
    print(f"  Trained on: {len(snapshots)} trades")
    print(f"  Features:   {len(snapshots[0].features)}-dim "
          f"({'extended' if extended else 'base'})")
    _ci = model.cv_metrics.get('overall_auc_ci', [None, None])
    _ci_str = f" [{_ci[0]:.3f}, {_ci[1]:.3f}]" if _ci[0] is not None else ""
    print(f"  CV AUC:     {model.cv_metrics['overall_auc']:.3f}{_ci_str}")
    print(f"  CV Brier:   {model.cv_metrics['overall_brier']:.3f} "
          f"(calibrated: {model.cv_metrics.get('calibrated_brier', 'N/A')})")
    print(f"  CV PnL MAE: {model.cv_metrics['overall_pnl_mae']:.4f}")
    print(f"  Saved to:   {args.output}")


if __name__ == '__main__':
    main()
