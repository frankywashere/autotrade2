#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning for LightGBM Signal Quality Model.

Optimizes LightGBM hyperparameters using leave-one-year-out CV AUC
as the objective. Saves best params to tuned_params.json.

Usage:
    python3 -m v15.validation.optuna_tune \
        --tsla data/TSLAMin.txt --spy data/SPYMin.txt \
        --n-trials 100 --timeout 3600
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import numpy as np

from v15.validation.signal_quality_model import TradeSnapshot, build_dataset


def loo_cv_auc(
    snapshots: List[TradeSnapshot],
    params: dict,
) -> float:
    """Leave-one-year-out CV, returns mean AUC across years."""
    import lightgbm as lgb
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from v15.core.surfer_ml import get_feature_names
    from v15.validation.signal_quality_model import (
        SIGNAL_META_EXTENDED_NAMES,
        SIGNAL_META_NAMES,
    )

    base_names = get_feature_names()
    n_feat = len(snapshots[0].features) if snapshots else 0
    n_meta = n_feat - len(base_names)
    if n_meta == len(SIGNAL_META_EXTENDED_NAMES):
        feature_names = base_names + SIGNAL_META_EXTENDED_NAMES
    else:
        feature_names = base_names + SIGNAL_META_NAMES

    X = np.array([s.features for s in snapshots], dtype=np.float32)
    y_win = np.array([s.win for s in snapshots], dtype=np.int32)
    years = np.array([s.year for s in snapshots])
    unique_years = sorted(set(int(y) for y in years))

    aucs = []

    for held_year in unique_years:
        train_mask = years != held_year
        test_mask = years == held_year
        n_test = int(test_mask.sum())
        if n_test < 10:
            continue

        X_train = X[train_mask]
        y_train = y_win[train_mask]
        X_test = X[test_mask]
        y_test = y_win[test_mask]

        names = feature_names if len(feature_names) == X.shape[1] else None
        X_train_df = pd.DataFrame(X_train, columns=names)
        X_test_df = pd.DataFrame(X_test, columns=names)

        model = lgb.LGBMClassifier(
            is_unbalance=True,
            verbose=-1,
            n_jobs=-1,
            **params,
        )
        model.fit(X_train_df, y_train)

        probs = model.predict_proba(X_test_df)[:, 1]
        try:
            auc = roc_auc_score(y_test, probs)
        except ValueError:
            auc = 0.5
        aucs.append(auc)

    return float(np.mean(aucs)) if aucs else 0.5


def run_optuna(
    snapshots: List[TradeSnapshot],
    n_trials: int = 100,
    timeout: int = 3600,
    verbose: bool = True,
) -> dict:
    """Run Optuna study, return best params and AUC."""
    try:
        import optuna
    except ImportError:
        print("\nERROR: optuna not installed. Run: pip3 install optuna")
        sys.exit(1)

    if not verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 200),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
        }
        return loo_cv_auc(snapshots, params)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_trial
    if verbose:
        print(f"\n{'='*60}")
        print("OPTUNA TUNING RESULTS")
        print(f"{'='*60}")
        print(f"  Trials completed: {len(study.trials)}")
        print(f"  Best AUC:         {best.value:.4f}")
        print(f"  Best params:")
        for k, v in best.params.items():
            print(f"    {k:<22s} = {v}")

    return {
        'best_auc': best.value,
        'best_params': best.params,
        'n_trials': len(study.trials),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Optuna hyperparameter tuning for LightGBM signal quality model')
    parser.add_argument('--tsla', type=str, default='data/TSLAMin.txt',
                        help='Path to TSLA minute data')
    parser.add_argument('--spy', type=str, default=None,
                        help='Path to SPY minute data (optional)')
    parser.add_argument('--years', type=str, default='2015-2024',
                        help='Year range (e.g., 2015-2024)')
    parser.add_argument('--eval-interval', type=int, default=6,
                        help='Bars between evaluations')
    parser.add_argument('--min-confidence', type=float, default=0.45,
                        help='Minimum signal confidence')
    parser.add_argument('--capital', type=float, default=100_000.0,
                        help='Initial capital')
    parser.add_argument('--n-trials', type=int, default=100,
                        help='Number of Optuna trials')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout in seconds')
    parser.add_argument('--output', type=str,
                        default='v15/validation/tuned_params.json',
                        help='Output path for best params JSON')
    parser.add_argument('--bounce-cap', type=float, default=12.0,
                        help='Max exposure cap multiplier for bounce signals (default: 12.0, c9/Arch418)')
    parser.add_argument('--max-trade-usd', type=float, default=1_000_000.0,
                        help='Hard dollar cap per trade (default: 1000000, Arch418)')
    args = parser.parse_args()

    # Parse year range
    parts = args.years.split('-')
    start_year = int(parts[0])
    end_year = int(parts[1]) if len(parts) > 1 else start_year
    years = list(range(start_year, end_year + 1))

    # Build dataset
    snapshots = build_dataset(
        tsla_path=args.tsla,
        spy_path=args.spy,
        years=years,
        eval_interval=args.eval_interval,
        min_confidence=args.min_confidence,
        capital=args.capital,
        bounce_cap=args.bounce_cap,
        max_trade_usd=args.max_trade_usd,
    )

    if len(snapshots) < 100:
        print(f"\nERROR: Only {len(snapshots)} snapshots — need at least 100")
        sys.exit(1)

    # Nested Optuna: hold out last 2 years as outer test to prevent leak
    all_years = sorted(set(s.year for s in snapshots))
    if len(all_years) >= 5:
        outer_test_years = set(all_years[-2:])
        tune_snapshots = [s for s in snapshots if s.year not in outer_test_years]
        test_snapshots = [s for s in snapshots if s.year in outer_test_years]
        print(f"\n  Nested CV: Optuna tunes on {sorted(set(s.year for s in tune_snapshots))}")
        print(f"  Held-out test years: {sorted(outer_test_years)} ({len(test_snapshots)} trades)")
    else:
        tune_snapshots = snapshots
        test_snapshots = []
        print(f"\n  WARNING: <5 years, can't hold out outer test set")

    # Run Optuna on tune subset only
    t0 = time.time()
    result = run_optuna(
        tune_snapshots,
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    elapsed = time.time() - t0

    # Evaluate on held-out years (unbiased)
    if test_snapshots:
        honest_auc = loo_cv_auc(test_snapshots, result['best_params'])
        result['honest_auc'] = honest_auc
        print(f"\n  Tuned AUC (inner):   {result['best_auc']:.4f} (Optuna saw these folds)")
        print(f"  Honest AUC (outer):  {honest_auc:.4f} (Optuna never saw these years)")
        gap = result['best_auc'] - honest_auc
        if gap > 0.02:
            print(f"  ⚠️  Gap={gap:.3f} — tuning overfit detected")
        else:
            print(f"  Gap={gap:.3f} — acceptable")

    # Save best params
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  Elapsed: {elapsed:.0f}s")
    print(f"  Saved to: {output_path}")

    # Compare with baseline
    baseline_auc = 0.776
    delta = result['best_auc'] - baseline_auc
    print(f"\n  Baseline AUC: {baseline_auc:.3f}")
    print(f"  Tuned AUC:    {result['best_auc']:.3f} ({delta:+.3f})")


if __name__ == '__main__':
    main()
