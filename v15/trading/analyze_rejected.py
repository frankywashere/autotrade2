#!/usr/bin/env python3
"""
Analyze REJECTED signals to find hidden profitable opportunities.

For every signal our filters reject, compute forward returns to see
if we're leaving money on the table.

Usage:
    python3 -m v15.trading.analyze_rejected [--checkpoint PATH]
"""
import sys
import time
from pathlib import Path
from collections import defaultdict


def analyze(checkpoint_path: str, calibration_path: str = None):
    """Analyze rejected signals and their forward returns."""
    from v15.inference import Predictor
    from v15.trading.run_backtest import fetch_data, fetch_native_tf_data
    from v15.trading.signals import (
        RegimeAdaptiveSignalEngine, SignalType, MarketRegime,
    )
    from v15.trading.backtester import _compute_atr

    print(f"[MODEL] Loading checkpoint: {checkpoint_path}")
    predictor = Predictor.load(checkpoint_path, calibration_path=calibration_path)
    print("[MODEL] Loaded")

    tsla_df, spy_df, vix_df = fetch_data(60)
    native_data = fetch_native_tf_data()
    print()

    signal_engine = RegimeAdaptiveSignalEngine()
    eval_interval = 12
    start_bar = 1000
    total_bars = len(tsla_df)

    # Current filter parameters
    HORIZON_MIN_CONF = {'short': 0.70, 'medium': 0.99, 'long': 0.75}
    MOM_1D_THRESHOLD = -0.005
    MOM_3D_THRESHOLD = -0.01

    # Collect ALL signals with rejection reason
    all_signals = []
    prev_hazard = None

    for bar_idx in range(start_bar, total_bars, eval_interval):
        current_price = float(tsla_df.iloc[bar_idx]['close'])

        def _mom(lookback):
            if bar_idx >= lookback:
                past = float(tsla_df.iloc[bar_idx - lookback]['close'])
                return (current_price - past) / past
            return 0.0

        mom_1d = _mom(78)
        mom_3d = _mom(234)

        try:
            prediction = predictor.predict_with_per_tf(
                tsla_df.iloc[:bar_idx + 1],
                spy_df.iloc[:bar_idx + 1],
                vix_df.iloc[:bar_idx + 1],
                native_bars_by_tf=native_data,
            )
            if prediction and prediction.per_tf_predictions:
                horizon_signals = signal_engine.generate_horizon_signals(
                    per_tf_predictions=prediction.per_tf_predictions,
                    previous_hazard=prev_hazard,
                )
                unified = signal_engine.generate_signal(
                    per_tf_predictions=prediction.per_tf_predictions,
                    previous_hazard=prev_hazard,
                )
                prev_hazard = unified.hazard

                for horizon, sig in horizon_signals.items():
                    # Determine rejection reason
                    rejection = None
                    accepted = False

                    min_conf = HORIZON_MIN_CONF.get(horizon, 0.99)
                    if sig.confidence < min_conf:
                        rejection = f'low_conf({sig.confidence:.2f}<{min_conf})'
                    elif not sig.actionable:
                        rejection = 'not_actionable'
                    elif horizon == 'long':
                        if sig.regime.regime == MarketRegime.TRANSITIONING:
                            rejection = 'transitioning_regime'
                        elif sig.signal_type == SignalType.LONG:
                            if mom_1d < MOM_1D_THRESHOLD:
                                rejection = f'neg_mom_1d({mom_1d:+.3f})'
                            elif mom_3d < MOM_3D_THRESHOLD:
                                rejection = f'neg_mom_3d({mom_3d:+.3f})'
                            else:
                                accepted = True
                        elif sig.signal_type == SignalType.SHORT:
                            if mom_1d > -MOM_1D_THRESHOLD:
                                rejection = f'pos_mom_1d({mom_1d:+.3f})'
                            elif mom_3d > -MOM_3D_THRESHOLD:
                                rejection = f'pos_mom_3d({mom_3d:+.3f})'
                            else:
                                accepted = True
                        elif sig.signal_type == SignalType.FLAT:
                            rejection = 'flat_signal'
                        else:
                            accepted = True
                    elif horizon == 'short':
                        if sig.regime.regime != MarketRegime.RANGING:
                            rejection = f'not_ranging({sig.regime.regime.value})'
                        else:
                            accepted = True
                    elif horizon == 'medium':
                        rejection = 'medium_disabled'

                    if rejection is None and not accepted:
                        rejection = 'unknown'

                    # Compute forward returns
                    fwd_returns = {}
                    for fwd_bars in [12, 24, 48, 78]:
                        target_idx = bar_idx + fwd_bars
                        if target_idx < total_bars:
                            fwd_price = float(tsla_df.iloc[target_idx]['close'])
                            if sig.signal_type == SignalType.LONG:
                                fwd_returns[fwd_bars] = (fwd_price - current_price) / current_price
                            elif sig.signal_type == SignalType.SHORT:
                                fwd_returns[fwd_bars] = (current_price - fwd_price) / current_price
                            else:
                                fwd_returns[fwd_bars] = 0.0

                    all_signals.append({
                        'bar_idx': bar_idx,
                        'horizon': horizon,
                        'signal_type': sig.signal_type.value,
                        'confidence': sig.confidence,
                        'entry_urgency': sig.entry_urgency,
                        'actionable': sig.actionable,
                        'regime': sig.regime.regime.value,
                        'primary_tf': sig.primary_tf,
                        'price': current_price,
                        'mom_1d': mom_1d,
                        'mom_3d': mom_3d,
                        'rejection': rejection,
                        'accepted': accepted,
                        'fwd_returns': fwd_returns,
                    })
        except Exception:
            pass

    print(f"\nTotal signal observations: {len(all_signals)}")
    accepted_count = sum(1 for s in all_signals if s['accepted'])
    rejected_count = sum(1 for s in all_signals if not s['accepted'])
    print(f"Accepted: {accepted_count}, Rejected: {rejected_count}")

    # === Analyze rejections by reason ===
    print(f"\n{'='*70}")
    print(f"REJECTION ANALYSIS")
    print(f"{'='*70}")

    # Group rejections
    by_reason = defaultdict(list)
    for s in all_signals:
        if not s['accepted']:
            # Simplify reason to category
            reason = s['rejection']
            if reason.startswith('low_conf'):
                category = 'low_confidence'
            elif reason.startswith('neg_mom') or reason.startswith('pos_mom'):
                category = 'momentum_filter'
            elif reason.startswith('not_ranging'):
                category = 'wrong_regime'
            else:
                category = reason
            by_reason[category].append(s)

    for reason in sorted(by_reason.keys()):
        sigs = by_reason[reason]
        print(f"\n--- {reason} ({len(sigs)} signals) ---")

        # Breakdown by type
        longs = [s for s in sigs if s['signal_type'] == 'long']
        shorts = [s for s in sigs if s['signal_type'] == 'short']
        flats = [s for s in sigs if s['signal_type'] == 'flat']
        print(f"  LONG: {len(longs)}, SHORT: {len(shorts)}, FLAT: {len(flats)}")

        # Confidence distribution
        confs = [s['confidence'] for s in sigs]
        if confs:
            print(f"  Confidence: min={min(confs):.2f}, max={max(confs):.2f}, "
                  f"mean={sum(confs)/len(confs):.2f}")

        # Forward returns for non-FLAT signals
        for fwd_bars in [12, 24, 48, 78]:
            rets = [s['fwd_returns'].get(fwd_bars, None) for s in sigs
                    if s['signal_type'] != 'flat' and s['fwd_returns'].get(fwd_bars) is not None]
            if rets:
                avg_ret = sum(rets) / len(rets)
                win_rate = sum(1 for r in rets if r > 0) / len(rets)
                print(f"  +{fwd_bars}bars ({fwd_bars*5/60:.0f}h): "
                      f"avg={avg_ret:+.2%}, WR={win_rate:.0%}, n={len(rets)}")

        # Top individual high-confidence rejected signals
        high_conf = sorted(
            [s for s in sigs if s['confidence'] >= 0.70 and s['signal_type'] != 'flat'],
            key=lambda x: x['confidence'],
            reverse=True
        )[:5]
        if high_conf:
            print(f"\n  Top 5 high-confidence rejected:")
            for s in high_conf:
                ret_48 = s['fwd_returns'].get(48, None)
                ret_str = f"ret48={ret_48:+.2%}" if ret_48 is not None else "ret48=N/A"
                print(f"    conf={s['confidence']:.2f} {s['signal_type']:5s} "
                      f"tf={s['primary_tf']:8s} regime={s['regime']:15s} "
                      f"mom1d={s['mom_1d']:+.3f} mom3d={s['mom_3d']:+.3f} "
                      f"{ret_str} [{s['rejection']}]")

    # === Find untapped edge: high-confidence rejected signals with positive returns ===
    print(f"\n{'='*70}")
    print(f"UNTAPPED OPPORTUNITIES (rejected but profitable)")
    print(f"{'='*70}")

    profitable_rejected = [
        s for s in all_signals
        if not s['accepted']
        and s['signal_type'] != 'flat'
        and s['confidence'] >= 0.70
        and s['fwd_returns'].get(48, 0) > 0.005  # >0.5% in 4 hours
    ]
    print(f"\nHigh-conf rejected signals with >0.5% return at 4h: {len(profitable_rejected)}")

    by_horizon = defaultdict(list)
    for s in profitable_rejected:
        by_horizon[s['horizon']].append(s)

    for horizon in ['short', 'medium', 'long']:
        sigs = by_horizon.get(horizon, [])
        if not sigs:
            continue
        avg_ret = sum(s['fwd_returns'].get(48, 0) for s in sigs) / len(sigs)
        print(f"\n  {horizon}: {len(sigs)} signals, avg 4h ret: {avg_ret:+.2%}")

        # Show reasons
        reasons = defaultdict(int)
        for s in sigs:
            reasons[s['rejection']] += 1
        for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"    {r}: {c}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze rejected signals')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--calibration', default=None)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        for c in ['/tmp/x23_best_per_tf.pt', 'models/x23_best_per_tf.pt']:
            if Path(c).exists():
                checkpoint = c
                break
    if checkpoint is None:
        print("ERROR: No checkpoint found")
        sys.exit(1)

    calibration = args.calibration
    if calibration is None:
        cp_dir = Path(checkpoint).parent
        for name in ['temperature_calibration_x23.json', 'temperature_calibration.json']:
            if (cp_dir / name).exists():
                calibration = str(cp_dir / name)
                break

    start = time.time()
    analyze(checkpoint, calibration)
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
