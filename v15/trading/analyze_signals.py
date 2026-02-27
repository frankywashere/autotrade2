#!/usr/bin/env python3
"""
Analyze signal distribution across all horizons.

Examines what signals are available, their confidence levels,
and which ones would be profitable.
"""
import sys
import time
from pathlib import Path
from collections import defaultdict


def analyze(checkpoint_path: str, calibration_path: str = None):
    """Analyze signal distribution."""
    from v15.inference import Predictor
    from v15.trading.run_backtest import fetch_data, fetch_native_tf_data
    from v15.trading.signals import RegimeAdaptiveSignalEngine, SignalType, MarketRegime

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

    # Collect all signals with context
    all_signals = []  # (bar_idx, horizon, signal, price, mom_1d, mom_3d)
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
                    })
        except Exception:
            pass

        pct = int((bar_idx - start_bar) / max(total_bars - start_bar, 1) * 100)
        if pct % 20 == 0 and bar_idx == start_bar + (total_bars - start_bar) * pct // 100:
            print(f"  [{pct}%]")

    print(f"\nTotal signal observations: {len(all_signals)}")

    # Analysis by horizon
    for horizon in ['short', 'medium', 'long']:
        sigs = [s for s in all_signals if s['horizon'] == horizon]
        if not sigs:
            continue

        actionable = [s for s in sigs if s['actionable']]
        longs = [s for s in actionable if s['signal_type'] in ('LONG', 'long')]
        shorts = [s for s in actionable if s['signal_type'] in ('SHORT', 'short')]
        flats = [s for s in actionable if s['signal_type'] in ('FLAT', 'flat')]

        # Confidence distribution
        confs = [s['confidence'] for s in actionable]
        if confs:
            conf_buckets = defaultdict(int)
            for c in confs:
                bucket = f"{int(c * 10) / 10:.1f}-{int(c * 10) / 10 + 0.1:.1f}"
                conf_buckets[bucket] += 1

        print(f"\n{'='*60}")
        print(f"HORIZON: {horizon.upper()}")
        print(f"{'='*60}")
        print(f"  Total observations: {len(sigs)}")
        print(f"  Actionable:         {len(actionable)} ({len(actionable)/len(sigs)*100:.0f}%)")
        print(f"  LONG signals:       {len(longs)}")
        print(f"  SHORT signals:      {len(shorts)}")
        print(f"  FLAT signals:       {len(flats)}")
        # Show unique signal types for debugging
        unique_types = set(s['signal_type'] for s in actionable)
        print(f"  Unique types:       {unique_types}")

        if confs:
            print(f"\n  Confidence distribution (actionable):")
            for bucket in sorted(conf_buckets.keys()):
                count = conf_buckets[bucket]
                bar = '#' * (count * 40 // max(conf_buckets.values()))
                print(f"    {bucket}: {count:4d} {bar}")

            # High confidence breakdown
            for threshold in [0.65, 0.70, 0.75, 0.80, 0.85]:
                high = [s for s in actionable if s['confidence'] >= threshold]
                high_long = [s for s in high if s['signal_type'] == 'LONG']
                high_short = [s for s in high if s['signal_type'] == 'SHORT']
                if high:
                    print(f"\n  conf >= {threshold:.2f}: {len(high)} signals "
                          f"({len(high_long)}L/{len(high_short)}S)")
                    # Show primary TFs
                    tf_counts = defaultdict(int)
                    for s in high:
                        tf_counts[s['primary_tf']] += 1
                    print(f"    TFs: {dict(tf_counts)}")
                    # Show regimes
                    regime_counts = defaultdict(int)
                    for s in high:
                        regime_counts[s['regime']] += 1
                    print(f"    Regimes: {dict(regime_counts)}")

        # Forward return analysis for actionable signals with conf >= 0.70
        print(f"\n  --- Forward Return Analysis ---")
        for threshold in [0.70, 0.75, 0.80]:
            high_sigs = [s for s in actionable if s['confidence'] >= threshold]
            if not high_sigs:
                continue

            # Look forward 12, 24, 48 bars to see what happens
            returns = {12: [], 24: [], 48: [], 78: []}
            for s in high_sigs:
                # Skip FLAT signals for forward return analysis
                if s['signal_type'] in ('FLAT', 'flat'):
                    continue
                for fwd_bars, rets in returns.items():
                    target_idx = s['bar_idx'] + fwd_bars
                    if target_idx < total_bars:
                        fwd_price = float(tsla_df.iloc[target_idx]['close'])
                        if s['signal_type'] in ('LONG', 'long'):
                            ret = (fwd_price - s['price']) / s['price']
                        else:
                            ret = (s['price'] - fwd_price) / s['price']
                        rets.append(ret)

            print(f"\n  conf >= {threshold:.2f} ({len(high_sigs)} signals):")
            for fwd_bars, rets in returns.items():
                if rets:
                    avg = sum(rets) / len(rets)
                    win_rate = sum(1 for r in rets if r > 0) / len(rets)
                    print(f"    +{fwd_bars} bars ({fwd_bars*5/60:.0f}h): "
                          f"avg={avg:+.2%}, WR={win_rate:.0%}, n={len(rets)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze signals')
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
