#!/usr/bin/env python3
"""
Parameter sweep for the trading engine.

Architecture: Precompute predictions once, then sweep parameters in-memory.
This is ~72x faster than re-running predictions for each parameter combo.

Usage:
    python3 -m v15.trading.param_sweep [--checkpoint PATH]
"""
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CachedBar:
    """Precomputed data for a single evaluation bar."""
    bar_idx: int
    price: float
    high: float
    low: float
    timestamp: object  # pandas Timestamp
    # Signals per horizon
    horizon_signals: dict  # horizon -> TradeSignal
    unified_hazard: Optional[float]
    # Momentum
    mom_1d: float
    mom_3d: float
    # Volatility
    atr_pct: float = 0.02  # ATR as fraction of price


def run_sweep(checkpoint_path: str, calibration_path: str = None):
    """Run parameter sweep with precomputed predictions."""
    from v15.inference import Predictor
    from v15.trading.run_backtest import fetch_data, fetch_native_tf_data

    # Load model once
    print(f"[MODEL] Loading checkpoint: {checkpoint_path}")
    predictor = Predictor.load(checkpoint_path, calibration_path=calibration_path)
    print("[MODEL] Loaded")

    # Fetch data once
    tsla_df, spy_df, vix_df = fetch_data(60)
    native_data = fetch_native_tf_data()
    print()

    # === Phase 1: Precompute all predictions ===
    print("[PRECOMPUTE] Running predictions for all evaluation bars...")
    cached_bars = _precompute_predictions(
        predictor, tsla_df, spy_df, vix_df, native_data
    )
    print(f"[PRECOMPUTE] Cached {len(cached_bars)} evaluation bars\n")

    # === Phase 2: Sweep parameters using cached predictions ===
    param_grid = {
        'min_confidence': [0.73, 0.74, 0.75, 0.78],
        'mom_1d_threshold': [0.0, -0.005, -0.01],
        'mom_3d_threshold': [-0.01, -0.02],
        'max_position_pct': [0.35, 0.40, 0.50],
        'bounce_min_confidence': [0.65, 0.68, 0.70, 0.99],
        'reentry_cooldown': [12, 24],
    }

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    total = len(combinations)

    print(f"Sweeping {total} parameter combinations (in-memory, fast)...")
    print(f"{'='*130}")
    print(f"{'#':>4s} {'MinConf':>7s} {'Mom1d':>7s} {'Mom3d':>7s} {'MaxPos':>6s} {'Bounce':>6s} {'CD':>3s} | "
          f"{'Trades':>6s} {'WR%':>5s} {'P&L':>10s} {'PF':>6s} {'Sharpe':>7s} {'MaxDD':>6s}")
    print(f"{'-'*130}")

    results = []

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))

        try:
            result = _simulate_with_params(cached_bars, tsla_df, params)
            results.append({**params, **result})

            # Print row
            bounce_str = f"{params.get('bounce_min_confidence', 0.99):6.2f}"
            cd_str = f"{params.get('reentry_cooldown', 24):3d}"
            print(
                f"{idx+1:4d} {params['min_confidence']:7.2f} "
                f"{params['mom_1d_threshold']:7.3f} "
                f"{params['mom_3d_threshold']:7.3f} "
                f"{params['max_position_pct']:6.2f} {bounce_str} {cd_str} | "
                f"{result['trades']:6d} {result['win_rate']:5.1f} "
                f"${result['pnl']:9,.2f} {result['pf']:6.2f} "
                f"{result['sharpe']:7.2f} {result['max_dd']:6.1%}"
            )
        except Exception as e:
            print(f"{idx+1:4d} ERROR: {e}")

    print(f"{'='*130}")

    # Sort results
    valid = [r for r in results if r['trades'] >= 3]
    if not valid:
        print("No valid results (minimum 3 trades)")
        return

    # Top 10 by P&L
    by_pnl = sorted(valid, key=lambda x: x['pnl'], reverse=True)[:10]
    print("\n--- TOP 10 BY P&L ---")
    for r in by_pnl:
        print(
            f"  conf={r['min_confidence']:.2f} "
            f"m1d={r['mom_1d_threshold']:+.3f} "
            f"m3d={r['mom_3d_threshold']:+.3f} "
            f"pos={r['max_position_pct']:.2f} "
            f"bnc={r.get('bounce_min_confidence', 0.99):.2f} "
            f"cd={r.get('reentry_cooldown', 24)} -> "
            f"${r['pnl']:,.0f} PF={r['pf']:.2f} "
            f"Sharpe={r['sharpe']:.2f} "
            f"DD={r['max_dd']:.1%} "
            f"({r['trades']}T, {r['win_rate']:.0f}%WR)"
        )

    # Top 10 by Sharpe
    by_sharpe = sorted(valid, key=lambda x: x['sharpe'], reverse=True)[:10]
    print("\n--- TOP 10 BY SHARPE ---")
    for r in by_sharpe:
        print(
            f"  conf={r['min_confidence']:.2f} "
            f"m1d={r['mom_1d_threshold']:+.3f} "
            f"m3d={r['mom_3d_threshold']:+.3f} "
            f"pos={r['max_position_pct']:.2f} "
            f"bnc={r.get('bounce_min_confidence', 0.99):.2f} "
            f"cd={r.get('reentry_cooldown', 24)} -> "
            f"Sharpe={r['sharpe']:.2f} "
            f"${r['pnl']:,.0f} PF={r['pf']:.2f} "
            f"DD={r['max_dd']:.1%} "
            f"({r['trades']}T, {r['win_rate']:.0f}%WR)"
        )

    # Top 10 by profit factor
    by_pf = sorted(valid, key=lambda x: x['pf'], reverse=True)[:10]
    print("\n--- TOP 10 BY PROFIT FACTOR ---")
    for r in by_pf:
        print(
            f"  conf={r['min_confidence']:.2f} "
            f"m1d={r['mom_1d_threshold']:+.3f} "
            f"m3d={r['mom_3d_threshold']:+.3f} "
            f"pos={r['max_position_pct']:.2f} "
            f"bnc={r.get('bounce_min_confidence', 0.99):.2f} "
            f"cd={r.get('reentry_cooldown', 24)} -> "
            f"PF={r['pf']:.2f} "
            f"${r['pnl']:,.0f} Sharpe={r['sharpe']:.2f} "
            f"DD={r['max_dd']:.1%} "
            f"({r['trades']}T, {r['win_rate']:.0f}%WR)"
        )

    # Best balanced (Sharpe > 0.5, PF > 1.5, DD < 3%)
    balanced = [r for r in valid if r['sharpe'] > 0.5 and r['pf'] > 1.5 and r['max_dd'] < 0.03]
    if balanced:
        best = sorted(balanced, key=lambda x: x['pnl'], reverse=True)[0]
        print(f"\n--- BEST BALANCED (Sharpe>0.5, PF>1.5, DD<3%) ---")
        print(
            f"  conf={best['min_confidence']:.2f} "
            f"m1d={best['mom_1d_threshold']:+.3f} "
            f"m3d={best['mom_3d_threshold']:+.3f} "
            f"pos={best['max_position_pct']:.2f} -> "
            f"${best['pnl']:,.0f} PF={best['pf']:.2f} "
            f"Sharpe={best['sharpe']:.2f} "
            f"DD={best['max_dd']:.1%} "
            f"({best['trades']}T, {best['win_rate']:.0f}%WR)"
        )
    else:
        print("\nNo balanced results found (Sharpe>0.5, PF>1.5, DD<3%)")

    # Best conservative (DD < 1.5%, trades >= 3)
    conservative = [r for r in valid if r['max_dd'] < 0.015 and r['pnl'] > 0]
    if conservative:
        best_c = sorted(conservative, key=lambda x: x['pnl'], reverse=True)[0]
        print(f"\n--- BEST CONSERVATIVE (DD<1.5%, profitable) ---")
        print(
            f"  conf={best_c['min_confidence']:.2f} "
            f"m1d={best_c['mom_1d_threshold']:+.3f} "
            f"m3d={best_c['mom_3d_threshold']:+.3f} "
            f"pos={best_c['max_position_pct']:.2f} -> "
            f"${best_c['pnl']:,.0f} PF={best_c['pf']:.2f} "
            f"Sharpe={best_c['sharpe']:.2f} "
            f"DD={best_c['max_dd']:.1%} "
            f"({best_c['trades']}T, {best_c['win_rate']:.0f}%WR)"
        )


def _precompute_predictions(predictor, tsla_df, spy_df, vix_df, native_data):
    """Run model predictions once for all evaluation bars."""
    from v15.trading.signals import RegimeAdaptiveSignalEngine

    signal_engine = RegimeAdaptiveSignalEngine()
    eval_interval = 12
    start_bar = 1000
    total_bars = len(tsla_df)

    cached_bars = []
    prev_hazard = None
    n_bars = (total_bars - start_bar) // eval_interval
    last_pct = -1

    for bar_idx in range(start_bar, total_bars, eval_interval):
        current_price = float(tsla_df.iloc[bar_idx]['close'])
        high = float(tsla_df.iloc[bar_idx]['high'])
        low = float(tsla_df.iloc[bar_idx]['low'])

        # Compute ATR (78-bar / 1 trading day)
        from v15.trading.backtester import _compute_atr
        atr_pct = _compute_atr(tsla_df, bar_idx, period=78)

        # Momentum
        def _mom(lookback):
            if bar_idx >= lookback:
                past = float(tsla_df.iloc[bar_idx - lookback]['close'])
                return (current_price - past) / past
            return 0.0

        mom_1d = _mom(78)
        mom_3d = _mom(234)

        horizon_signals = {}
        unified_hazard = prev_hazard

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
                unified_hazard = unified.hazard
        except Exception:
            pass

        cached_bars.append(CachedBar(
            bar_idx=bar_idx,
            price=current_price,
            high=high,
            low=low,
            timestamp=tsla_df.index[bar_idx],
            horizon_signals=horizon_signals,
            unified_hazard=unified_hazard,
            mom_1d=mom_1d,
            mom_3d=mom_3d,
            atr_pct=atr_pct,
        ))

        # Progress
        pct = int((bar_idx - start_bar) / max(total_bars - start_bar, 1) * 100)
        if pct >= last_pct + 10:
            last_pct = pct
            print(f"  [{pct}%] bar {bar_idx}/{total_bars}")

    return cached_bars


def _simulate_with_params(cached_bars, tsla_df, params):
    """Simulate trading with given parameters using precomputed predictions.

    This is pure in-memory computation — no model calls.
    """
    from v15.trading.signals import SignalType, MarketRegime
    from v15.trading.position_sizer import PositionSizer
    from v15.trading.metrics import TradeMetrics, Trade
    from v15.trading.backtester import OpenPosition, _to_datetime
    from v15.config import TF_TO_HORIZON

    initial_capital = 100000.0
    sizer = PositionSizer(
        capital=initial_capital,
        max_position_pct=params['max_position_pct'],
    )

    metrics = TradeMetrics()
    equity = initial_capital
    sizer.current_equity = equity
    sizer.peak_equity = equity

    open_position = None

    # Re-entry tracking
    last_exit_bar = 0
    last_exit_direction = None
    last_exit_profitable = False
    REENTRY_COOLDOWN = params.get('reentry_cooldown', 24)

    HORIZON_ATR_MULT = {'short': 1.5, 'medium': 2.0, 'long': 2.5}
    HORIZON_TRAIL_PCT = {'short': 0.015, 'medium': 0.020, 'long': 0.030}
    HORIZON_MAX_HOLD = {'short': 78, 'medium': 156, 'long': 390}

    for cb in cached_bars:
        bar_idx = cb.bar_idx
        current_price = cb.price

        # Check exit
        if open_position is not None:
            high = cb.high
            low = cb.low
            bars_held = bar_idx - open_position.entry_bar

            exit_price, exit_reason = None, None

            # Progressive trail tightening (matches backtester)
            horizon = TF_TO_HORIZON.get(open_position.primary_tf, 'medium')
            max_hold = min(HORIZON_MAX_HOLD.get(horizon, 390), 390)
            hold_pct = min(1.0, bars_held / max(max_hold, 1))
            tightening = 1.0 - hold_pct * 0.4
            effective_trail = open_position.trailing_stop_pct * tightening

            if open_position.direction == 'long':
                if high > open_position.best_price:
                    open_position.best_price = high
                if low <= open_position.stop_loss_price:
                    exit_price, exit_reason = open_position.stop_loss_price, 'stop_loss'
                elif open_position.best_price > open_position.entry_price:
                    ts = open_position.best_price * (1 - effective_trail)
                    if ts > open_position.stop_loss_price and low <= ts:
                        exit_price, exit_reason = ts, 'trailing_stop'
                if exit_price is None and high >= open_position.take_profit_price:
                    exit_price, exit_reason = open_position.take_profit_price, 'take_profit'
            else:  # short
                if open_position.best_price == 0 or low < open_position.best_price:
                    open_position.best_price = low
                if high >= open_position.stop_loss_price:
                    exit_price, exit_reason = open_position.stop_loss_price, 'stop_loss'
                elif open_position.best_price > 0 and open_position.best_price < open_position.entry_price:
                    ts = open_position.best_price * (1 + effective_trail)
                    if ts < open_position.stop_loss_price and high >= ts:
                        exit_price, exit_reason = ts, 'trailing_stop'
                if exit_price is None and low <= open_position.take_profit_price:
                    exit_price, exit_reason = open_position.take_profit_price, 'take_profit'

            if exit_price is None and bars_held >= max_hold:
                exit_price, exit_reason = current_price, 'timeout'

            if exit_price is not None:
                slippage = exit_price * 0.0001
                if open_position.direction == 'long':
                    actual_exit = exit_price - slippage
                    raw_pnl = (actual_exit - open_position.entry_price) * open_position.shares
                else:
                    actual_exit = exit_price + slippage
                    raw_pnl = (open_position.entry_price - actual_exit) * open_position.shares
                commission = open_position.shares * 0.005 * 2
                net_pnl = raw_pnl - commission
                trade = Trade(
                    entry_time=open_position.entry_time,
                    exit_time=_to_datetime(cb.timestamp),
                    direction=open_position.direction,
                    entry_price=open_position.entry_price,
                    exit_price=actual_exit,
                    shares=open_position.shares,
                    pnl=net_pnl,
                    pnl_pct=net_pnl / (open_position.entry_price * open_position.shares),
                    commission=commission,
                    slippage=slippage * open_position.shares * 2,
                    signal_confidence=open_position.signal_confidence,
                    regime=open_position.regime,
                    primary_tf=open_position.primary_tf,
                    exit_reason=exit_reason,
                    hold_bars=bars_held,
                )
                metrics.add_trade(trade)
                equity += net_pnl
                sizer.update_equity(equity)
                sizer.record_trade_result(net_pnl > 0)
                last_exit_bar = bar_idx
                last_exit_direction = open_position.direction
                last_exit_profitable = net_pnl > 0
                open_position = None

        # Entry logic using cached signals
        if open_position is None and cb.horizon_signals:
            best_signal = None
            best_score = -1.0
            for horizon, sig in cb.horizon_signals.items():
                if not sig.actionable:
                    continue

                if horizon == 'long':
                    # Trend strategy
                    if sig.confidence < params['min_confidence']:
                        continue
                    if sig.regime.regime == MarketRegime.TRANSITIONING:
                        continue
                    if sig.signal_type == SignalType.LONG:
                        if cb.mom_1d < params['mom_1d_threshold'] or cb.mom_3d < params['mom_3d_threshold']:
                            continue
                    elif sig.signal_type == SignalType.SHORT:
                        if cb.mom_1d > -params['mom_1d_threshold'] or cb.mom_3d > -params['mom_3d_threshold']:
                            continue
                    score = sig.confidence * sig.entry_urgency * 2.0
                    if score > best_score:
                        best_score = score
                        best_signal = sig

                elif horizon == 'short':
                    # Bounce strategy: ranging only
                    bounce_conf = params.get('bounce_min_confidence', 0.75)
                    if sig.confidence < bounce_conf:
                        continue
                    if sig.regime.regime != MarketRegime.RANGING:
                        continue
                    score = sig.confidence * sig.entry_urgency
                    if score > best_score:
                        best_score = score
                        best_signal = sig

            if best_signal is not None and best_signal.entry_urgency > 0.3:
                # Re-entry cooldown check
                signal_dir = 'long' if best_signal.signal_type == SignalType.LONG else 'short'
                in_cooldown = (
                    last_exit_profitable
                    and (bar_idx - last_exit_bar) < REENTRY_COOLDOWN
                    and signal_dir == last_exit_direction
                )
                if in_cooldown:
                    best_signal = None

            if best_signal is not None and best_signal.entry_urgency > 0.3:
                position = sizer.size_position(best_signal, current_price, atr_pct=cb.atr_pct)
                if position.should_trade:
                    # Confidence scaling
                    conf_scale = max(0.5, min(1.5,
                        0.7 + (best_signal.confidence - 0.72) * 10.0
                    ))

                    # Cross-horizon agreement bonus (matches backtester)
                    signal_dir = best_signal.signal_type
                    agreeing_horizons = 0
                    total_horizons = 0
                    for h, hsig in cb.horizon_signals.items():
                        if hsig.signal_type != SignalType.FLAT:
                            total_horizons += 1
                            if hsig.signal_type == signal_dir:
                                agreeing_horizons += 1
                    if total_horizons >= 2:
                        agreement_pct = agreeing_horizons / total_horizons
                        cross_horizon_mult = 0.8 + agreement_pct * 0.4
                    else:
                        cross_horizon_mult = 1.0

                    total_scale = conf_scale * cross_horizon_mult
                    if total_scale != 1.0:
                        position.shares = max(1, int(position.shares * total_scale))

                    slippage = current_price * 0.0001
                    signal_dir = 'long' if best_signal.signal_type == SignalType.LONG else 'short'
                    entry_price = current_price + slippage if signal_dir == 'long' else current_price - slippage

                    # ATR-adaptive trailing stop
                    h = TF_TO_HORIZON.get(best_signal.primary_tf, 'medium')
                    atr_mult = HORIZON_ATR_MULT.get(h, 2.0)
                    trail_pct = cb.atr_pct * atr_mult
                    min_trail = HORIZON_TRAIL_PCT.get(h, 0.020) * 0.5
                    trail_pct = max(min_trail, min(0.06, trail_pct))

                    if signal_dir == 'long':
                        stop_price = entry_price * (1 - position.stop_loss_pct)
                        tp_price = entry_price * (1 + position.take_profit_pct)
                    else:
                        stop_price = entry_price * (1 + position.stop_loss_pct)
                        tp_price = entry_price * (1 - position.take_profit_pct)

                    open_position = OpenPosition(
                        entry_time=_to_datetime(cb.timestamp),
                        entry_bar=bar_idx,
                        direction=signal_dir,
                        entry_price=entry_price,
                        shares=position.shares,
                        stop_loss_price=stop_price,
                        take_profit_price=tp_price,
                        signal_confidence=best_signal.confidence,
                        regime=best_signal.regime.regime.value,
                        primary_tf=best_signal.primary_tf,
                        commission_entry=position.shares * 0.005,
                        best_price=entry_price,
                        trailing_stop_pct=trail_pct,
                    )

    # Close remaining
    if open_position is not None:
        last_bar = cached_bars[-1]
        final_price = last_bar.price
        slippage = final_price * 0.0001
        actual_exit = final_price - slippage
        raw_pnl = (actual_exit - open_position.entry_price) * open_position.shares
        commission = open_position.shares * 0.005 * 2
        net_pnl = raw_pnl - commission
        equity += net_pnl
        trade = Trade(
            entry_time=open_position.entry_time,
            exit_time=_to_datetime(last_bar.timestamp),
            direction=open_position.direction,
            entry_price=open_position.entry_price,
            exit_price=actual_exit,
            shares=open_position.shares,
            pnl=net_pnl,
            pnl_pct=net_pnl / (open_position.entry_price * open_position.shares),
            commission=commission,
            slippage=slippage * open_position.shares * 2,
            signal_confidence=open_position.signal_confidence,
            regime=open_position.regime,
            primary_tf=open_position.primary_tf,
            exit_reason='end_of_data',
            hold_bars=last_bar.bar_idx - open_position.entry_bar,
        )
        metrics.add_trade(trade)

    return {
        'trades': metrics.total_trades,
        'win_rate': metrics.win_rate * 100,
        'pnl': metrics.total_pnl,
        'pf': metrics.profit_factor,
        'sharpe': metrics.sharpe_ratio,
        'max_dd': metrics.max_drawdown,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Parameter sweep')
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
    run_sweep(checkpoint, calibration)
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
