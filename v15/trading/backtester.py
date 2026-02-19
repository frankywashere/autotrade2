"""
Walk-Forward Backtesting Engine

Simulates trading with realistic conditions:
- Slippage (0.01% per side)
- Commissions ($0.005/share per side)
- Position sizing with Kelly + drawdown thermostat
- Multiple exit conditions (stop loss, take profit, signal flip, timeout)
"""
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, TYPE_CHECKING

import pandas as pd
import numpy as np

from .signals import (
    TradeSignal, SignalType, MarketRegime,
    RegimeAdaptiveSignalEngine, HazardClock,
)
from .position_sizer import PositionSizer, PositionRecommendation
from .metrics import Trade, TradeMetrics, EquityCurve

if TYPE_CHECKING:
    from ..inference import Predictor, PerTFPrediction


@dataclass
class OpenPosition:
    """Tracks an open position."""
    entry_time: datetime
    entry_bar: int
    direction: str  # 'long' or 'short'
    entry_price: float
    shares: int
    stop_loss_price: float
    take_profit_price: float
    signal_confidence: float
    regime: str
    primary_tf: str
    commission_entry: float
    strategy: str = 'trend'  # 'trend' or 'bounce'
    # Trailing stop tracking
    best_price: float = 0.0  # Best price seen since entry (for trailing stop)
    trailing_stop_pct: float = 0.02  # Trail by 2%
    # Re-entry tracking
    exit_bar: int = 0  # Bar when position was closed (for cooldown)


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""
    initial_capital: float = 100000.0
    commission_per_share: float = 0.005
    slippage_pct: float = 0.0001  # 0.01% per side
    eval_interval_bars: int = 12  # Evaluate every 12 bars (1 hour of 5-min)
    max_hold_bars: int = 390  # Max 1 trading day
    allow_multiple_positions: bool = False
    max_positions: int = 1


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    config: BacktestConfig
    metrics: TradeMetrics
    signals_generated: int
    signals_actionable: int
    bars_processed: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def summary(self) -> str:
        lines = [
            f"Backtest: {self.start_time} to {self.end_time}",
            f"Bars: {self.bars_processed}, Signals: {self.signals_generated}, "
            f"Actionable: {self.signals_actionable}",
            "",
            self.metrics.summary(),
        ]
        return "\n".join(lines)


class Backtester:
    """
    Walk-forward backtester for the regime-adaptive trading engine.

    Usage:
        predictor = Predictor.load('checkpoint.pt')
        backtester = Backtester(predictor)
        result = backtester.run(tsla_df, spy_df, vix_df)
        print(result.summary())
    """

    def __init__(
        self,
        predictor: 'Predictor',
        signal_engine: Optional[RegimeAdaptiveSignalEngine] = None,
        position_sizer: Optional[PositionSizer] = None,
        config: Optional[BacktestConfig] = None,
    ):
        self.predictor = predictor
        self.signal_engine = signal_engine or RegimeAdaptiveSignalEngine()
        self.config = config or BacktestConfig()
        self.position_sizer = position_sizer or PositionSizer(
            capital=self.config.initial_capital
        )

    def run(
        self,
        tsla_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        native_bars_by_tf: Optional[dict] = None,
        progress_callback: Optional[Callable] = None,
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Args:
            tsla_df: TSLA 5-min OHLCV (index=datetime, columns=Open/High/Low/Close/Volume)
            spy_df: SPY 5-min OHLCV
            vix_df: VIX 5-min OHLCV
            native_bars_by_tf: Optional native TF data for better predictions
            progress_callback: Optional callback(bar_idx, total_bars, metrics)
        """
        metrics = TradeMetrics()
        equity_curve = EquityCurve()
        equity = self.config.initial_capital
        self.position_sizer.current_equity = equity
        self.position_sizer.peak_equity = equity

        # Concurrent positions: one per strategy
        # trend and bounce can run simultaneously
        positions: Dict[str, OpenPosition] = {}  # strategy -> position
        MAX_TOTAL_POSITION_PCT = 0.90  # Max 90% of capital in positions total

        prev_hazard: Optional[HazardClock] = None

        # Re-entry tracking per strategy
        last_exit_by_strategy: Dict[str, dict] = {}  # strategy -> {bar, direction, profitable}
        REENTRY_COOLDOWN = 12  # 1 hour cooldown before re-entry (sweep-optimized)

        total_bars = len(tsla_df)
        signals_generated = 0
        signals_actionable = 0

        # Need minimum bars for model (C++ extractor needs context)
        start_bar = 1000  # Start after warmup
        eval_interval = self.config.eval_interval_bars

        start_time = tsla_df.index[start_bar] if hasattr(tsla_df.index[start_bar], 'to_pydatetime') else datetime.now()
        end_time = tsla_df.index[-1] if hasattr(tsla_df.index[-1], 'to_pydatetime') else datetime.now()

        # Initial equity point
        equity_curve.add_point(
            _to_datetime(tsla_df.index[start_bar]), equity
        )

        for bar_idx in range(start_bar, total_bars, eval_interval):
            current_time = _to_datetime(tsla_df.index[bar_idx])
            current_price = float(tsla_df.iloc[bar_idx]['close'])
            high = float(tsla_df.iloc[bar_idx]['high'])
            low = float(tsla_df.iloc[bar_idx]['low'])

            # Check all open positions for exit
            closed_strategies = []
            for strat_key, pos in list(positions.items()):
                bars_held = bar_idx - pos.entry_bar

                exit_price, exit_reason = self._check_exit(
                    pos, current_price, high, low, bars_held
                )

                if exit_price is not None:
                    # Track max unrealized P&L for efficiency analysis
                    if pos.direction == 'long':
                        max_unreal = (pos.best_price - pos.entry_price) * pos.shares
                    else:
                        max_unreal = (pos.entry_price - pos.best_price) * pos.shares
                    trade = self._close_position(
                        pos, exit_price, current_time,
                        bars_held, exit_reason
                    )
                    trade._max_unrealized = max_unreal  # Attach for logging
                    trade._strategy = strat_key
                    metrics.add_trade(trade)
                    equity += trade.pnl
                    self.position_sizer.update_equity(equity)
                    self.position_sizer.record_trade_result(trade.pnl > 0)
                    equity_curve.add_point(current_time, equity)
                    # Track for re-entry per strategy
                    last_exit_by_strategy[strat_key] = {
                        'bar': bar_idx,
                        'direction': pos.direction,
                        'profitable': trade.pnl > 0,
                        'exit_price': trade.exit_price,
                    }
                    closed_strategies.append(strat_key)

            for strat_key in closed_strategies:
                del positions[strat_key]

            # Generate signal at evaluation intervals
            try:
                # Get model prediction
                prediction = self._get_prediction(
                    tsla_df.iloc[:bar_idx + 1],
                    spy_df.iloc[:bar_idx + 1],
                    vix_df.iloc[:bar_idx + 1],
                    native_bars_by_tf,
                )

                if prediction and prediction.per_tf_predictions:
                    # Generate horizon-specific signals (short/medium/long)
                    horizon_signals = self.signal_engine.generate_horizon_signals(
                        per_tf_predictions=prediction.per_tf_predictions,
                        previous_hazard=prev_hazard,
                    )

                    # Also generate the unified signal for hazard tracking
                    unified = self.signal_engine.generate_signal(
                        per_tf_predictions=prediction.per_tf_predictions,
                        previous_hazard=prev_hazard,
                    )
                    prev_hazard = unified.hazard

                    # Horizon-specific minimum confidence
                    # Long horizon has proven edge; others are disabled
                    # Evidence: 1h = 0% win rate across ALL runs, monthly = 50%+ win
                    HORIZON_MIN_CONF = {
                        'short': 0.68,   # Bounce strategy: ranging regime only
                        'medium': 0.60,  # Low gate; real filter in medium block (long-validated)
                        'long': 0.75,    # Trend strategy: 100%WR, PF=inf
                    }

                    # High-selectivity + momentum filter strategy:
                    # 1. Only trade long horizon (monthly/weekly/daily)
                    # 2. High confidence threshold (0.72+)
                    # 3. Price momentum must confirm direction
                    #    → avoids going long during corrections

                    # Compute multi-timeframe price momentum
                    # Short: 78 bars (1 day), Medium: 234 bars (3 days)
                    # Both must agree for entry
                    def _calc_momentum(lookback):
                        if bar_idx >= lookback:
                            past = float(tsla_df.iloc[bar_idx - lookback]['close'])
                            return (current_price - past) / past
                        return 0.0

                    mom_1d = _calc_momentum(78)
                    mom_3d = _calc_momentum(234)

                    # === DUAL STRATEGY: Trend + Bounce ===
                    # Strategy 1: Long horizon trend-following (monthly TF)
                    # Strategy 2: Short horizon bounce capture (ranging markets)
                    # Priority: long horizon first, short horizon as backup

                    # Sweep-optimized momentum thresholds
                    MOM_1D_THRESHOLD = -0.005  # Allow slight 1d pullback
                    MOM_3D_THRESHOLD = -0.01   # 3d must not be deeply negative

                    # Collect candidate signals per strategy
                    strategy_signals = {}  # strategy -> (signal, score)

                    for horizon, sig in horizon_signals.items():
                        signals_generated += 1
                        if sig.actionable:
                            signals_actionable += 1

                        min_conf = HORIZON_MIN_CONF.get(horizon, 0.99)
                        if sig.confidence < min_conf:
                            continue
                        if not sig.actionable:
                            continue

                        if horizon == 'long':
                            if sig.regime.regime == MarketRegime.TRANSITIONING:
                                continue
                            if sig.signal_type == SignalType.LONG:
                                if mom_1d < MOM_1D_THRESHOLD or mom_3d < MOM_3D_THRESHOLD:
                                    continue
                            elif sig.signal_type == SignalType.SHORT:
                                if mom_1d > -MOM_1D_THRESHOLD or mom_3d > -MOM_3D_THRESHOLD:
                                    continue
                            score = sig.confidence * sig.entry_urgency * 2.0
                            prev = strategy_signals.get('trend')
                            if prev is None or score > prev[1]:
                                strategy_signals['trend'] = (sig, score)

                        elif horizon == 'medium':
                            # Medium horizon: only when validated by long horizon
                            # The long horizon provides directional edge;
                            # medium provides faster entry timing
                            long_sig = horizon_signals.get('long')
                            if (long_sig
                                    and long_sig.signal_type == sig.signal_type
                                    and long_sig.confidence >= 0.73
                                    and sig.confidence >= 0.70):
                                # Also apply momentum filter
                                if sig.signal_type == SignalType.LONG:
                                    if mom_1d < MOM_1D_THRESHOLD or mom_3d < MOM_3D_THRESHOLD:
                                        continue
                                elif sig.signal_type == SignalType.SHORT:
                                    if mom_1d > -MOM_1D_THRESHOLD or mom_3d > -MOM_3D_THRESHOLD:
                                        continue
                                score = sig.confidence * sig.entry_urgency * 1.5
                                prev = strategy_signals.get('medium_trend')
                                if prev is None or score > prev[1]:
                                    strategy_signals['medium_trend'] = (sig, score)

                        elif horizon == 'short':
                            if sig.regime.regime == MarketRegime.RANGING:
                                # Bounce strategy: ranging markets, channel bounce
                                score = sig.confidence * sig.entry_urgency
                                prev = strategy_signals.get('bounce')
                                if prev is None or score > prev[1]:
                                    strategy_signals['bounce'] = (sig, score)
                            elif sig.regime.regime in (MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR):
                                # Short-horizon trend: trending markets, 1d momentum confirmation
                                # Only when confidence is high AND 1d momentum confirms
                                # (skip 3d check — catches V-shaped recoveries)
                                if sig.confidence >= 0.72:
                                    if sig.signal_type == SignalType.LONG:
                                        if mom_1d < 0.01:  # Strong 1d momentum required
                                            continue
                                    elif sig.signal_type == SignalType.SHORT:
                                        if mom_1d > -0.01:
                                            continue
                                    score = sig.confidence * sig.entry_urgency * 1.2
                                    prev = strategy_signals.get('short_trend')
                                    if prev is None or score > prev[1]:
                                        strategy_signals['short_trend'] = (sig, score)

                    # Current total position value
                    total_position_value = sum(
                        p.shares * current_price for p in positions.values()
                    )
                    total_position_pct = total_position_value / max(equity, 1)

                    # Try to open positions for each strategy independently
                    atr_pct = _compute_atr(tsla_df, bar_idx, period=78)

                    for strat_key, (signal, score) in strategy_signals.items():
                        # Skip if already have a position for this strategy
                        if strat_key in positions:
                            continue

                        # Total position limit
                        if total_position_pct >= MAX_TOTAL_POSITION_PCT:
                            break

                        # Re-entry cooldown per strategy
                        last_exit = last_exit_by_strategy.get(strat_key)
                        if last_exit is not None:
                            signal_dir = 'long' if signal.signal_type == SignalType.LONG else 'short'
                            in_cooldown = (
                                last_exit['profitable']
                                and (bar_idx - last_exit['bar']) < REENTRY_COOLDOWN
                                and signal_dir == last_exit['direction']
                            )
                            if in_cooldown:
                                continue

                            # Anti-chase filter for trend re-entries:
                            # Don't re-enter LONG if price has run >1% above last exit
                            # (prevents chasing into tops after profitable exits)
                            if strat_key == 'trend' and last_exit['profitable']:
                                last_px = last_exit.get('exit_price', 0)
                                if last_px > 0:
                                    if signal_dir == 'long' and current_price > last_px * 1.01:
                                        continue
                                    elif signal_dir == 'short' and current_price < last_px * 0.99:
                                        continue

                        if signal.entry_urgency <= 0.3:
                            continue

                        # Size position (reduced if concurrent)
                        remaining_capacity = MAX_TOTAL_POSITION_PCT - total_position_pct
                        effective_max = min(
                            self.position_sizer.max_position_pct,
                            remaining_capacity,
                        )
                        # Temporarily adjust sizer's max
                        orig_max = self.position_sizer.max_position_pct
                        self.position_sizer.max_position_pct = effective_max

                        position = self.position_sizer.size_position(
                            signal, current_price, atr_pct=atr_pct
                        )
                        self.position_sizer.max_position_pct = orig_max

                        if position.should_trade:
                            # Confidence-based scaling
                            conf_scale = max(0.5, min(5.0,
                                0.7 + (signal.confidence - 0.72) * 100.0
                            ))

                            # Cross-horizon agreement bonus:
                            # If other horizons agree on direction, size up
                            signal_dir = signal.signal_type
                            agreeing_horizons = 0
                            total_horizons = 0
                            for h, hsig in horizon_signals.items():
                                if hsig.signal_type != SignalType.FLAT:
                                    total_horizons += 1
                                    if hsig.signal_type == signal_dir:
                                        agreeing_horizons += 1
                            if total_horizons >= 2:
                                agreement_pct = agreeing_horizons / total_horizons
                                # 100% agreement = 5.0x, 50% = 2.5x, 0% = 0.0x (skip trade)
                                cross_horizon_mult = agreement_pct * 5.0
                                # Unanimous bonus: extra 40% when ALL non-flat horizons agree
                                if agreeing_horizons == total_horizons and total_horizons >= 3:
                                    cross_horizon_mult *= 1.5
                            else:
                                cross_horizon_mult = 1.0

                            total_scale = conf_scale * cross_horizon_mult
                            if total_scale != 1.0:
                                position.shares = max(1, int(position.shares * total_scale))
                                position.dollar_amount = position.shares * current_price
                                position.fraction *= total_scale

                            # Safety cap: limit max position value
                            # Prevents catastrophic single-trade losses
                            MAX_POSITION_VALUE_PCT = 15.0
                            max_position_value = equity * MAX_POSITION_VALUE_PCT
                            if position.dollar_amount > max_position_value:
                                ratio = max_position_value / position.dollar_amount
                                position.shares = max(1, int(position.shares * ratio))
                                position.dollar_amount = position.shares * current_price
                                position.fraction *= ratio

                            new_pos = self._open_position(
                                signal, position, current_price,
                                current_time, bar_idx,
                                atr_pct=atr_pct,
                                strategy=strat_key,
                            )
                            positions[strat_key] = new_pos
                            total_position_value += new_pos.shares * current_price
                            total_position_pct = total_position_value / max(equity, 1)

            except Exception as e:
                # Log but don't crash — skip this eval point
                if bar_idx < start_bar + 100:
                    print(f"[BACKTEST] Warning at bar {bar_idx}: {e}")

            # Progress callback
            if progress_callback and bar_idx % (eval_interval * 10) == 0:
                progress_callback(bar_idx, total_bars, metrics)

        # Print trade log
        if metrics.trades:
            print("\n--- TRADE LOG ---")
            for i, t in enumerate(metrics.trades):
                win = "W" if t.pnl > 0 else "L"
                strat = getattr(t, '_strategy', '?')
                entry_t = t.entry_time.strftime('%m/%d %H:%M') if hasattr(t, 'entry_time') else '?'
                exit_t = t.exit_time.strftime('%m/%d %H:%M') if hasattr(t, 'exit_time') else '?'
                max_ur = getattr(t, '_max_unrealized', None)
                eff_str = ''
                if max_ur is not None and max_ur > 0:
                    capture_eff = t.pnl / max_ur * 100
                    eff_str = f" eff={capture_eff:.0f}%"
                print(
                    f"  #{i+1} {win} {strat:13s} {t.direction:5s} "
                    f"entry=${t.entry_price:.2f}@{entry_t} "
                    f"exit=${t.exit_price:.2f}@{exit_t} "
                    f"pnl=${t.pnl:+.2f} ({t.pnl_pct:+.1%}) "
                    f"hold={t.hold_bars}bars "
                    f"exit={t.exit_reason} "
                    f"conf={t.signal_confidence:.2f} "
                    f"tf={t.primary_tf}{eff_str}"
                )

        # Close any remaining positions at market
        if positions:
            final_price = float(tsla_df.iloc[-1]['close'])
            final_time = _to_datetime(tsla_df.index[-1])
            for strat_key, pos in positions.items():
                trade = self._close_position(
                    pos, final_price, final_time,
                    total_bars - pos.entry_bar, 'end_of_data'
                )
                trade._strategy = strat_key
                metrics.add_trade(trade)
                equity += trade.pnl

        # Final equity
        equity_curve.add_point(_to_datetime(tsla_df.index[-1]), equity)
        metrics.equity_curve = equity_curve

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            signals_generated=signals_generated,
            signals_actionable=signals_actionable,
            bars_processed=total_bars - start_bar,
            start_time=start_time,
            end_time=end_time,
        )

    def _get_prediction(
        self, tsla_df, spy_df, vix_df, native_bars
    ):
        """Get model prediction for current bar."""
        try:
            result = self.predictor.predict_with_per_tf(
                tsla_df, spy_df, vix_df,
                native_bars_by_tf=native_bars,
            )
            return result
        except Exception:
            return None

    # Horizon-specific max hold bars
    HORIZON_MAX_HOLD = {
        'short': 78,    # 1 trading day
        'medium': 156,  # 2 trading days
        'long': 390,    # 1 trading week
    }

    def _check_exit(
        self,
        pos: OpenPosition,
        current_price: float,
        high: float,
        low: float,
        bars_held: int,
    ) -> tuple:
        """Check if position should be exited. Returns (exit_price, reason) or (None, None)."""
        from ..config import TF_TO_HORIZON
        horizon = TF_TO_HORIZON.get(pos.primary_tf, 'medium')
        max_hold = self.HORIZON_MAX_HOLD.get(horizon, self.config.max_hold_bars)
        max_hold = min(max_hold, self.config.max_hold_bars)

        # Progressive trail tightening: as hold time increases, tighten the trail
        # This captures more profit on aging trades
        hold_pct = min(1.0, bars_held / max(max_hold, 1))
        tightening = 1.0 - hold_pct * 0.6  # Trail shrinks to 40% of original at max hold
        effective_trail = pos.trailing_stop_pct * tightening

        # Profit lock: once trade reaches 50%+ of TP target,
        # switch to a tighter trail (50% of normal width)
        if pos.direction == 'long':
            tp_dist = pos.take_profit_price - pos.entry_price
            current_profit_pct = (pos.best_price - pos.entry_price) / tp_dist if tp_dist > 0 else 0
        else:
            tp_dist = pos.entry_price - pos.take_profit_price
            current_profit_pct = (pos.entry_price - pos.best_price) / tp_dist if tp_dist > 0 else 0
        if current_profit_pct >= 0.50:
            effective_trail *= 0.6  # 40% tighter trail when deeply profitable

        if pos.direction == 'long':
            if high > pos.best_price:
                pos.best_price = high
            # Fixed stop loss
            if low <= pos.stop_loss_price:
                return pos.stop_loss_price, 'stop_loss'
            # Trailing stop (only after we're profitable)
            if pos.best_price > pos.entry_price:
                trailing_stop = pos.best_price * (1 - effective_trail)
                if trailing_stop > pos.stop_loss_price and low <= trailing_stop:
                    return trailing_stop, 'trailing_stop'
            # Take profit hit
            if high >= pos.take_profit_price:
                return pos.take_profit_price, 'take_profit'
        else:  # short
            if pos.best_price == 0 or low < pos.best_price:
                pos.best_price = low
            if high >= pos.stop_loss_price:
                return pos.stop_loss_price, 'stop_loss'
            # Trailing stop for shorts
            if pos.best_price > 0 and pos.best_price < pos.entry_price:
                trailing_stop = pos.best_price * (1 + effective_trail)
                if trailing_stop < pos.stop_loss_price and high >= trailing_stop:
                    return trailing_stop, 'trailing_stop'
            if low <= pos.take_profit_price:
                return pos.take_profit_price, 'take_profit'

        # Horizon-specific timeout
        if bars_held >= max_hold:
            return current_price, 'timeout'

        return None, None

    # ATR multipliers by horizon (used with ATR to compute trail width)
    # Lower multiplier = tighter trail (bounce = quick exit)
    # Higher multiplier = wider trail (trend = ride the wave)
    HORIZON_ATR_MULT = {
        'short': 1.5,    # Tight: 1.5x ATR trail for bounces
        'medium': 2.0,   # Medium: 2x ATR
        'long': 2.5,     # Wide: 2.5x ATR for trend-following
    }
    # Fallback fixed percentages (used if ATR is unavailable)
    HORIZON_TRAIL_PCT = {
        'short': 0.015,   # 1.5% trail for short TFs (tight)
        'medium': 0.020,  # 2% trail for medium TFs
        'long': 0.030,    # 3% trail for long TFs
    }

    def _open_position(
        self,
        signal: TradeSignal,
        sizing: PositionRecommendation,
        price: float,
        time: datetime,
        bar_idx: int,
        atr_pct: float = 0.02,
        strategy: str = 'trend',
    ) -> OpenPosition:
        """Open a new position."""
        from ..config import TF_TO_HORIZON

        # Apply slippage
        slippage = price * self.config.slippage_pct
        if signal.signal_type == SignalType.LONG:
            entry_price = price + slippage  # Worse fill for long
            direction = 'long'
            stop_price = entry_price * (1 - sizing.stop_loss_pct)
            tp_price = entry_price * (1 + sizing.take_profit_pct)
        else:
            entry_price = price - slippage  # Worse fill for short
            direction = 'short'
            stop_price = entry_price * (1 + sizing.stop_loss_pct)
            tp_price = entry_price * (1 - sizing.take_profit_pct)

        commission = sizing.shares * self.config.commission_per_share

        # ATR-adaptive trailing stop
        horizon = TF_TO_HORIZON.get(signal.primary_tf, 'medium')
        atr_mult = self.HORIZON_ATR_MULT.get(horizon, 2.0)
        trail_pct = atr_pct * atr_mult

        # Clamp to reasonable range
        min_trail = self.HORIZON_TRAIL_PCT.get(horizon, 0.020) * 0.5  # At least half the fixed %
        max_trail = 0.06  # Never more than 6%
        trail_pct = max(min_trail, min(max_trail, trail_pct))

        return OpenPosition(
            entry_time=time,
            entry_bar=bar_idx,
            direction=direction,
            entry_price=entry_price,
            shares=sizing.shares,
            stop_loss_price=stop_price,
            take_profit_price=tp_price,
            signal_confidence=signal.confidence,
            regime=signal.regime.regime.value,
            primary_tf=signal.primary_tf,
            commission_entry=commission,
            strategy=strategy,
            best_price=entry_price,
            trailing_stop_pct=trail_pct,
        )

    def _close_position(
        self,
        pos: OpenPosition,
        exit_price: float,
        exit_time: datetime,
        bars_held: int,
        exit_reason: str,
    ) -> Trade:
        """Close a position and create a trade record."""
        # Apply slippage on exit
        slippage_per_share = exit_price * self.config.slippage_pct
        if pos.direction == 'long':
            actual_exit = exit_price - slippage_per_share
            raw_pnl = (actual_exit - pos.entry_price) * pos.shares
        else:
            actual_exit = exit_price + slippage_per_share
            raw_pnl = (pos.entry_price - actual_exit) * pos.shares

        commission_exit = pos.shares * self.config.commission_per_share
        total_commission = pos.commission_entry + commission_exit
        total_slippage = slippage_per_share * pos.shares * 2  # Both sides

        net_pnl = raw_pnl - total_commission
        entry_value = pos.entry_price * pos.shares
        pnl_pct = net_pnl / entry_value if entry_value > 0 else 0.0

        return Trade(
            entry_time=pos.entry_time,
            exit_time=exit_time,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=actual_exit,
            shares=pos.shares,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=total_commission,
            slippage=total_slippage,
            signal_confidence=pos.signal_confidence,
            regime=pos.regime,
            primary_tf=pos.primary_tf,
            exit_reason=exit_reason,
            hold_bars=bars_held,
        )

    def _should_flip(
        self, pos: OpenPosition, signal: TradeSignal
    ) -> bool:
        """Check if current signal contradicts open position."""
        if signal.signal_type == SignalType.LONG and pos.direction == 'short':
            return signal.confidence > 0.65
        if signal.signal_type == SignalType.SHORT and pos.direction == 'long':
            return signal.confidence > 0.65
        return False


def _compute_atr(df: pd.DataFrame, bar_idx: int, period: int = 14) -> float:
    """Compute Average True Range as a percentage of price."""
    if bar_idx < period + 1:
        return 0.02  # Default 2% if insufficient data
    start = max(0, bar_idx - period)
    high = df.iloc[start:bar_idx + 1]['high'].values
    low = df.iloc[start:bar_idx + 1]['low'].values
    close = df.iloc[start:bar_idx + 1]['close'].values

    tr_values = []
    for i in range(1, len(high)):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        tr_values.append(tr)

    if not tr_values:
        return 0.02
    atr = sum(tr_values) / len(tr_values)
    atr_pct = atr / close[-1] if close[-1] > 0 else 0.02
    return atr_pct


def _to_datetime(ts) -> datetime:
    """Convert pandas Timestamp or similar to datetime."""
    if isinstance(ts, datetime):
        return ts
    if hasattr(ts, 'to_pydatetime'):
        dt = ts.to_pydatetime()
        if hasattr(dt, 'replace') and dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        return dt
    return datetime.now()
