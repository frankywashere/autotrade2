"""
Evolutionary Meta-Strategy with Multiplicative Weights.

From Codex research: Each strategy is treated as an "agent" in a multi-armed
bandit setting. Weights are updated using multiplicative weights (EXP3-style):

    w_{k,t+1} ∝ w_{k,t} * exp(η * R_{k,t})

where R_{k,t} is the net P&L (after slippage) for strategy k at time t.

This makes the system adapt to which strategies work in current market conditions
without needing hard-coded regime switches.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..inference import PerTFPrediction

from .signals import (
    SignalType, RegimeState, HazardClock,
    RegimeAdaptiveSignalEngine, TradeSignal,
)
from .strategies import BaseStrategy, StrategySignal, ALL_STRATEGIES


@dataclass
class StrategyTrack:
    """Tracks a strategy's performance over time."""
    name: str
    weight: float = 1.0
    cumulative_pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    recent_pnl: List[float] = field(default_factory=list)
    max_recent: int = 20  # Track last 20 trades

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.trades, 1)

    @property
    def avg_recent_pnl(self) -> float:
        if not self.recent_pnl:
            return 0.0
        return sum(self.recent_pnl) / len(self.recent_pnl)

    def record_trade(self, pnl: float):
        self.cumulative_pnl += pnl
        self.trades += 1
        if pnl > 0:
            self.wins += 1
        self.recent_pnl.append(pnl)
        if len(self.recent_pnl) > self.max_recent:
            self.recent_pnl.pop(0)


class MetaStrategy:
    """
    Evolutionary meta-strategy using multiplicative weights.

    Evaluates all sub-strategies on each prediction, selects the one
    with highest weighted score, and adapts weights based on outcomes.
    """

    def __init__(
        self,
        strategies: Optional[List[BaseStrategy]] = None,
        learning_rate: float = 0.1,
        min_weight: float = 0.05,
        agreement_bonus: float = 0.2,
    ):
        """
        Args:
            strategies: List of strategy instances (default: ALL_STRATEGIES)
            learning_rate: EXP3 learning rate (η)
            min_weight: Minimum weight for any strategy (prevents extinction)
            agreement_bonus: Bonus when multiple strategies agree
        """
        self.strategies = strategies or list(ALL_STRATEGIES)
        self.learning_rate = learning_rate
        self.min_weight = min_weight
        self.agreement_bonus = agreement_bonus

        # Initialize tracking
        self.tracks: Dict[str, StrategyTrack] = {}
        for s in self.strategies:
            self.tracks[s.name] = StrategyTrack(name=s.name)

    def evaluate(
        self,
        per_tf_predictions: Dict[str, 'PerTFPrediction'],
        regime: RegimeState,
        hazard: HazardClock,
    ) -> tuple:
        """
        Evaluate all strategies and return best weighted signal.

        Returns:
            (best_signal: StrategySignal, all_signals: Dict[str, StrategySignal])
        """
        signals: Dict[str, StrategySignal] = {}

        for strategy in self.strategies:
            try:
                signal = strategy.evaluate(per_tf_predictions, regime, hazard)
                signals[strategy.name] = signal
            except Exception as e:
                # Strategy failed — skip it
                signals[strategy.name] = StrategySignal(
                    name=strategy.name,
                    signal_type=SignalType.FLAT,
                    confidence=0.0,
                    primary_tf='1h',
                    edge_estimate=0.0,
                    stop_loss_pct=0.02,
                    take_profit_pct=0.04,
                    reasoning=f"Error: {e}",
                )

        # Score each signal: confidence × weight × agreement_bonus
        scored = {}
        for name, signal in signals.items():
            track = self.tracks[name]
            base_score = signal.confidence * track.weight

            # Agreement bonus: boost if multiple strategies agree on direction
            if signal.signal_type != SignalType.FLAT:
                agreeing = sum(
                    1 for s in signals.values()
                    if s.signal_type == signal.signal_type and s.confidence > 0.4
                )
                if agreeing >= 2:
                    base_score *= (1 + self.agreement_bonus * (agreeing - 1))

            scored[name] = base_score

        # Select best
        best_name = max(scored, key=scored.get)
        best_signal = signals[best_name]

        return best_signal, signals

    def update_weights(self, strategy_name: str, pnl: float):
        """
        Update strategy weight after a trade completes.

        Uses multiplicative weights update:
            w_{k,t+1} = w_{k,t} * exp(η * normalized_pnl)
        """
        if strategy_name not in self.tracks:
            return

        track = self.tracks[strategy_name]
        track.record_trade(pnl)

        # Normalize PnL to [-1, 1] range for stable learning
        # Assume typical trade PnL is in range [-3%, +3%]
        normalized_pnl = max(-1.0, min(1.0, pnl / 1000.0))

        # Multiplicative weight update
        track.weight *= math.exp(self.learning_rate * normalized_pnl)

        # Ensure minimum weight (prevent extinction)
        track.weight = max(self.min_weight, track.weight)

        # Normalize weights so they sum to num_strategies
        total = sum(t.weight for t in self.tracks.values())
        n = len(self.tracks)
        if total > 0:
            for t in self.tracks.values():
                t.weight = t.weight / total * n

    def summary(self) -> str:
        """Print strategy performance summary."""
        lines = [
            "=" * 60,
            "META-STRATEGY PERFORMANCE",
            "=" * 60,
        ]
        for name in sorted(self.tracks.keys()):
            t = self.tracks[name]
            lines.append(
                f"  {name:25s}: w={t.weight:.3f}, "
                f"trades={t.trades}, win={t.win_rate:.0%}, "
                f"P&L=${t.cumulative_pnl:,.2f}, "
                f"recent_avg=${t.avg_recent_pnl:,.2f}"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


class MetaBacktester:
    """
    Backtester that uses the meta-strategy for adaptive strategy selection.

    This wraps the standard backtester but replaces the signal engine
    with the meta-strategy ensemble.
    """

    def __init__(
        self,
        predictor,
        meta: Optional[MetaStrategy] = None,
        initial_capital: float = 100000.0,
        eval_interval: int = 12,
        max_hold_bars: int = 390,
    ):
        from .position_sizer import PositionSizer
        from .backtester import BacktestConfig

        self.predictor = predictor
        self.meta = meta or MetaStrategy()
        self.config = BacktestConfig(
            initial_capital=initial_capital,
            eval_interval_bars=eval_interval,
            max_hold_bars=max_hold_bars,
        )
        self.sizer = PositionSizer(capital=initial_capital)
        self.signal_engine = RegimeAdaptiveSignalEngine()

    def run(self, tsla_df, spy_df, vix_df, native_bars=None, progress_cb=None):
        """
        Run meta-strategy backtest.

        Similar to standard backtester but:
        1. Evaluates ALL strategies at each step
        2. Uses meta-strategy weighted selection
        3. Updates weights after each trade
        4. Reports per-strategy performance
        """
        import pandas as pd
        from .metrics import TradeMetrics, EquityCurve, Trade
        from .backtester import OpenPosition, BacktestResult, _to_datetime

        metrics = TradeMetrics()
        equity_curve = EquityCurve()
        equity = self.config.initial_capital
        self.sizer.current_equity = equity
        self.sizer.peak_equity = equity

        open_position = None
        active_strategy = None
        prev_hazard = None

        total_bars = len(tsla_df)
        start_bar = 1000
        eval_interval = self.config.eval_interval_bars
        signals_generated = 0
        signals_actionable = 0

        start_time = _to_datetime(tsla_df.index[start_bar])
        equity_curve.add_point(start_time, equity)

        for bar_idx in range(start_bar, total_bars, eval_interval):
            current_time = _to_datetime(tsla_df.index[bar_idx])
            current_price = float(tsla_df.iloc[bar_idx]['close'])

            # Check exit for open position
            if open_position is not None:
                high = float(tsla_df.iloc[bar_idx]['high'])
                low = float(tsla_df.iloc[bar_idx]['low'])
                bars_held = bar_idx - open_position.entry_bar

                exit_price, exit_reason = self._check_exit(
                    open_position, current_price, high, low, bars_held
                )
                if exit_price is not None:
                    trade = self._close_position(
                        open_position, exit_price, current_time,
                        bars_held, exit_reason
                    )
                    metrics.add_trade(trade)
                    equity += trade.pnl
                    self.sizer.update_equity(equity)
                    equity_curve.add_point(current_time, equity)

                    # Update meta-strategy weights
                    if active_strategy:
                        self.meta.update_weights(active_strategy, trade.pnl)
                    open_position = None
                    active_strategy = None

            # Generate predictions
            try:
                prediction = self.predictor.predict_with_per_tf(
                    tsla_df.iloc[:bar_idx + 1],
                    spy_df.iloc[:bar_idx + 1],
                    vix_df.iloc[:bar_idx + 1],
                    native_bars_by_tf=native_bars,
                )

                if prediction and prediction.per_tf_predictions:
                    # Compute regime and hazard
                    regime = self.signal_engine._detect_regime(
                        prediction.per_tf_predictions
                    )
                    hazard = self.signal_engine._compute_hazard(
                        prediction.per_tf_predictions, None, prev_hazard
                    )
                    prev_hazard = hazard

                    # Evaluate all strategies via meta
                    best_signal, all_signals = self.meta.evaluate(
                        prediction.per_tf_predictions, regime, hazard
                    )
                    signals_generated += 1

                    if best_signal.confidence >= 0.55 and best_signal.signal_type != SignalType.FLAT:
                        signals_actionable += 1

                    # Open position
                    if (open_position is None
                            and best_signal.signal_type != SignalType.FLAT
                            and best_signal.confidence >= 0.55):

                        # Build a TradeSignal-like object for position sizer
                        from .signals import TradeSignal
                        trade_signal = TradeSignal(
                            signal_type=best_signal.signal_type,
                            regime=regime,
                            hazard=hazard,
                            confidence=best_signal.confidence,
                            edge_estimate=best_signal.edge_estimate,
                            primary_tf=best_signal.primary_tf,
                            entry_urgency=0.8,
                            bars_to_optimal_entry=0,
                            direction_agreement=regime.tf_agreement,
                            next_channel_alignment=0.5,
                            per_tf_scores={},
                            per_tf_directions={},
                        )

                        sizing = self.sizer.size_position(
                            trade_signal, current_price
                        )
                        if sizing.should_trade:
                            slippage = current_price * self.config.slippage_pct
                            if best_signal.signal_type == SignalType.LONG:
                                entry_price = current_price + slippage
                                stop = entry_price * (1 - best_signal.stop_loss_pct)
                                tp = entry_price * (1 + best_signal.take_profit_pct)
                            else:
                                entry_price = current_price - slippage
                                stop = entry_price * (1 + best_signal.stop_loss_pct)
                                tp = entry_price * (1 - best_signal.take_profit_pct)

                            open_position = OpenPosition(
                                entry_time=current_time,
                                entry_bar=bar_idx,
                                direction='long' if best_signal.signal_type == SignalType.LONG else 'short',
                                entry_price=entry_price,
                                shares=sizing.shares,
                                stop_loss_price=stop,
                                take_profit_price=tp,
                                signal_confidence=best_signal.confidence,
                                regime=regime.regime.value,
                                primary_tf=best_signal.primary_tf,
                                commission_entry=sizing.shares * self.config.commission_per_share,
                            )
                            active_strategy = best_signal.name

            except Exception as e:
                if bar_idx < start_bar + 100:
                    print(f"[META-BT] Warning at bar {bar_idx}: {e}")

            if progress_cb and bar_idx % (eval_interval * 10) == 0:
                progress_cb(bar_idx, total_bars, metrics)

        # Close remaining
        if open_position is not None:
            final_price = float(tsla_df.iloc[-1]['close'])
            final_time = _to_datetime(tsla_df.index[-1])
            trade = self._close_position(
                open_position, final_price, final_time,
                total_bars - open_position.entry_bar, 'end_of_data'
            )
            metrics.add_trade(trade)
            equity += trade.pnl
            if active_strategy:
                self.meta.update_weights(active_strategy, trade.pnl)

        equity_curve.add_point(_to_datetime(tsla_df.index[-1]), equity)
        metrics.equity_curve = equity_curve

        result = BacktestResult(
            config=self.config,
            metrics=metrics,
            signals_generated=signals_generated,
            signals_actionable=signals_actionable,
            bars_processed=total_bars - start_bar,
            start_time=start_time,
            end_time=_to_datetime(tsla_df.index[-1]),
        )

        return result

    def _check_exit(self, pos, current_price, high, low, bars_held):
        if pos.direction == 'long':
            if low <= pos.stop_loss_price:
                return pos.stop_loss_price, 'stop_loss'
            if high >= pos.take_profit_price:
                return pos.take_profit_price, 'take_profit'
        else:
            if high >= pos.stop_loss_price:
                return pos.stop_loss_price, 'stop_loss'
            if low <= pos.take_profit_price:
                return pos.take_profit_price, 'take_profit'
        if bars_held >= self.config.max_hold_bars:
            return current_price, 'timeout'
        return None, None

    def _close_position(self, pos, exit_price, exit_time, bars_held, reason):
        from .metrics import Trade
        slippage = exit_price * self.config.slippage_pct
        if pos.direction == 'long':
            actual_exit = exit_price - slippage
            raw_pnl = (actual_exit - pos.entry_price) * pos.shares
        else:
            actual_exit = exit_price + slippage
            raw_pnl = (pos.entry_price - actual_exit) * pos.shares

        commission_exit = pos.shares * self.config.commission_per_share
        total_commission = pos.commission_entry + commission_exit
        total_slippage = slippage * pos.shares * 2
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
            exit_reason=reason,
            hold_bars=bars_held,
        )
