"""
Position Sizing with Adaptive Kelly + Drawdown Thermostat

From Codex research: f_final = f * exp(-kappa * DD)

Uses model outputs to estimate Kelly fraction:
- Edge from signal confidence + regime
- Variance from duration_std
- Drawdown thermostat auto-reduces size during drawdowns
"""
import math
from dataclasses import dataclass
from typing import Optional

from .signals import TradeSignal, SignalType, MarketRegime


@dataclass
class PositionRecommendation:
    """Position sizing recommendation."""
    # Core sizing
    fraction: float  # Fraction of capital to risk (0-1)
    shares: int  # Number of shares (given capital and price)
    dollar_amount: float  # Dollar value of position
    # Risk parameters
    stop_loss_pct: float  # Suggested stop loss (percent from entry)
    take_profit_pct: float  # Suggested take profit (percent from entry)
    risk_reward_ratio: float  # TP / SL
    # Scaling info
    kelly_raw: float  # Raw Kelly fraction before adjustments
    drawdown_factor: float  # Drawdown thermostat multiplier
    regime_factor: float  # Regime-based adjustment
    confidence_factor: float  # Signal confidence adjustment
    # Status
    should_trade: bool  # Whether to take this trade
    reason: str  # Why/why not


class PositionSizer:
    """
    Adaptive position sizer using Kelly criterion + drawdown thermostat.

    Kelly formula: f* = (p * b - q) / b
    where p = win probability, b = win/loss ratio, q = 1-p

    Drawdown thermostat: f_final = f * exp(-kappa * DD)
    where DD = current drawdown fraction, kappa = sensitivity

    Additional adjustments:
    - Half-Kelly for safety (Kelly is optimal but volatile)
    - Regime-based scaling (trend = larger, range = smaller)
    - Coherence bonus (more TF agreement = more confidence)
    """

    def __init__(
        self,
        capital: float = 100000.0,
        max_position_pct: float = 0.40,  # Max 40% of capital per trade
        min_position_pct: float = 0.02,  # Min 2% per trade
        kelly_fraction: float = 0.5,  # Half-Kelly for safety
        drawdown_kappa: float = 3.0,  # Drawdown sensitivity
        max_drawdown_halt: float = 0.15,  # Stop trading at 15% drawdown
        base_stop_loss_pct: float = 0.02,  # 2% default stop
        base_take_profit_pct: float = 0.04,  # 4% default TP (2:1 R:R)
    ):
        self.capital = capital
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.kelly_fraction = kelly_fraction
        self.drawdown_kappa = drawdown_kappa
        self.max_drawdown_halt = max_drawdown_halt
        self.base_stop_loss_pct = base_stop_loss_pct
        self.base_take_profit_pct = base_take_profit_pct

        # Track equity for drawdown
        self.peak_equity = capital
        self.current_equity = capital

        # Win streak tracking (anti-martingale position scaling)
        self.consecutive_wins = 0
        self.streak_base_pct = max_position_pct  # Base position size
        self.streak_increment = 0.05  # +5% per consecutive win
        self.streak_max_pct = 0.60  # Never exceed 60%

    def update_equity(self, new_equity: float):
        """Update current equity (call after each trade closes)."""
        self.current_equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)

    def record_trade_result(self, won: bool):
        """Update win streak tracking for anti-martingale sizing."""
        if won:
            self.consecutive_wins += 1
        else:
            self.consecutive_wins = 0
        # Adjust max position based on streak
        streak_bonus = self.consecutive_wins * self.streak_increment
        self.max_position_pct = min(
            self.streak_max_pct,
            self.streak_base_pct + streak_bonus,
        )

    @property
    def current_drawdown(self) -> float:
        """Current drawdown as fraction (0 = at peak, 0.1 = 10% below peak)."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.current_equity / self.peak_equity)

    def size_position(
        self,
        signal: TradeSignal,
        current_price: float,
        win_rate: Optional[float] = None,
        avg_win_loss_ratio: Optional[float] = None,
        atr_pct: Optional[float] = None,
    ) -> PositionRecommendation:
        """
        Compute position size for a trade signal.

        Args:
            signal: The trade signal from the signal engine
            current_price: Current asset price
            win_rate: Historical win rate (if None, estimated from confidence)
            avg_win_loss_ratio: Average win/loss size ratio (if None, use 1.5)
        """
        # Check if we should trade at all
        if signal.signal_type == SignalType.FLAT:
            return self._no_trade(current_price, "Signal is FLAT")

        if self.current_drawdown >= self.max_drawdown_halt:
            return self._no_trade(
                current_price,
                f"Drawdown halt: {self.current_drawdown:.1%} >= {self.max_drawdown_halt:.1%}"
            )

        if not signal.actionable:
            return self._no_trade(current_price, "Signal not actionable")

        # Estimate win probability from signal confidence
        if win_rate is not None:
            p = win_rate
        else:
            # Map confidence to win probability
            # Confidence 0.55 => ~52% win rate (small edge)
            # Confidence 0.75 => ~60% win rate (solid edge)
            # Confidence 0.90 => ~68% win rate (strong edge)
            p = 0.50 + signal.confidence * 0.25

        # Win/loss ratio
        b = avg_win_loss_ratio if avg_win_loss_ratio is not None else 1.5

        q = 1.0 - p

        # Kelly criterion: f* = (p*b - q) / b
        if b <= 0:
            return self._no_trade(current_price, "Invalid win/loss ratio")

        kelly_raw = (p * b - q) / b

        if kelly_raw <= 0:
            return self._no_trade(
                current_price,
                f"Negative Kelly ({kelly_raw:.3f}): edge insufficient"
            )

        # Apply half-Kelly for safety
        kelly_adjusted = kelly_raw * self.kelly_fraction

        # Drawdown thermostat: f = f * exp(-kappa * DD)
        dd = self.current_drawdown
        dd_factor = math.exp(-self.drawdown_kappa * dd)

        # Regime factor
        regime_factor = self._regime_factor(signal)

        # Confidence factor (boost for high confidence, reduce for low)
        conf_factor = 0.5 + signal.confidence  # Range: 0.5 - 1.5

        # Win-streak multiplier (anti-martingale: compound after wins)
        streak_mult = 1.0 + self.consecutive_wins * 0.15  # +15% per consecutive win
        streak_mult = min(streak_mult, 2.0)  # Cap at 2x

        # Final position fraction
        fraction = kelly_adjusted * dd_factor * regime_factor * conf_factor * streak_mult

        # Clamp to limits
        fraction = max(self.min_position_pct, min(self.max_position_pct, fraction))

        # Compute shares and dollars
        dollar_amount = self.current_equity * fraction
        shares = int(dollar_amount / current_price)

        if shares <= 0:
            return self._no_trade(current_price, "Position too small for 1 share")

        dollar_amount = shares * current_price

        # Stop loss and take profit (ATR-based if available)
        stop_loss, take_profit = self._compute_stops(signal, current_price, atr_pct=atr_pct)

        return PositionRecommendation(
            fraction=fraction,
            shares=shares,
            dollar_amount=dollar_amount,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            risk_reward_ratio=take_profit / max(stop_loss, 0.001),
            kelly_raw=kelly_raw,
            drawdown_factor=dd_factor,
            regime_factor=regime_factor,
            confidence_factor=conf_factor,
            should_trade=True,
            reason=f"Kelly={kelly_raw:.3f}, DD={dd:.1%}, Regime={signal.regime.regime.value}",
        )

    def _regime_factor(self, signal: TradeSignal) -> float:
        """Scale position size based on market regime."""
        regime = signal.regime.regime

        if regime == MarketRegime.TRENDING_BULL:
            # Trends are more predictable, can size larger
            return 1.0 + signal.regime.confidence * 0.3
        elif regime == MarketRegime.TRENDING_BEAR:
            # Bear trends: still good, slightly more cautious
            return 0.9 + signal.regime.confidence * 0.2
        elif regime == MarketRegime.RANGING:
            # Ranging: smaller positions, more frequent trades
            return 0.7 + signal.regime.confidence * 0.2
        else:
            # Transitioning: be cautious
            return 0.5

    def _compute_stops(
        self, signal: TradeSignal, price: float,
        atr_pct: Optional[float] = None,
    ) -> tuple:
        """Compute stop loss and take profit percentages."""
        # Fixed percentage stops (proven to work for trend-following)
        if signal.regime.regime in (MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR):
            stop = self.base_stop_loss_pct * 1.5  # 3% in trends
            tp = self.base_take_profit_pct * 2.0  # 8% in trends
        elif signal.regime.regime == MarketRegime.RANGING:
            stop = self.base_stop_loss_pct * 0.8
            tp = self.base_take_profit_pct * 0.7
        else:
            stop = self.base_stop_loss_pct
            tp = self.base_take_profit_pct

        # Adjust by confidence (higher confidence = can afford tighter stop)
        confidence_adj = 1.0 - (signal.confidence - 0.5) * 0.3
        stop *= confidence_adj

        # Ensure minimum R:R of 1.5:1
        if tp / max(stop, 0.001) < 1.5:
            tp = stop * 1.5

        return stop, tp

    def _no_trade(self, price: float, reason: str) -> PositionRecommendation:
        """Return a no-trade recommendation."""
        return PositionRecommendation(
            fraction=0.0,
            shares=0,
            dollar_amount=0.0,
            stop_loss_pct=0.0,
            take_profit_pct=0.0,
            risk_reward_ratio=0.0,
            kelly_raw=0.0,
            drawdown_factor=1.0,
            regime_factor=1.0,
            confidence_factor=1.0,
            should_trade=False,
            reason=reason,
        )
