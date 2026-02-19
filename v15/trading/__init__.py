"""
Regime-Adaptive Trading Engine (c3)

A profit-maximizing trading system that uses multi-timeframe channel predictions
with hazard-based timing, regime detection, and adaptive position sizing.
"""
from .signals import TradeSignal, SignalType, RegimeAdaptiveSignalEngine
from .position_sizer import PositionSizer, PositionRecommendation
from .backtester import Backtester, BacktestResult
from .metrics import TradeMetrics, EquityCurve

__all__ = [
    'TradeSignal', 'SignalType', 'RegimeAdaptiveSignalEngine',
    'PositionSizer', 'PositionRecommendation',
    'Backtester', 'BacktestResult',
    'TradeMetrics', 'EquityCurve',
]
