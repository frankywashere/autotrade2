__all__ = ['RSIMonitor', 'DataFetcher', 'SignalGenerator', 'VIXAnalyzer', 'BounceRuleEngine']


def __getattr__(name):
    if name == 'RSIMonitor':
        from .core import RSIMonitor
        return RSIMonitor
    elif name == 'DataFetcher':
        from .data import DataFetcher
        return DataFetcher
    elif name == 'SignalGenerator':
        from .signals import SignalGenerator
        return SignalGenerator
    elif name == 'VIXAnalyzer':
        from .vix_analyzer import VIXAnalyzer
        return VIXAnalyzer
    elif name == 'BounceRuleEngine':
        from .bounce_rules import BounceRuleEngine
        return BounceRuleEngine
    raise AttributeError(f"module 'rsi_monitor' has no attribute {name!r}")
