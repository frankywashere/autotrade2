"""
ScannerManager — Registry of algo adapters with evaluate/exit/trail dispatch.

Handles both IB (two-phase commit) and yfinance (instant fill) paths.
"""

import logging
from typing import Optional

from .base import AlgoAdapter, Signal, ExitSignal
from ..db.trade_db import TradeDB

logger = logging.getLogger(__name__)


class ScannerManager:
    """Registry of algo adapters. Start/stop/kill per-algo."""

    def __init__(self, trade_db: TradeDB, source: str, ib_client=None):
        """
        Args:
            trade_db: Shared trade database
            source: 'ib' or 'yf'
            ib_client: IBClient instance (required for source='ib')
        """
        self._adapters: dict[str, AlgoAdapter] = {}
        self._db = trade_db
        self._source = source
        self._ib = ib_client
        self._kill_all = False

    def register(self, adapter: AlgoAdapter):
        """Register an algo adapter."""
        self._adapters[adapter.algo_id] = adapter
        logger.info(f"Registered adapter: {adapter.algo_id} "
                     f"(source={self._source})")

    def get_adapter(self, algo_id: str) -> Optional[AlgoAdapter]:
        """Get adapter by algo_id."""
        return self._adapters.get(algo_id)

    @property
    def adapters(self) -> dict[str, AlgoAdapter]:
        return self._adapters

    def set_enabled(self, algo_id: str, enabled: bool):
        """Toggle specific algo on/off."""
        if algo_id in self._adapters:
            self._adapters[algo_id].enabled = enabled
            logger.info(f"Adapter {algo_id} {'enabled' if enabled else 'disabled'}")

    def kill_all(self):
        """Emergency stop — disable all algos."""
        self._kill_all = True
        for adapter in self._adapters.values():
            adapter.enabled = False
        logger.warning("KILL ALL — all adapters disabled")

    def unkill(self):
        """Re-enable after kill."""
        self._kill_all = False
        for adapter in self._adapters.values():
            adapter.enabled = True
        logger.info("Kill switch released — all adapters re-enabled")

    def evaluate_all(self, price: float, analysis: dict,
                     features: dict = None,
                     ib_degraded: bool = False) -> list[Signal]:
        """Run all enabled adapters, return entry signals.

        For source='ib': uses include_pending=True for gating.
        """
        if self._kill_all:
            return []

        signals = []
        source = self._source

        # Get open trades for gating
        if source == 'ib':
            open_trades = self._db.get_open_trades(
                source='ib', include_pending=True)
        else:
            open_trades = self._db.get_open_trades(source='yf')

        for algo_id, adapter in self._adapters.items():
            if not adapter.enabled:
                continue
            if source == 'ib' and ib_degraded:
                continue

            # Filter to this algo's trades
            algo_open = [t for t in open_trades if t['algo_id'] == algo_id]

            try:
                sig = adapter.evaluate(price, analysis, algo_open, features)
                if sig:
                    signals.append(sig)
            except Exception:
                logger.exception(f"Error evaluating {algo_id}")

        return signals

    def check_all_exits(self, price: float,
                        bid: float = 0, ask: float = 0,
                        ib_degraded: bool = False) -> list[ExitSignal]:
        """Check exits for all open positions across all algos."""
        source = self._source
        open_trades = self._db.get_open_trades(source=source)
        exits = []

        for trade in open_trades:
            algo_id = trade['algo_id']

            # Manual trades: user manages via UI
            if algo_id == 'manual':
                continue

            adapter = self._adapters.get(algo_id)
            if adapter is None:
                logger.warning(f"No adapter for algo_id={algo_id} — "
                               f"position {trade['id']} has no exit coverage")
                if source == 'ib':
                    # Flag for manual intervention
                    logger.error(f"IB position {trade['id']} ({algo_id}) "
                                 f"has no adapter — set ib_degraded")
                continue

            try:
                exit_sig = adapter.check_exit(trade, price, bid, ask)
                if exit_sig:
                    exits.append(exit_sig)
            except Exception:
                logger.exception(f"Error checking exit for trade "
                                 f"{trade['id']} ({algo_id})")

        return exits

    def update_all_trailing(self, price: float) -> list[tuple[int, dict]]:
        """Update trailing stops for all open positions.

        Returns list of (trade_id, changes_dict) for positions that changed.
        Caller MUST persist changes via trade_db.update_trade_state() and
        modify the IB stop if stop_price changed.
        """
        source = self._source
        open_trades = self._db.get_open_trades(source=source)
        updates = []

        for trade in open_trades:
            algo_id = trade['algo_id']

            if algo_id == 'manual':
                continue

            adapter = self._adapters.get(algo_id)
            if adapter is None:
                continue

            try:
                changes = adapter.update_trailing(trade, price)
                if changes:
                    updates.append((trade['id'], changes))
            except Exception:
                logger.exception(f"Error updating trailing for trade "
                                 f"{trade['id']} ({algo_id})")

        return updates
