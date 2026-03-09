"""
LiveEngine — Coordination layer for running unified backtester algos with live IB data.

Dispatches bar-close events to algos, manages delayed entries, syncs broker-side
stops, and handles IB fill callbacks. Follows the same causal loop order as
BacktestEngine: fill pending → exits → ratchet → signals.

Threading: all algo evaluation is serialized through _eval_lock.
"""

import datetime as dt
import json
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from v15.validation.unified_backtester.algo_base import (
    AlgoBase, AlgoConfig, Signal, ExitSignal, TradeContext,
)

logger = logging.getLogger(__name__)


@dataclass
class PendingEntry:
    """A delayed entry waiting for next bar open."""
    signal: Signal
    algo: AlgoBase
    queued_time: pd.Timestamp
    fill_at: str  # 'next_rth_open' or 'next_1min_open'


class LiveEngine:
    """Live execution engine for unified algo classes.

    Receives bar-close events from LiveDataProvider and dispatches to algos.
    Routes entry/exit signals to IBOrderHandler for IB paper trading.
    """

    def __init__(self, algos: List[AlgoBase], data, trade_db=None,
                 ib_order_handler=None, source: str = 'live'):
        self._algos = algos
        self._algo_map: Dict[str, AlgoBase] = {a.algo_id: a for a in algos}
        self._data = data
        self._db = trade_db
        self._orders = ib_order_handler
        self._source = source
        self._eval_lock = threading.Lock()
        self._pending_entries: List[PendingEntry] = []
        self._eval_counters: Dict[str, int] = {}
        self._algo_enabled: Dict[str, bool] = {a.algo_id: True for a in algos}
        self._kill_epoch: int = 0
        self._pre_kill_snapshot: Dict[str, bool] = {}  # saved enabled state for unkill

        # Wire fill callback from IBOrderHandler
        if self._orders and hasattr(self._orders, 'register_live_engine_callback'):
            self._orders.register_live_engine_callback(self.on_fill)

        # Start bar dispatch thread
        if self._data and hasattr(self._data, '_bar_queue'):
            self._dispatch_thread = threading.Thread(
                target=self._bar_dispatch_loop, daemon=True,
                name='LiveEngine-dispatch')
            self._dispatch_thread.start()

    def _bar_dispatch_loop(self):
        """Drain bar-close events from LiveDataProvider queue.

        Crash-safe: exceptions in on_bar_close are caught and logged,
        the loop continues. The thread never dies.
        """
        while True:
            try:
                info = self._data._bar_queue.get(timeout=60)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Unexpected error in bar queue: %s", e, exc_info=True)
                import time as _t
                _t.sleep(1)  # backoff to avoid spinning
                continue

            try:
                self.on_bar_close(info['tf'], info['time'], info['bar'])
            except Exception as e:
                logger.error("LiveEngine dispatch failed for %s bar at %s: %s",
                             info.get('tf'), info.get('time'), e, exc_info=True)

    def on_bar_close(self, tf: str, time: pd.Timestamp, bar: dict):
        """Called when a TF bar closes. Serialized, deadlock-safe.

        Collects IB operations in a list during _process_bar(), then
        executes them AFTER releasing _eval_lock to prevent deadlock
        (IB fill callbacks need to acquire _eval_lock).
        """
        # Normalize to tz-naive — bar DataFrames from historical seeding are
        # tz-naive, but live bar timestamps from tick aggregators are tz-aware ET.
        # All algo comparisons (df.index <= time) require consistent tz.
        if hasattr(time, 'tz') and time.tz is not None:
            time = time.tz_localize(None)
        elif hasattr(time, 'tzinfo') and time.tzinfo is not None:
            time = time.replace(tzinfo=None)

        logger.info("Bar close: %s @ %s  O=%.2f H=%.2f L=%.2f C=%.2f",
                     tf, time, bar.get('open', 0), bar.get('high', 0),
                     bar.get('low', 0), bar.get('close', 0))
        deferred_ib_ops = []
        with self._eval_lock:
            self._process_bar(tf, time, bar, deferred_ib_ops)

        # Execute IB operations AFTER releasing eval_lock
        for op in deferred_ib_ops:
            try:
                op()
            except Exception as e:
                logger.error("CRITICAL: Deferred IB op failed: %s", e, exc_info=True)

    def _process_bar(self, tf, time, bar, deferred_ib_ops=None):
        """Process a bar close. Follows backtester causal loop order.

        IB operations are appended to deferred_ib_ops (if provided) instead
        of being executed directly, to prevent deadlock with IB fill callbacks.
        """
        if deferred_ib_ops is None:
            deferred_ib_ops = []

        # 1. Fill pending delayed entries at this bar's open (1-min bars only)
        self._fill_pending_entries(tf, time, bar, deferred_ib_ops)

        for algo in self._algos:
            # Skip if wrong TF for this algo
            if tf != algo.config.primary_tf and tf != algo.config.exit_check_tf:
                continue

            algo_enabled = self._algo_enabled.get(algo.algo_id, True)

            # Respect eval_interval
            should_eval = False
            if tf == algo.config.primary_tf:
                counter = self._eval_counters.get(algo.algo_id, 0) + 1
                self._eval_counters[algo.algo_id] = counter
                should_eval = (counter % algo.config.eval_interval == 0)

            # Get open positions for this algo from DB
            positions = self._get_positions(algo)

            # Filter out manual-mode trades — algo must not generate exits for them
            algo_positions = positions
            if self._db and positions:
                algo_positions = []
                for pos in positions:
                    trade = self._db.get_trade(int(pos.pos_id))
                    if trade and trade.get('management_mode') == 'manual':
                        continue
                    algo_positions.append(pos)

            # 2. Check exits (ALWAYS — regardless of enabled state)
            if tf == algo.config.exit_check_tf and algo_positions:
                try:
                    exits = algo.check_exits(time, bar, algo_positions)
                except Exception as e:
                    logger.error("CRITICAL: %s.check_exits() FAILED — exits SKIPPED, "
                                 "positions may be unprotected: %s",
                                 algo.algo_id, e, exc_info=True)
                    continue
                for exit_sig in exits:
                    self._execute_exit(algo, exit_sig, deferred_ib_ops)

                # 3. Re-fetch positions after exits, then ratchet
                positions = self._get_positions(algo)
                self._ratchet_positions(algo, positions, bar)

                # 4. Sync broker-side trailing stops (deferred)
                self._sync_trailing_stops(algo, positions, deferred_ib_ops)

            # 5. Active window gate — only applies to ENTRY generation
            if algo.config.active_start and algo.config.active_end:
                bar_time = time.time() if hasattr(time, 'time') else None
                if bar_time:
                    if not (algo.config.active_start <= bar_time
                            <= algo.config.active_end):
                        continue

            # 6. Generate new entry signals (signals always run for visibility)
            if should_eval:
                context = self._build_trade_context(algo)
                try:
                    signals = algo.on_bar(time, bar, positions, context=context)
                except Exception as e:
                    logger.error("CRITICAL: %s.on_bar() FAILED — signals SKIPPED: %s",
                                 algo.algo_id, e, exc_info=True)
                    continue
                for sig in signals:
                    if not algo_enabled:
                        logger.info("BLOCKED: %s signal %s (algo disabled)",
                                    algo.algo_id, sig.direction)
                        continue
                    if sig.delayed_entry:
                        fill_at = ('next_rth_open'
                                   if algo.config.primary_tf == 'daily'
                                   else 'next_1min_open')
                        self._pending_entries.append(PendingEntry(
                            signal=sig, algo=algo, queued_time=time,
                            fill_at=fill_at))
                        self._persist_pending_entries()
                        logger.info("Queued delayed entry for %s (%s, conf=%.2f, fill_at=%s)",
                                    algo.algo_id, sig.direction, sig.confidence, fill_at)
                    else:
                        self._execute_entry(algo, sig, bar['close'],
                                            deferred_ib_ops)

    def _get_positions(self, algo):
        """Get open positions for an algo from DB, convert to Position objects.

        For IB algos (live_orders=True): passes source='ib' which excludes
        pending rows (only partial/filled = real broker exposure).
        For sim algos: passes engine source (e.g. 'yf').
        """
        if not self._db:
            return []
        try:
            from v15.validation.unified_backtester.portfolio import Position
            source = 'ib' if algo.config.live_orders else self._source
            open_trades = self._db.get_open_trades(
                algo_id=algo.algo_id, source=source)
            positions = []
            for trade in open_trades:
                metadata = {}
                if trade.get('metadata'):
                    try:
                        metadata = json.loads(trade['metadata'])
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.error("Failed to parse metadata for trade %d: %s",
                                     trade['id'], e)
                pos = Position(
                    pos_id=str(trade['id']),
                    algo_id=trade['algo_id'],
                    direction=trade.get('direction', 'long'),
                    entry_price=trade['entry_price'],
                    entry_time=trade.get('entry_time', ''),
                    shares=trade.get('shares', 0),
                    notional=trade['entry_price'] * trade.get('shares', 0),
                    stop_price=trade.get('stop_price', 0),
                    tp_price=trade.get('tp_price', 0),
                    best_price=trade.get('best_price', trade['entry_price']),
                    worst_price=trade.get('worst_price', trade['entry_price']),
                    hold_bars=trade.get('hold_bars', 0),
                    confidence=trade.get('confidence', 0.5),
                    signal_type=trade.get('signal_type', ''),
                    metadata=metadata,
                )
                positions.append(pos)
            return positions
        except Exception as e:
            logger.error("CRITICAL: Failed to get positions for %s: %s",
                         algo.algo_id, e, exc_info=True)
            return []

    def _ratchet_positions(self, algo, positions, bar):
        """Direction-aware best/worst/hold_bars update."""
        if not self._db:
            return
        for pos in positions:
            trade_id = int(pos.pos_id)
            changes = {}
            if pos.direction == 'long':
                if bar['high'] > pos.best_price:
                    changes['best_price'] = bar['high']
                if bar['low'] < pos.worst_price:
                    changes['worst_price'] = bar['low']
            else:  # short
                if bar['low'] < pos.best_price:
                    changes['best_price'] = bar['low']
                if bar['high'] > pos.worst_price:
                    changes['worst_price'] = bar['high']
            changes['hold_bars'] = pos.hold_bars + 1
            try:
                self._db.update_trade_state(trade_id, **changes)
            except Exception as e:
                logger.error("Ratchet failed for trade %d: %s", trade_id, e)

    def _sync_trailing_stops(self, algo, positions, deferred_ib_ops=None):
        """Sync effective stop from algo state to IB resting stop orders."""
        import math
        for pos in positions:
            # Skip manual trades — user manages their own stops
            if self._db:
                trade = self._db.get_trade(int(pos.pos_id))
                if trade and trade.get('management_mode') == 'manual':
                    continue
            try:
                effective_stop = algo.get_effective_stop(pos)
            except Exception as e:
                logger.error("CRITICAL: %s.get_effective_stop() FAILED for trade %s: %s",
                             algo.algo_id, pos.pos_id, e, exc_info=True)
                continue
            trade_id = int(pos.pos_id)
            if effective_stop is not None and not isinstance(effective_stop, (int, float)):
                logger.error("CRITICAL: %s.get_effective_stop() returned invalid type %s "
                             "for trade %d", algo.algo_id, type(effective_stop), trade_id)
                continue
            if effective_stop and (math.isnan(effective_stop) or math.isinf(effective_stop)):
                logger.error("CRITICAL: %s.get_effective_stop() returned %s for trade %d",
                             algo.algo_id, effective_stop, trade_id)
                continue
            if effective_stop and effective_stop <= 0:
                logger.error("CRITICAL: %s.get_effective_stop() returned %.2f for trade %d",
                             algo.algo_id, effective_stop, trade_id)
                continue
            if effective_stop and effective_stop != pos.stop_price:
                try:
                    self._db.update_trade_state(trade_id,
                                                stop_price=effective_stop)
                except Exception as e:
                    logger.error("Stop sync DB failed for trade %d: %s",
                                 trade_id, e)
                if self._orders:
                    # Defer IB call to prevent deadlock
                    if deferred_ib_ops is not None:
                        deferred_ib_ops.append(
                            lambda tid=trade_id, s=effective_stop:
                                self._orders.modify_trailing_stop(tid, s))
                    else:
                        try:
                            self._orders.modify_trailing_stop(trade_id,
                                                               effective_stop)
                        except Exception as e:
                            logger.error("Stop sync IB failed for trade %d: %s",
                                         trade_id, e)

    def _fill_pending_entries(self, tf, time, bar, deferred_ib_ops=None):
        """Fill delayed entries. Only on 1-min bars for precise timing."""
        if tf != '1min':
            return
        remaining = []
        # Use wall-clock Eastern time for RTH check — bar timestamps may be
        # in the server's local timezone (CDT), not ET
        from datetime import datetime as _dt
        from zoneinfo import ZoneInfo
        now_et = _dt.now(ZoneInfo('US/Eastern')).time()

        for pending in self._pending_entries:
            # Check enabled gate at fill time (algo may have been disabled since queued)
            if not self._algo_enabled.get(pending.algo.algo_id, True):
                logger.info("BLOCKED: %s pending entry (algo disabled)",
                            pending.algo.algo_id)
                continue  # Drop the pending entry

            if pending.fill_at == 'next_rth_open':
                # RTH opens at 9:30 ET; first end-indexed 1-min bar = 9:31 ET
                # Use wall-clock ET time (server may be in CDT/PST/etc.)
                # Window is 9:31-9:45 to handle late first ticks after open
                if dt.time(9, 31) <= now_et < dt.time(9, 45):
                    self._execute_entry(pending.algo, pending.signal,
                                        bar['open'], deferred_ib_ops)
                else:
                    remaining.append(pending)
            elif pending.fill_at == 'next_1min_open':
                # Guard: must NOT fill on the same bar the signal was generated
                if pending.queued_time >= time:
                    remaining.append(pending)
                else:
                    self._execute_entry(pending.algo, pending.signal,
                                        bar['open'], deferred_ib_ops)
            else:
                remaining.append(pending)
        n_before = len(self._pending_entries)
        self._pending_entries = remaining
        # Update persisted entries if any were consumed or dropped
        if n_before > 0 and len(remaining) < n_before:
            self._persist_pending_entries()

    def _execute_entry(self, algo, signal, fill_price, deferred_ib_ops=None):
        """Place entry order or record sim trade."""
        # Final enabled check (belt-and-suspenders — catches race with kill_all)
        if not self._algo_enabled.get(algo.algo_id, True):
            logger.info("BLOCKED: %s entry in _execute_entry (algo disabled)",
                        algo.algo_id)
            return

        if not algo.config.live_orders:
            # Sim mode: instant DB write (no IB calls, safe inside lock)
            if self._db:
                try:
                    if signal.direction == 'long':
                        stop_price = round(fill_price * (1 - signal.stop_pct), 2)
                        tp_price = round(fill_price * (1 + signal.tp_pct), 2)
                    else:
                        stop_price = round(fill_price * (1 + signal.stop_pct), 2)
                        tp_price = round(fill_price * (1 - signal.tp_pct), 2)

                    shares = signal.shares
                    if shares == 0:
                        equity = algo.config.max_equity_per_trade
                        shares = max(1, int(equity / fill_price))

                    from datetime import datetime
                    from zoneinfo import ZoneInfo
                    now_et = datetime.now(ZoneInfo('US/Eastern')).isoformat()
                    trade_id = self._db.open_trade(
                        source=self._source,
                        algo_id=algo.algo_id,
                        symbol='TSLA',
                        direction=signal.direction,
                        entry_time=now_et,
                        entry_price=fill_price,
                        shares=shares,
                        stop_price=stop_price,
                        tp_price=tp_price,
                        confidence=signal.confidence,
                        signal_type=signal.signal_type,
                        best_price=fill_price,
                        worst_price=fill_price,
                        ou_half_life=signal.metadata.get('ou_half_life'),
                        el_flagged=signal.metadata.get('el_flagged', False),
                        trail_width_mult=signal.metadata.get('trail_width_mult', 1.0),
                        metadata=signal.metadata,
                    )
                    trade = self._db.get_trade(trade_id)
                    if trade:
                        pos = self._get_positions(algo)
                        for p in pos:
                            if p.pos_id == str(trade_id):
                                try:
                                    algo.on_position_opened(p)
                                except Exception as e2:
                                    logger.error("CRITICAL: %s.on_position_opened() FAILED "
                                                 "in sim entry for trade %d: %s",
                                                 algo.algo_id, trade_id, e2, exc_info=True)
                                break
                except Exception as e:
                    logger.error("CRITICAL: Sim entry FAILED for %s — entry DROPPED: %s",
                                 algo.algo_id, e, exc_info=True)
            return

        # IB mode: defer order placement to prevent deadlock
        if self._orders:
            if signal.direction == 'long':
                stop_price = round(fill_price * (1 - signal.stop_pct), 2)
                tp_price = round(fill_price * (1 + signal.tp_pct), 2)
            else:
                stop_price = round(fill_price * (1 + signal.stop_pct), 2)
                tp_price = round(fill_price * (1 - signal.tp_pct), 2)

            shares = signal.shares
            if shares == 0:
                equity = algo.config.max_equity_per_trade
                shares = max(1, int(equity / fill_price))

            entry_kwargs = dict(
                algo_id=algo.algo_id,
                direction=signal.direction,
                shares=shares,
                stop_price=stop_price,
                tp_price=tp_price,
                confidence=signal.confidence,
                signal_type=signal.signal_type,
                trail_width=signal.metadata.get('trail_width', 0.01),
                ou_half_life=signal.metadata.get('ou_half_life', 5.0),
                el_flagged=signal.metadata.get('el_flagged', False),
                trail_width_mult=signal.metadata.get('trail_width_mult', 1.0),
                entry_price=fill_price,
                stop_pct=signal.stop_pct,
                tp_pct=signal.tp_pct,
            )
            if deferred_ib_ops is not None:
                epoch_at_signal = self._kill_epoch
                deferred_ib_ops.append(
                    lambda kw=entry_kwargs, ep=epoch_at_signal: (
                        self._orders.place_entry(**kw)
                        if self._kill_epoch == ep
                        else logger.info("STALE: entry for %s dropped (kill epoch %d→%d)",
                                         kw['algo_id'], ep, self._kill_epoch)
                    ))
            else:
                try:
                    self._orders.place_entry(**entry_kwargs)
                except Exception as e:
                    logger.error("IB entry failed for %s: %s",
                                 algo.algo_id, e)

    def _execute_exit(self, algo, exit_signal, deferred_ib_ops=None):
        """Place closing order."""
        trade_id = int(exit_signal.pos_id)

        if not algo.config.live_orders:
            # Sim mode: instant close (no IB calls, safe inside lock)
            if self._db:
                try:
                    from datetime import datetime
                    from zoneinfo import ZoneInfo
                    now_et = datetime.now(ZoneInfo('US/Eastern')).isoformat()
                    self._db.close_trade(
                        trade_id, exit_time=now_et,
                        exit_price=exit_signal.price,
                        exit_reason=exit_signal.reason)
                except Exception as e:
                    logger.error("CRITICAL: Sim exit FAILED for trade %d — "
                                 "position NOT closed in DB: %s",
                                 trade_id, e, exc_info=True)
            return

        # IB mode: defer order placement to prevent deadlock
        if self._orders:
            if deferred_ib_ops is not None:
                deferred_ib_ops.append(
                    lambda tid=trade_id, reason=exit_signal.reason,
                           price=exit_signal.price:
                        self._orders.place_exit(tid, reason, price))
            else:
                try:
                    self._orders.place_exit(
                        trade_id, exit_signal.reason, exit_signal.price)
                except Exception as e:
                    logger.error("IB exit failed for trade %d: %s",
                                 trade_id, e)

    def on_fill(self, trade_id: int, fill_price: float, fill_qty: int,
                is_entry: bool):
        """Callback from IBOrderHandler on IB fill."""
        with self._eval_lock:
            if not self._db:
                logger.error("CRITICAL: on_fill(trade=%d) — DB unavailable, fill LOST",
                             trade_id)
                return
            trade = self._db.get_trade(trade_id)
            if not trade:
                logger.error("CRITICAL: on_fill(trade=%d) — trade NOT FOUND in DB, "
                             "IB has fill but no DB record (price=%.2f, qty=%d, entry=%s)",
                             trade_id, fill_price, fill_qty, is_entry)
                return
            algo = self._algo_map.get(trade.get('algo_id'))
            if not algo:
                logger.error("CRITICAL: on_fill(trade=%d) — algo_id '%s' not in engine "
                             "(known algos: %s)",
                             trade_id, trade.get('algo_id'),
                             list(self._algo_map.keys()))
                return
            if is_entry:
                # Only initialize algo state on first fill, not partial follow-ups
                prior_filled = (trade.get('filled_shares', 0) or 0) - fill_qty
                if prior_filled <= 0:
                    pos = self._get_positions(algo)
                    for p in pos:
                        if p.pos_id == str(trade_id):
                            try:
                                algo.on_position_opened(p)
                            except Exception as e:
                                logger.error("CRITICAL: %s.on_position_opened() FAILED "
                                             "for trade %d — algo state NOT initialized: %s",
                                             algo.algo_id, trade_id, e, exc_info=True)
                            # Persist algo state
                            try:
                                state = algo.serialize_state(p.pos_id)
                                if state:
                                    metadata = json.loads(
                                        trade.get('metadata', '{}') or '{}')
                                    metadata['algo_state'] = state
                                    self._db.update_trade_state(
                                        trade_id, metadata=json.dumps(metadata))
                            except Exception as e:
                                logger.error("Failed to persist algo state for trade %d: %s",
                                             trade_id, e, exc_info=True)
                            break
            else:
                # Adapt dict to have pos_id attribute for algo.on_fill()
                class _TradeProxy:
                    def __init__(self, d):
                        self.pos_id = str(d.get('id', ''))
                        self.algo_id = d.get('algo_id', '')
                        self.pnl = d.get('pnl', 0.0)
                        self.net_pnl = d.get('pnl', 0.0)
                try:
                    algo.on_fill(_TradeProxy(trade))
                except Exception as e:
                    logger.error("CRITICAL: %s.on_fill() FAILED for trade %d: %s",
                                 algo.algo_id, trade_id, e, exc_info=True)

    def _build_trade_context(self, algo) -> TradeContext:
        """Build TradeContext from TradeDB queries."""
        if not self._db:
            return TradeContext()
        try:
            source = 'ib' if algo.config.live_orders else self._source
            closed = self._db.get_closed_trades(algo_id=algo.algo_id,
                                                 source=source, limit=10)
            recent_dicts = [{'pnl': t.get('pnl', 0), 'pnl_pct': t.get('pnl_pct', 0)}
                            for t in (closed or [])]
            win_streak = 0
            loss_streak = 0
            for t in reversed(closed or []):
                pnl = t.get('pnl', 0)
                if pnl > 0:
                    if loss_streak > 0:
                        break
                    win_streak += 1
                elif pnl < 0:
                    if win_streak > 0:
                        break
                    loss_streak += 1
            return TradeContext(
                recent_trades=recent_dicts,
                win_streak=win_streak,
                loss_streak=loss_streak,
            )
        except Exception as e:
            logger.error("Failed to build TradeContext for %s: %s",
                         algo.algo_id, e, exc_info=True)
            return TradeContext()

    # ── UI Mutation Methods (all acquire _eval_lock) ──

    def kill_all(self):
        """Disable all algos, purge pending entries, cancel IB pending + stop orders."""
        deferred_entry_cancels = []
        deferred_stop_cancels = []
        with self._eval_lock:
            # Save pre-kill state for unkill restore
            self._pre_kill_snapshot = dict(self._algo_enabled)
            self._kill_epoch += 1
            for algo_id in self._algo_enabled:
                self._algo_enabled[algo_id] = False
            # Purge pending entries
            n_purged = len(self._pending_entries)
            self._pending_entries.clear()
            # Collect pending entry order cancels
            if self._orders:
                for order_id in list(self._orders._entry_orders):
                    ctx = self._orders._entry_orders[order_id]
                    if not ctx.get('trade_id'):
                        deferred_entry_cancels.append(order_id)
            # Collect resting stop order cancels (leaves positions unprotected!)
            if self._orders and self._db:
                try:
                    open_trades = self._db.get_open_trades(source='ib')
                    for trade in open_trades:
                        stop_oid = trade.get('ib_stop_order_id')
                        if stop_oid:
                            deferred_stop_cancels.append(
                                (stop_oid, trade['id']))
                except Exception as e:
                    logger.error("kill_all: failed to get open trades: %s", e)
            logger.warning("LiveEngine: kill_all — epoch=%d, purged=%d entries, "
                           "%d stops to cancel",
                           self._kill_epoch, n_purged, len(deferred_stop_cancels))

        # Cancel IB orders outside lock
        for oid in deferred_entry_cancels:
            try:
                self._orders.ib.cancel_order(oid)
                logger.info("kill_all: cancelled IB entry order %d", oid)
            except Exception as e:
                logger.error("kill_all: failed to cancel entry %d: %s", oid, e)
        for stop_oid, trade_id in deferred_stop_cancels:
            try:
                # NULL the stop in DB before cancelling
                self._db.update_trade_state(trade_id, ib_stop_order_id=None)
                self._orders.ib.cancel_order(stop_oid)
                logger.info("kill_all: cancelled resting stop %d (trade %d)",
                            stop_oid, trade_id)
            except Exception as e:
                logger.error("kill_all: failed to cancel stop %d: %s",
                             stop_oid, e)

    def unkill(self):
        """Restore pre-kill enabled state (or enable all if no snapshot)."""
        with self._eval_lock:
            if self._pre_kill_snapshot:
                for algo_id, was_enabled in self._pre_kill_snapshot.items():
                    if algo_id in self._algo_enabled:
                        self._algo_enabled[algo_id] = was_enabled
                logger.info("LiveEngine: unkill — restored pre-kill state: %s",
                            self._pre_kill_snapshot)
                self._pre_kill_snapshot.clear()
            else:
                for algo_id in self._algo_enabled:
                    self._algo_enabled[algo_id] = True
                logger.info("LiveEngine: unkill — all algos enabled (no snapshot)")

    def set_algo_enabled(self, algo_id: str, enabled: bool):
        """Enable/disable a specific algo. Persists to DB metadata."""
        with self._eval_lock:
            self._algo_enabled[algo_id] = enabled
            if not enabled:
                # Purge pending entries for this algo
                before = len(self._pending_entries)
                self._pending_entries = [
                    p for p in self._pending_entries
                    if p.algo.algo_id != algo_id
                ]
                purged = before - len(self._pending_entries)
                if purged:
                    logger.info("Purged %d pending entries for disabled %s",
                                purged, algo_id)
            logger.info("LiveEngine: %s %s",
                         algo_id, 'enabled' if enabled else 'disabled')
        # Persist outside lock
        if self._db:
            try:
                self._db.set_metadata(f'enabled_{algo_id}',
                                       '1' if enabled else '0')
            except Exception as e:
                logger.error("Failed to persist enabled state for %s: %s",
                             algo_id, e)

    def set_algo_equity(self, algo_id: str, equity: float):
        """Set per-algo max equity. Thread-safe (mutates config under _eval_lock)."""
        with self._eval_lock:
            algo = self._algo_map.get(algo_id)
            if algo:
                algo.config.max_equity_per_trade = equity
                logger.info("LiveEngine: %s equity set to $%.0f", algo_id, equity)
        # Persist outside lock
        if self._db:
            try:
                self._db.set_metadata(f'equity_{algo_id}', str(int(equity)))
            except Exception as e:
                logger.error("Failed to persist equity for %s: %s", algo_id, e)

    def get_algo_equity(self, algo_id: str) -> float:
        """Get current max equity for an algo."""
        algo = self._algo_map.get(algo_id)
        if algo:
            return algo.config.max_equity_per_trade
        return 0.0

    def reset_algo(self, algo_id: str):
        """Reset an algo's counters and pending entries."""
        with self._eval_lock:
            self._eval_counters[algo_id] = 0
            self._pending_entries = [p for p in self._pending_entries
                                      if p.algo.algo_id != algo_id]
            logger.info("LiveEngine: reset %s", algo_id)

    def serialize_all(self):
        """Persist all algo state to DB."""
        with self._eval_lock:
            for algo in self._algos:
                positions = self._get_positions(algo)
                for pos in positions:
                    try:
                        state = algo.serialize_state(pos.pos_id)
                    except Exception as e:
                        logger.error("CRITICAL: %s.serialize_state() FAILED for "
                                     "trade %s: %s", algo.algo_id, pos.pos_id, e,
                                     exc_info=True)
                        continue
                    if state and self._db:
                        trade_id = int(pos.pos_id)
                        trade = self._db.get_trade(trade_id)
                        if trade:
                            metadata = json.loads(
                                trade.get('metadata', '{}') or '{}')
                            metadata['algo_state'] = state
                            self._db.update_trade_state(
                                trade_id, metadata=json.dumps(metadata))

    def recover_after_restart(self):
        """Recover all state from DB after dashboard restart."""
        with self._eval_lock:
            for algo in self._algos:
                if not self._db:
                    continue
                open_trades = self._db.get_open_trades(algo_id=algo.algo_id)
                for trade in open_trades:
                    metadata = {}
                    if trade.get('metadata'):
                        try:
                            metadata = json.loads(trade['metadata'])
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.error("Failed to parse metadata for trade %d "
                                         "during recovery: %s", trade['id'], e)
                    algo_state = metadata.get('algo_state')
                    if algo_state:
                        try:
                            algo.restore_state(str(trade['id']), algo_state)
                        except Exception as e:
                            logger.error("CRITICAL: %s.restore_state() FAILED for "
                                         "trade %d — falling back to on_position_opened: %s",
                                         algo.algo_id, trade['id'], e, exc_info=True)
                            # Fallback: reconstruct from DB
                            positions = self._get_positions(algo)
                            for pos in positions:
                                if pos.pos_id == str(trade['id']):
                                    try:
                                        algo.on_position_opened(pos)
                                    except Exception as e2:
                                        logger.error("CRITICAL: %s.on_position_opened() "
                                                     "also FAILED for trade %d: %s",
                                                     algo.algo_id, trade['id'], e2,
                                                     exc_info=True)
                                    break
                    else:
                        # Reconstruct from DB fields
                        positions = self._get_positions(algo)
                        for pos in positions:
                            if pos.pos_id == str(trade['id']):
                                try:
                                    algo.on_position_opened(pos)
                                except Exception as e:
                                    logger.error("CRITICAL: %s.on_position_opened() FAILED "
                                                 "during recovery for trade %d: %s",
                                                 algo.algo_id, trade['id'], e, exc_info=True)
                                break

                # Restore per-algo counters
                if hasattr(algo, '_current_day'):
                    algo._current_day = dt.date.today()
                if hasattr(algo, '_trades_today'):
                    today_trades = self._db.get_trades_for_day(
                        algo_id=algo.algo_id) if hasattr(self._db, 'get_trades_for_day') else []
                    algo._trades_today = len([t for t in today_trades
                                              if t.get('exit_time')])

            # Recover pending entries from DB metadata
            self._recover_pending_entries()

            logger.info("LiveEngine: recovery complete for %d algos",
                         len(self._algos))

    def _persist_pending_entries(self):
        """Persist pending delayed entries to DB metadata as JSON.

        Called after queuing a delayed entry so it survives restarts.
        Daily algos fire at 16:00 and queue for 9:31 next day — the
        dashboard may restart overnight.
        """
        if not self._db:
            return
        try:
            entries = []
            for p in self._pending_entries:
                entries.append({
                    'algo_id': p.algo.algo_id,
                    'direction': p.signal.direction,
                    'confidence': p.signal.confidence,
                    'stop_pct': p.signal.stop_pct,
                    'tp_pct': p.signal.tp_pct,
                    'signal_type': p.signal.signal_type,
                    'shares': p.signal.shares,
                    'metadata': p.signal.metadata,
                    'queued_time': str(p.queued_time),
                    'fill_at': p.fill_at,
                })
            self._db.set_metadata('pending_entries', json.dumps(entries))
        except Exception as e:
            logger.error("Failed to persist pending entries: %s", e)

    def _recover_pending_entries(self):
        """Recover pending delayed entries from DB metadata after restart."""
        if not self._db:
            return
        try:
            raw = self._db.get_metadata('pending_entries')
            if not raw:
                return
            entries = json.loads(raw)
            if not entries:
                return
            for entry in entries:
                algo = self._algo_map.get(entry['algo_id'])
                if not algo:
                    logger.warning("Pending entry recovery: algo %s not found, skipping",
                                   entry['algo_id'])
                    continue
                sig = Signal(
                    algo_id=entry['algo_id'],
                    direction=entry['direction'],
                    price=0,  # Will be filled at bar open
                    confidence=entry['confidence'],
                    stop_pct=entry['stop_pct'],
                    tp_pct=entry['tp_pct'],
                    signal_type=entry.get('signal_type', ''),
                    shares=entry.get('shares', 0),
                    metadata=entry.get('metadata', {}),
                    delayed_entry=True,
                )
                self._pending_entries.append(PendingEntry(
                    signal=sig, algo=algo,
                    queued_time=pd.Timestamp(entry['queued_time']),
                    fill_at=entry['fill_at'],
                ))
            logger.info("Recovered %d pending entries from DB: %s",
                         len(entries),
                         [e['algo_id'] for e in entries])
            # Clear from DB after loading (will be re-persisted if still pending)
            self._db.set_metadata('pending_entries', '[]')
        except Exception as e:
            logger.error("Failed to recover pending entries: %s", e)
