"""
IB Two-Phase Commit Order Handler.

Manages the full lifecycle of IB orders: entry, exit, protective stops.
Handles race conditions between IB fill callbacks and DB writes via
in-memory buffers protected by _buffer_lock.

This module sits between LiveEngine/order_entry and IBClient.
Both automated and manual orders use the same handler.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
ET = ZoneInfo('US/Eastern')


def _now_eastern() -> str:
    """Current time as US/Eastern ISO 8601."""
    return datetime.now(ET).isoformat()


# ── Data classes for buffers ────────────────────────────────────────

@dataclass
class FillData:
    """Single execution fill from IB."""
    exec_id: str
    shares: int
    price: float
    time: str  # broker execution timestamp (US/Eastern ISO 8601)
    order_id: int = 0


@dataclass
class TerminalStatus:
    """Terminal order status (Cancelled/Rejected/Filled)."""
    status: str
    timestamp: str
    filled_shares: int = 0


@dataclass
class FailedOrderContext:
    """Context for an entry order whose DB write failed."""
    trade_id: Optional[int]  # None if row never created
    stop_price: float
    algo_id: str
    direction: str
    shares: int
    filled_shares: int = 0


@dataclass
class FailedExitContext:
    """Context for an exit order whose DB update failed."""
    trade_id: int
    open_shares_at_failure: int
    rearmed_stop_order_id: Optional[int] = None


@dataclass
class StopInfo:
    """Tracking info for an emergency/protective stop."""
    stop_order_id: int
    stop_perm_id: int
    trade_id: Optional[int]
    stop_price: float
    qty: int


@dataclass
class TerminalInfo:
    """Tombstone for a terminal order (persists for session)."""
    status: str
    timestamp: str
    filled_shares_at_terminal: int
    context: dict = field(default_factory=dict)


class IBOrderHandler:
    """Manages IB order lifecycle with two-phase commit.

    Shared between LiveEngine (automated) and order_entry (manual).
    All buffers are protected by _buffer_lock for thread safety.
    """

    def __init__(self, state):
        """
        Args:
            state: DashboardState with .ib_client, .trade_db, .ib_degraded, etc.
        """
        self._state = state

        # Early-fill buffers (IB callback can fire before DB row exists)
        self._pending_fills: dict[int, list[FillData]] = {}        # entry order_id -> fills
        self._pending_exit_fills: dict[int, list[FillData]] = {}   # exit order_id -> fills
        self._pending_terminal: dict[int, TerminalStatus] = {}     # entry order_id -> status
        self._pending_exit_terminal: dict[int, TerminalStatus] = {}  # exit order_id -> status
        self._buffer_lock = threading.Lock()

        # Failed order tracking
        self._failed_orders: dict[int, FailedOrderContext] = {}     # entry order_id -> context
        self._failed_exit_orders: dict[int, FailedExitContext] = {} # exit order_id -> context

        # Emergency/protective stop tracking
        self._emergency_stops: dict[int, StopInfo] = {}  # keyed by entry order_id
        self._rearmed_stops: dict[int, int] = {}         # trade_id -> stop_order_id

        # Per-trade stop serialization
        self._stop_locks: dict[int, threading.Lock] = {}
        self._stop_dirty: dict[int, bool] = {}

        # Exit-in-progress flags
        self._exit_in_progress: dict[int, bool] = {}  # trade_id -> bool

        # Terminal order tombstones (persist for entire session)
        self._terminal_orders: dict[int, TerminalInfo] = {}

        # Execution deduplication
        self.seen_exec_ids: set[str] = set()

        # Registered order_ids -> trade context for callback routing
        self._entry_orders: dict[int, dict] = {}  # order_id -> {trade_id, algo_id, direction, stop_price, tp_price}
        self._exit_orders: dict[int, dict] = {}   # order_id -> {trade_id, exit_reason}
        self._stop_orders: dict[int, dict] = {}   # order_id -> {trade_id}

        # LiveEngine fill callback (set via register_live_engine_callback)
        self._live_engine_callback = None

    @property
    def db(self):
        return self._state.trade_db

    @property
    def ib(self):
        return self._state.ib_client

    def register_live_engine_callback(self, callback):
        """Register LiveEngine fill callback.

        callback(trade_id: int, fill_price: float, fill_qty: int, is_entry: bool)
        Called from IB event loop thread. Callback should acquire its own lock.
        """
        self._live_engine_callback = callback

    def _notify_live_engine(self, trade_id, fill_price, fill_qty, is_entry):
        """Notify LiveEngine of a fill (if callback registered)."""
        if self._live_engine_callback:
            try:
                self._live_engine_callback(
                    trade_id=trade_id, fill_price=fill_price,
                    fill_qty=fill_qty, is_entry=is_entry)
            except Exception as e:
                logger.error("LiveEngine %s callback failed: %s",
                             'entry' if is_entry else 'exit', e)

    def _set_degraded(self, reason: str):
        """Set ib_degraded=True and persist to DB metadata."""
        self._state.ib_degraded = True
        try:
            self.db.set_metadata('ib_degraded', '1')
        except Exception as e:
            logger.error("Failed to persist ib_degraded: %s", e)
        logger.error("IB DEGRADED: %s", reason)

    def _fire_trade_alert(self, action: str, algo_id: str, direction: str,
                          shares: int, price: float, **kwargs):
        """Fire UI alert card + Telegram notification for entry/exit.

        Works for any algo — uses algo_id dynamically.
        """
        import threading as _th

        now = datetime.now(ET).strftime('%H:%M:%S ET')
        if action == 'entry':
            signal_type = kwargs.get('signal_type', '')
            stop_price = kwargs.get('stop_price', 0)
            tp_price = kwargs.get('tp_price', 0)
            emoji = '\U0001f7e2' if direction == 'long' else '\U0001f534'
            title = f"{emoji} {algo_id} — {direction.upper()} Entry"
            body = (f"{shares} shares TSLA @ ~${price:.2f}\n"
                    f"Signal: {signal_type}\n"
                    f"Stop: ${stop_price:.2f} | TP: ${tp_price:.2f}\n"
                    f"Time: {now}")
            alert_type = 'entry'
            card_color = '#1a472a' if direction == 'long' else '#4a1a1a'
            card_icon = '\u2191' if direction == 'long' else '\u2193'
            card_html = (
                f'<div style="background:{card_color};border-radius:8px;'
                f'padding:12px 16px;margin:8px 0;color:#fff;'
                f'font-family:monospace;border-left:4px solid '
                f'{"#4caf50" if direction == "long" else "#f44336"};">'
                f'<b>{card_icon} {algo_id}</b> &mdash; '
                f'{direction.upper()} {shares} shares @ ${price:.2f}<br>'
                f'<small>{signal_type} | Stop ${stop_price:.2f} '
                f'| TP ${tp_price:.2f} | {now}</small></div>')
        else:
            exit_reason = kwargs.get('exit_reason', 'exit')
            pnl = kwargs.get('pnl', 0)
            entry_price = kwargs.get('entry_price', 0)
            is_profit = pnl >= 0
            emoji = '\u2705' if is_profit else '\u274c'
            title = f"{emoji} {algo_id} — {direction.upper()} Exit ({exit_reason})"
            body = (f"{shares} shares TSLA @ ${price:.2f}\n"
                    f"Entry: ${entry_price:.2f} | P&L: ${pnl:+.2f}\n"
                    f"Reason: {exit_reason}\n"
                    f"Time: {now}")
            alert_type = 'exit_profit' if is_profit else 'exit_loss'
            card_color = '#1a3a1a' if is_profit else '#3a1a1a'
            border_color = '#4caf50' if is_profit else '#f44336'
            card_html = (
                f'<div style="background:{card_color};border-radius:8px;'
                f'padding:12px 16px;margin:8px 0;color:#fff;'
                f'font-family:monospace;border-left:4px solid {border_color};">'
                f'<b>{emoji} {algo_id}</b> &mdash; '
                f'{exit_reason} {shares} shares @ ${price:.2f}<br>'
                f'<small>Entry ${entry_price:.2f} | '
                f'P&L <span style="color:{border_color}">${pnl:+.2f}</span>'
                f' | {now}</small></div>')

        # Set UI alert (triggers card + audio reactively)
        self._state.trade_alert_type = alert_type
        self._state.trade_alert_html = card_html

        # Send Telegram in background thread (non-blocking)
        def _send():
            try:
                self._state.send_notification(body, title=title)
            except Exception as e:
                logger.error("Trade alert Telegram failed: %s", e)
        _th.Thread(target=_send, daemon=True, name='trade-alert-tg').start()

    # ── Entry Flow ──────────────────────────────────────────────────

    def place_entry(self, algo_id: str, direction: str, shares: int,
                    stop_price: float, tp_price: float, confidence: float,
                    signal_type: str, trail_width: float = 0.01,
                    ou_half_life: float = 5.0, el_flagged: bool = False,
                    trail_width_mult: float = 1.0,
                    entry_price: float = 0.0,
                    stop_pct: float = 0.0, tp_pct: float = 0.0) -> Optional[int]:
        """Place an IB entry order with two-phase commit.

        Returns trade_id on success, None on failure.
        """
        if self._state.ib_degraded:
            logger.warning("Entry blocked: ib_degraded=True")
            return None
        if not self.ib or not self.ib.is_connected():
            logger.warning("Entry blocked: IB not connected")
            return None

        # Block opposite-direction entry when another algo has an open position.
        # Non-FA accounts have a single IB position bucket — opposite trades
        # would net to 0 at the broker, making stops/exits/reconciliation invalid.
        open_trades = self.db.get_open_trades(source='ib')
        opposite = [t for t in open_trades
                    if t.get('direction') != direction
                    and t.get('open_shares', 0) > 0]
        if opposite:
            opp_algos = [t.get('algo_id', '?') for t in opposite]
            opp_shares = sum(t.get('open_shares', 0) for t in opposite)
            logger.warning(
                "Entry BLOCKED: %s wants %s %d shares but %d opposite-direction "
                "trades open (algos=%s, %d shares) — would corrupt IB net position",
                algo_id, direction, shares, len(opposite), opp_algos, opp_shares)
            return None

        action = 'BUY' if direction == 'long' else 'SELL'
        order_ref = f'entry:{algo_id}:{direction}:{stop_price:.2f}:{tp_price:.2f}'
        model_code = algo_id if self._state.fa_supported else None

        # Step 1: Place IB order
        result = self.ib.place_order(
            'TSLA', action, shares, 'MKT',
            order_ref=order_ref, model_code=model_code)

        if 'error' in result:
            error_msg = result['error']
            logger.error("Entry order failed: %s", error_msg)
            # If timeout, the order may be in-flight — set degraded so
            # recovery on next restart can find it via orderRef
            if 'Timeout' in str(error_msg) or 'in-flight' in str(error_msg):
                self._set_degraded(f"Entry order timeout for {algo_id} — "
                                   f"order may be in-flight at IB")
            return None

        order_id = result['order_id']
        perm_id = result.get('perm_id', 0)

        # Register for callback routing
        self._entry_orders[order_id] = {
            'algo_id': algo_id, 'direction': direction,
            'stop_price': stop_price, 'tp_price': tp_price,
            'shares': shares, 'signal_type': signal_type,
            'stop_pct': stop_pct, 'tp_pct': tp_pct,
        }

        # Step 3: Insert DB row with pending status
        try:
            now_et = _now_eastern()
            trade_id = self.db.open_trade(
                source='ib', algo_id=algo_id, symbol='TSLA',
                direction=direction, entry_time=now_et,
                entry_price=entry_price or self._state.tsla_price,
                shares=shares, stop_price=stop_price, tp_price=tp_price,
                confidence=confidence, signal_type=signal_type,
                trail_width=trail_width, ou_half_life=ou_half_life,
                el_flagged=el_flagged, trail_width_mult=trail_width_mult,
                ib_entry_order_id=order_id, ib_perm_id=perm_id,
                ib_fill_status='pending',
                filled_shares=0, open_shares=0,
            )
        except Exception as e:
            logger.error("DB insert failed for entry order %d: %s", order_id, e)
            self._handle_entry_db_failure(order_id, stop_price, algo_id,
                                          direction, shares)
            return None

        # Step 4: Drain any early-arriving fills
        self._drain_pending_entry(order_id, trade_id)

        # Update context with trade_id
        self._entry_orders[order_id]['trade_id'] = trade_id

        logger.info("Entry order placed: order_id=%d, trade_id=%d, algo=%s, "
                     "%s %d shares", order_id, trade_id, algo_id, direction, shares)

        # Fire trade alert (UI card + Telegram)
        self._fire_trade_alert(
            'entry', algo_id, direction, shares,
            self._state.tsla_price, signal_type=signal_type,
            stop_price=stop_price, tp_price=tp_price)

        self._state.positions_version += 1
        self._state.trades_version += 1
        return trade_id

    def _handle_entry_db_failure(self, order_id: int, stop_price: float,
                                 algo_id: str, direction: str, shares: int):
        """Handle DB insert failure: cancel order, place emergency stop if filled."""
        # Cancel the IB order
        cancel_ok = self.ib.cancel_order(order_id)

        # Check if any fills arrived
        with self._buffer_lock:
            buffered = list(self._pending_fills.get(order_id, []))

        total_filled = sum(f.shares for f in buffered)

        if total_filled > 0 or not cancel_ok:
            self._set_degraded(f"Entry DB write failed for order {order_id}, "
                               f"filled={total_filled}, cancel_ok={cancel_ok}")
            # Seed failed_orders for late-fill tracking
            self._failed_orders[order_id] = FailedOrderContext(
                trade_id=None, stop_price=stop_price,
                algo_id=algo_id, direction=direction,
                shares=shares, filled_shares=total_filled,
            )

            if total_filled > 0:
                # Place emergency protective stop
                self._place_emergency_stop(
                    order_id, stop_price, total_filled, direction)

    def _place_emergency_stop(self, entry_order_id: int, stop_price: float,
                              qty: int, direction: str):
        """Place an emergency stop for untracked broker exposure."""
        close_action = 'SELL' if direction == 'long' else 'BUY'
        order_ref = f'emstop:{entry_order_id}:{stop_price:.2f}'

        result = self.ib.place_order(
            'TSLA', close_action, qty, 'STP', price=stop_price,
            tif='GTC', order_ref=order_ref, outside_rth=True)

        if 'error' in result:
            logger.critical("EMERGENCY STOP FAILED for entry %d: %s — "
                            "UNPROTECTED EXPOSURE of %d shares",
                            entry_order_id, result['error'], qty)
            return

        stop_order_id = result['order_id']
        stop_perm_id = result.get('perm_id', 0)
        self._emergency_stops[entry_order_id] = StopInfo(
            stop_order_id=stop_order_id, stop_perm_id=stop_perm_id,
            trade_id=None, stop_price=stop_price, qty=qty)
        logger.warning("Emergency stop placed: order_id=%d for entry %d, "
                        "%d shares @ $%.2f", stop_order_id, entry_order_id,
                        qty, stop_price)

    # ── Exit Flow ───────────────────────────────────────────────────

    def place_exit(self, trade_id: int, exit_reason: str,
                   exit_price: float = 0.0) -> bool:
        """Place an IB exit order with two-phase commit.

        Returns True if exit order placed, False otherwise.
        """
        trade = self._get_trade(trade_id)
        if not trade:
            logger.error("place_exit: trade %d not found", trade_id)
            return False

        # Check exit_pending suppression — verify order is actually alive at IB
        exit_oid = trade.get('ib_exit_order_id')
        if exit_oid:
            # Check if the exit order is still active at IB
            order_alive = False
            try:
                for bt in self.ib.ib.trades():
                    if bt.order.permId == trade.get('ib_exit_perm_id', 0) or \
                       bt.order.orderId == exit_oid:
                        if bt.orderStatus.status in ('PreSubmitted', 'Submitted',
                                                      'PendingSubmit'):
                            order_alive = True
                        break
            except Exception as e:
                logger.warning("place_exit: could not verify exit order %d: %s — "
                               "assuming alive", exit_oid, e)
                order_alive = True

            if order_alive:
                logger.info("Exit already pending for trade %d (order %d)",
                            trade_id, exit_oid)
                return False
            else:
                # Exit order is dead (Cancelled/Filled/Inactive) — clear stale DB ref
                logger.warning("Exit order %d for trade %d is dead at IB — clearing "
                               "stale DB ref and proceeding", exit_oid, trade_id)
                try:
                    self.db.update_trade_state(trade_id, ib_exit_order_id=None,
                                               ib_exit_perm_id=None)
                except Exception as e:
                    logger.error("Failed to clear stale exit order for trade %d: %s",
                                 trade_id, e)
                    return False

        if self._exit_in_progress.get(trade_id):
            logger.info("Exit already in progress for trade %d", trade_id)
            return False

        direction = trade.get('direction', 'long')
        close_action = 'SELL' if direction == 'long' else 'BUY'
        effective_open = self._effective_open_shares(trade)

        if effective_open <= 0:
            logger.warning("place_exit: trade %d has no open shares", trade_id)
            return False

        # Step 2: If partial entry, cancel remaining entry first
        if trade.get('ib_fill_status') == 'partial' and trade.get('ib_entry_order_id'):
            logger.info("Cancelling partial entry order %d before exit",
                        trade['ib_entry_order_id'])
            self.ib.cancel_order(trade['ib_entry_order_id'])
            # Promote to filled
            filled = trade.get('filled_shares', 0)
            self.db.update_trade_state(trade_id,
                                       shares=filled, ib_fill_status='filled',
                                       open_shares=filled)
            effective_open = filled

        # Step 5: Clear stop before placing exit (one-active-close-side rule)
        stop_order_id = trade.get('ib_stop_order_id')
        if stop_order_id:
            # NULL the stop_id in DB BEFORE cancelling
            try:
                self.db.update_trade_state(trade_id, ib_stop_order_id=None)
            except Exception as e:
                logger.error("Failed to NULL stop_id for trade %d: %s — "
                             "aborting exit", trade_id, e)
                self._set_degraded(f"Stop NULL write failed for trade {trade_id}")
                return False

            self._exit_in_progress[trade_id] = True
            cancel_ok = self.ib.cancel_order(stop_order_id)

            if cancel_ok:
                # Wait up to 3s for terminal status (Cancelled or Filled)
                import time as _time
                for _ in range(30):  # 30 × 0.1s = 3s
                    _time.sleep(0.1)
                    # Check if stop filled while we waited (DB close applied by callback)
                    trade_now = self._get_trade(trade_id)
                    if trade_now and trade_now.get('exit_time'):
                        logger.warning("Stop %d filled during cancel wait for trade %d — "
                                       "exit already applied, skipping MKT exit",
                                       stop_order_id, trade_id)
                        self._exit_in_progress.pop(trade_id, None)
                        return True  # Trade is closed
                    # Check broker-side status
                    with self.ib._order_lock:
                        bt = self.ib._trades.get(stop_order_id)
                        if bt and bt.orderStatus.status in ('Cancelled', 'ApiCancelled'):
                            break  # Safe to proceed with MKT exit
                        if bt and bt.orderStatus.status == 'Filled':
                            logger.warning("Stop %d FILLED during cancel for trade %d — "
                                           "skipping MKT exit",
                                           stop_order_id, trade_id)
                            self._exit_in_progress.pop(trade_id, None)
                            return True
                else:
                    logger.warning("Stop cancel timeout for trade %d (stop=%d) — "
                                   "proceeding cautiously", trade_id, stop_order_id)

        # Step 3: Place closing order
        order_ref = f'exit:{trade_id}'
        model_code = trade.get('algo_id') if self._state.fa_supported else None

        result = self.ib.place_order(
            'TSLA', close_action, effective_open, 'MKT',
            tif='GTC', outside_rth=True,
            order_ref=order_ref, model_code=model_code)

        if 'error' in result:
            logger.error("Exit order failed for trade %d: %s",
                         trade_id, result['error'])
            # Re-arm protective stop
            self._place_protective_stop(trade_id, trade.get('stop_price', 0),
                                        effective_open, direction)
            self._exit_in_progress.pop(trade_id, None)
            return False

        exit_order_id = result['order_id']
        exit_perm_id = result.get('perm_id', 0)

        # Register for callback routing
        self._exit_orders[exit_order_id] = {
            'trade_id': trade_id, 'exit_reason': exit_reason}

        # Step 4: Update DB with exit order info
        try:
            self.db.update_trade_state(trade_id,
                                       ib_exit_order_id=exit_order_id,
                                       ib_exit_perm_id=exit_perm_id)
        except Exception as e:
            logger.error("DB update failed for exit order %d (trade %d): %s",
                         exit_order_id, trade_id, e)
            # Cancel exit, re-arm stop
            self.ib.cancel_order(exit_order_id)
            self._failed_exit_orders[exit_order_id] = FailedExitContext(
                trade_id=trade_id, open_shares_at_failure=effective_open)
            self._set_degraded(f"Exit DB write failed for trade {trade_id}")
            self._place_protective_stop(trade_id, trade.get('stop_price', 0),
                                        effective_open, direction)
            self._exit_in_progress.pop(trade_id, None)
            return False

        # Drain any early exit fills
        self._drain_pending_exit(exit_order_id, trade_id, exit_reason)

        logger.info("Exit order placed: order_id=%d, trade_id=%d, "
                     "%s %d shares, reason=%s",
                     exit_order_id, trade_id, close_action,
                     effective_open, exit_reason)
        return True

    # ── Protective Stop ─────────────────────────────────────────────

    def _place_protective_stop(self, trade_id: int, stop_price: float,
                               qty: int, direction: str) -> Optional[int]:
        """Place or re-arm a protective stop for a trade.

        Returns stop order_id on success, None on failure.
        """
        if stop_price <= 0 or qty <= 0:
            return None

        # Round to tick size ($0.01 for TSLA) — IB Error 110 rejects
        # prices that don't conform to minimum price variation
        stop_price = round(stop_price, 2)

        close_action = 'SELL' if direction == 'long' else 'BUY'
        order_ref = f'stop:{trade_id}'

        result = self.ib.place_order(
            'TSLA', close_action, qty, 'STP', price=stop_price,
            tif='GTC', order_ref=order_ref, outside_rth=True)

        if 'error' in result:
            error_msg = result['error']
            if 'Timeout' in str(error_msg) or 'in-flight' in str(error_msg):
                # Order may be live at IB despite timeout — do NOT re-arm.
                # Recovery on next restart will find it via orderRef.
                logger.warning("Stop placement TIMED OUT for trade %d — "
                               "order may be in-flight, NOT re-arming "
                               "(recovery will reconcile)", trade_id)
                return -1  # Sentinel: not None (don't re-arm), not a real ID
            logger.error("Protective stop failed for trade %d: %s",
                         trade_id, result['error'])
            self._set_degraded(f"Stop placement failed for trade {trade_id}")
            return None

        stop_order_id = result['order_id']
        stop_perm_id = result.get('perm_id', 0)

        # Persist stop order info to DB
        try:
            self.db.update_trade_state(trade_id,
                                       ib_stop_order_id=stop_order_id,
                                       ib_stop_perm_id=stop_perm_id)
        except Exception as e:
            logger.error("Failed to persist stop IDs for trade %d: %s",
                         trade_id, e)
            self._set_degraded(f"Stop ID persistence failed for trade {trade_id}")

        # Register for callback routing
        self._stop_orders[stop_order_id] = {'trade_id': trade_id}

        logger.info("Protective stop placed: order_id=%d, trade %d, "
                     "%d shares @ $%.2f", stop_order_id, trade_id,
                     qty, stop_price)
        return stop_order_id

    def place_or_resize_stop(self, trade_id: int, stop_price: float,
                             qty: int, direction: str):
        """Place or resize a protective stop, serialized per trade_id.

        Uses _stop_lock with dirty flag to ensure at most one placement
        is in flight per trade.
        """
        lock = self._stop_locks.setdefault(trade_id, threading.Lock())

        while True:
            acquired = lock.acquire(blocking=False)
            if not acquired:
                # Another thread is handling this trade's stop
                self._stop_dirty[trade_id] = True
                return

            try:
                self._stop_dirty[trade_id] = False

                trade = self._get_trade(trade_id)
                if not trade:
                    return

                existing_stop = trade.get('ib_stop_order_id')
                if existing_stop:
                    # Modify existing stop
                    # TODO: Use ib.modifyOrder() instead of cancel+replace
                    # For now, cancel and re-place
                    self.ib.cancel_order(existing_stop)

                self._place_protective_stop(trade_id, stop_price, qty, direction)
            finally:
                lock.release()

            # Check dirty flag — if set, loop back
            if not self._stop_dirty.get(trade_id, False):
                break

    def modify_trailing_stop(self, trade_id: int, new_stop_price: float):
        """Update the resting IB stop to a new price level (trailing ratchet)."""
        # Guard: skip if exit is already in progress (stop may have been cancelled)
        if self._exit_in_progress.get(trade_id):
            return

        trade = self._get_trade(trade_id)
        if not trade:
            return

        stop_order_id = trade.get('ib_stop_order_id')
        if not stop_order_id:
            return

        old_stop = trade.get('stop_price', 0)
        direction = trade.get('direction', 'long')
        effective_open = self._effective_open_shares(trade)

        if effective_open <= 0:
            return

        # Cancel old stop and place new one at updated price
        self.ib.cancel_order(stop_order_id)
        new_stop_id = self._place_protective_stop(
            trade_id, new_stop_price, effective_open, direction)

        if new_stop_id is None:
            # Real failure — revert DB stop_price to match what IB still has
            logger.error("Stop modification failed for trade %d — "
                         "reverting to $%.2f", trade_id, old_stop)
            try:
                self.db.update_trade_state(trade_id, stop_price=old_stop)
            except Exception as e:
                logger.error("CRITICAL: Failed to revert stop price for trade %d: %s",
                             trade_id, e)
            self._set_degraded(f"Stop modification failed for trade {trade_id}")
        elif new_stop_id == -1:
            # Timeout — order may be in-flight, recovery will reconcile
            logger.warning("Stop modification timed out for trade %d — "
                           "order may be in-flight at IB", trade_id)

    # ── Fill Callbacks ──────────────────────────────────────────────

    def on_exec_details(self, trade, fill):
        """Handle execDetailsEvent from IB.

        Called from IB event loop thread. Must be wired AFTER recovery + seeding.
        Routes to entry/exit/stop fill handlers based on order registration.
        """
        exec_obj = fill.execution
        exec_id = exec_obj.execId

        # Deduplicate
        if exec_id in self.seen_exec_ids:
            return
        self.seen_exec_ids.add(exec_id)

        order_id = exec_obj.orderId
        fill_shares = int(exec_obj.shares)
        fill_price = float(exec_obj.price)

        # Convert broker timestamp to ET
        fill_time = _now_eastern()
        broker_time = getattr(exec_obj, 'time', None)
        if broker_time:
            try:
                if hasattr(broker_time, 'isoformat'):
                    fill_time = broker_time.astimezone(ET).isoformat()
                else:
                    fill_time = str(broker_time)
            except Exception as e:
                logger.warning("Failed to parse broker fill time: %s", e)

        perm_id = int(getattr(exec_obj, 'permId', 0))

        fill_data = FillData(
            exec_id=exec_id, shares=fill_shares,
            price=fill_price, time=fill_time, order_id=order_id)

        # Route based on registration
        if order_id in self._entry_orders:
            # Update ib_perm_id on first fill (was 0 at placement time)
            if perm_id > 0:
                ctx = self._entry_orders.get(order_id, {})
                trade_id = ctx.get('trade_id')
                if trade_id:
                    try:
                        self.db.update_trade_state(trade_id, ib_perm_id=perm_id)
                    except Exception:
                        pass  # non-critical, just for linkage
            self._on_entry_fill(order_id, fill_data)
        elif order_id in self._exit_orders:
            self._on_exit_fill(order_id, fill_data)
        elif order_id in self._stop_orders:
            self._on_stop_fill(order_id, fill_data)
        elif order_id in self._failed_orders:
            self._on_failed_entry_fill(order_id, fill_data)
        else:
            # Unknown order — could be foreign or from before restart
            logger.warning("Fill for unregistered order %d: %d shares @ $%.2f "
                           "(exec_id=%s). NOT auto-cancelling.",
                           order_id, fill_shares, fill_price, exec_id)

    def _on_entry_fill(self, order_id: int, fill: FillData):
        """Handle a fill on an entry order."""
        ctx = self._entry_orders.get(order_id, {})
        trade_id = ctx.get('trade_id')

        if trade_id is None:
            # DB row doesn't exist yet — buffer
            with self._buffer_lock:
                self._pending_fills.setdefault(order_id, []).append(fill)
            logger.info("Entry fill buffered (pre-DB): order %d, %d @ $%.2f",
                        order_id, fill.shares, fill.price)
            return

        # Apply fill to DB
        self._apply_entry_fill(trade_id, fill, ctx)

    def _apply_entry_fill(self, trade_id: int, fill: FillData, ctx: dict):
        """Apply an entry fill to the DB and manage protective stop."""
        trade = self._get_trade(trade_id)
        if not trade:
            logger.error("Trade %d not found for entry fill", trade_id)
            return

        # Check if trade is already closed (late fill after stop-close)
        if trade.get('exit_time'):
            logger.error("LATE ENTRY FILL on closed trade %d: %d shares @ $%.2f — "
                         "placing emergency stop", trade_id, fill.shares, fill.price)
            direction = trade.get('direction', 'long')
            stop_price = trade.get('stop_price', 0)
            self._place_emergency_stop(
                fill.order_id, stop_price or fill.price * 0.98,
                fill.shares, direction)
            self._set_degraded(f"Late entry fill on closed trade {trade_id}")
            return

        old_filled = trade.get('filled_shares', 0)
        old_avg = trade.get('avg_fill_price', 0) or trade.get('entry_price', 0)
        total_shares = trade.get('shares', 0)

        new_filled = old_filled + fill.shares
        # VWAP
        if old_filled > 0 and old_avg > 0:
            new_avg = (old_avg * old_filled + fill.price * fill.shares) / new_filled
        else:
            new_avg = fill.price

        updates = {
            'filled_shares': new_filled,
            'avg_fill_price': new_avg,
            'entry_price': new_avg,
            'open_shares': new_filled,  # open = filled (no exits yet)
        }

        # First fill: update entry_time to broker timestamp + recalculate stop/TP
        if old_filled == 0:
            updates['entry_time'] = fill.time
            updates['best_price'] = fill.price
            updates['worst_price'] = fill.price
            # Recalculate stop/TP from actual fill price (not estimated)
            # This corrects for MKT order slippage
            s_pct = ctx.get('stop_pct', 0)
            t_pct = ctx.get('tp_pct', 0)
            direction = trade.get('direction', ctx.get('direction', 'long'))
            if s_pct > 0:
                if direction == 'long':
                    updates['stop_price'] = round(fill.price * (1 - s_pct), 2)
                else:
                    updates['stop_price'] = round(fill.price * (1 + s_pct), 2)
            if t_pct > 0:
                if direction == 'long':
                    updates['tp_price'] = round(fill.price * (1 + t_pct), 2)
                else:
                    updates['tp_price'] = round(fill.price * (1 - t_pct), 2)

        # Update fill status
        if new_filled >= total_shares:
            updates['ib_fill_status'] = 'filled'
        else:
            updates['ib_fill_status'] = 'partial'

        try:
            self.db.update_trade_state(trade_id, **updates)
        except Exception as e:
            logger.error("CRITICAL: DB update failed for entry fill on trade %d: %s",
                         trade_id, e)
            self._set_degraded(f"Entry fill DB write failed for trade {trade_id}")
            return

        logger.info("Entry fill applied: trade %d, %d/%d shares @ $%.2f (avg $%.2f)",
                     trade_id, new_filled, total_shares, fill.price, new_avg)

        # Place/resize protective stop on EVERY fill
        # Use recalculated stop from updates (actual fill price) if available,
        # otherwise fall back to DB value (which may be stale pre-update)
        direction = trade.get('direction', ctx.get('direction', 'long'))
        stop_price = updates.get('stop_price',
                                  trade.get('stop_price', ctx.get('stop_price', 0)))
        self.place_or_resize_stop(trade_id, stop_price, new_filled, direction)

        self._state.positions_version += 1
        self._notify_live_engine(trade_id, fill.price, fill.shares, is_entry=True)

    def _on_exit_fill(self, order_id: int, fill: FillData):
        """Handle a fill on an exit order."""
        ctx = self._exit_orders.get(order_id, {})
        trade_id = ctx.get('trade_id')

        if trade_id is None:
            with self._buffer_lock:
                self._pending_exit_fills.setdefault(order_id, []).append(fill)
            logger.info("Exit fill buffered (pre-DB): order %d, %d @ $%.2f",
                        order_id, fill.shares, fill.price)
            return

        self._apply_exit_fill(trade_id, fill, ctx)

    def _apply_exit_fill(self, trade_id: int, fill: FillData, ctx: dict):
        """Apply an exit fill to the DB.

        Guards against double-close from stop-cancel race: if the stop fills
        between cancel_order() and exit MKT placement, both the stop fill and
        the exit fill arrive. The second one finds the trade already closed.
        """
        trade = self._get_trade(trade_id)
        if not trade:
            return

        # Guard: trade already closed (stop-cancel race — stop filled first)
        if trade.get('exit_time'):
            logger.error("DOUBLE-CLOSE RACE: exit fill on already-closed trade %d "
                         "(%d shares @ $%.2f, reason=%s). Placing counter-trade.",
                         trade_id, fill.shares, fill.price,
                         ctx.get('exit_reason', '?'))
            # We now have excess broker exposure. Place a counter-trade to flatten.
            direction = trade.get('direction', 'long')
            # The exit sold (long) or bought (short) — reverse it
            counter_action = 'BUY' if direction == 'long' else 'SELL'
            counter_ref = f'counter:{trade_id}'
            try:
                self.ib.place_order('TSLA', counter_action, fill.shares, 'MKT',
                                    tif='GTC', order_ref=counter_ref)
                logger.warning("Counter-trade placed: %s %d shares for double-close on trade %d",
                               counter_action, fill.shares, trade_id)
            except Exception as e:
                logger.critical("COUNTER-TRADE FAILED for trade %d: %s — "
                                "MANUAL INTERVENTION REQUIRED", trade_id, e)
            self._set_degraded(f"Double-close race on trade {trade_id}")
            return

        old_exit_filled = trade.get('exit_filled_shares', 0)
        old_exit_avg = trade.get('avg_exit_price', 0)
        filled_shares = trade.get('filled_shares', 0)

        new_exit_filled = old_exit_filled + fill.shares
        # VWAP for exit price
        if old_exit_filled > 0 and old_exit_avg > 0:
            new_exit_avg = (old_exit_avg * old_exit_filled +
                            fill.price * fill.shares) / new_exit_filled
        else:
            new_exit_avg = fill.price

        new_open = filled_shares - new_exit_filled
        exit_reason = ctx.get('exit_reason', 'exit')

        try:
            self.db.update_trade_state(trade_id,
                                       exit_filled_shares=new_exit_filled,
                                       avg_exit_price=new_exit_avg,
                                       open_shares=max(new_open, 0))
        except Exception as e:
            logger.error("CRITICAL: DB update failed for exit fill on trade %d: %s",
                         trade_id, e)
            self._set_degraded(f"Exit fill DB write failed for trade {trade_id}")
            return

        logger.info("Exit fill applied: trade %d, %d/%d shares exited @ $%.2f",
                     trade_id, new_exit_filled, filled_shares, fill.price)

        # Check if fully exited
        if new_exit_filled >= filled_shares:
            try:
                self.db.close_trade(
                    trade_id, exit_time=fill.time,
                    exit_price=new_exit_avg, exit_reason=exit_reason,
                    effective_filled_shares=filled_shares,
                    effective_avg_fill_price=trade.get('avg_fill_price',
                                                       trade.get('entry_price', 0)),
                    effective_avg_exit_price=new_exit_avg)
                self._exit_in_progress.pop(trade_id, None)
                self._state.trades_version += 1
                logger.info("Trade %d closed: %s @ $%.2f", trade_id,
                             exit_reason, new_exit_avg)

                # Fire exit alert (UI card + Telegram)
                entry_price = trade.get('avg_fill_price',
                                         trade.get('entry_price', 0))
                direction = trade.get('direction', 'long')
                pnl = (new_exit_avg - entry_price) * filled_shares
                if direction == 'short':
                    pnl = -pnl
                self._fire_trade_alert(
                    'exit', trade.get('algo_id', ''), direction,
                    filled_shares, new_exit_avg,
                    exit_reason=exit_reason, pnl=pnl,
                    entry_price=entry_price)
            except Exception as e:
                logger.error("CRITICAL: close_trade failed for trade %d: %s",
                             trade_id, e)
                self._set_degraded(f"close_trade DB write failed for trade {trade_id}")

        self._state.positions_version += 1
        self._notify_live_engine(trade_id, fill.price, fill.shares, is_entry=False)

    def _on_stop_fill(self, order_id: int, fill: FillData):
        """Handle a fill on a protective stop (stop fill = exit)."""
        ctx = self._stop_orders.get(order_id, {})
        trade_id = ctx.get('trade_id')
        if not trade_id:
            logger.error("Stop fill for unknown trade: order %d", order_id)
            return

        # Route through exit fill logic with exit_reason='sl'
        exit_ctx = {'trade_id': trade_id, 'exit_reason': 'sl'}
        self._apply_exit_fill(trade_id, fill, exit_ctx)

    def _on_failed_entry_fill(self, order_id: int, fill: FillData):
        """Handle a fill on a failed-DB entry order."""
        ctx = self._failed_orders.get(order_id)
        if not ctx:
            return

        ctx.filled_shares += fill.shares
        self._set_degraded(f"Late fill on failed entry order {order_id}: "
                           f"{fill.shares} shares")

        # Resize or place emergency stop
        em = self._emergency_stops.get(order_id)
        if em:
            # Resize
            self.ib.cancel_order(em.stop_order_id)
            self._place_emergency_stop(
                order_id, ctx.stop_price, ctx.filled_shares, ctx.direction)
        else:
            # Place new
            self._place_emergency_stop(
                order_id, ctx.stop_price, ctx.filled_shares, ctx.direction)

    # ── Status Callbacks ────────────────────────────────────────────

    def on_order_status(self, trade):
        """Handle statusEvent from IB.

        Used for terminal status detection (Cancelled/Rejected),
        stop runtime monitoring, and permId backfill (when permId was 0
        at placement time). NOT used for fill tracking (use execDetailsEvent).
        """
        order_id = trade.order.orderId
        status = trade.orderStatus.status
        perm_id = trade.order.permId

        # Backfill permId if it was 0 at placement time
        if perm_id > 0 and order_id in self._entry_orders:
            ctx = self._entry_orders[order_id]
            trade_id = ctx.get('trade_id')
            if trade_id:
                try:
                    db_trade = self._get_trade(trade_id)
                    if db_trade and (db_trade.get('ib_perm_id') or 0) == 0:
                        self.db.update_trade_state(trade_id, ib_perm_id=perm_id)
                        logger.info("Backfilled permId=%d for trade %d (order %d)",
                                    perm_id, trade_id, order_id)
                except Exception:
                    pass  # Non-critical — on_exec_details also updates permId

        if status not in ('Cancelled', 'Inactive', 'ApiCancelled', 'Rejected'):
            return  # Only care about terminal statuses below

        # Check if this is an entry order
        if order_id in self._entry_orders:
            self._on_entry_terminal(order_id, status)
        elif order_id in self._exit_orders:
            self._on_exit_terminal(order_id, status)
        elif order_id in self._stop_orders:
            self._on_stop_terminal(order_id, status)

    def _on_entry_terminal(self, order_id: int, status: str):
        """Handle terminal status for an entry order."""
        ctx = self._entry_orders.get(order_id, {})
        trade_id = ctx.get('trade_id')

        if trade_id is None:
            # DB row doesn't exist yet — buffer terminal
            with self._buffer_lock:
                self._pending_terminal[order_id] = TerminalStatus(
                    status=status, timestamp=_now_eastern())
            return

        trade = self._get_trade(trade_id)
        if not trade:
            return

        filled = trade.get('filled_shares', 0)

        if filled == 0:
            # No fills — delete the pending row
            try:
                self.db.delete_trade(trade_id)
                logger.info("Entry %d cancelled with 0 fills — deleted trade %d",
                             order_id, trade_id)
            except Exception as e:
                logger.error("Failed to delete cancelled trade %d: %s",
                             trade_id, e)
        else:
            # Partial fill then cancel — promote to filled
            try:
                self.db.update_trade_state(trade_id,
                                           ib_fill_status='filled',
                                           shares=filled,
                                           open_shares=filled)
                logger.info("Entry %d cancelled with %d fills — "
                             "promoted trade %d to filled",
                             order_id, filled, trade_id)
            except Exception as e:
                logger.error("Failed to promote trade %d: %s", trade_id, e)

        # Tombstone
        self._terminal_orders[order_id] = TerminalInfo(
            status=status, timestamp=_now_eastern(),
            filled_shares_at_terminal=filled, context=ctx)

    def _on_exit_terminal(self, order_id: int, status: str):
        """Handle terminal status for an exit order (rejected/cancelled)."""
        ctx = self._exit_orders.get(order_id, {})
        trade_id = ctx.get('trade_id')
        if not trade_id:
            return

        trade = self._get_trade(trade_id)
        if not trade:
            return

        # Idempotency guard
        if trade.get('ib_exit_order_id') != order_id:
            logger.info("Stale exit terminal for order %d (trade %d has order %s)",
                        order_id, trade_id, trade.get('ib_exit_order_id'))
            return

        # Clear exit order ID so next cycle can retry
        try:
            self.db.update_trade_state(trade_id, ib_exit_order_id=None,
                                       ib_exit_perm_id=None)
        except Exception as e:
            logger.error("Failed to clear exit order ID for trade %d: %s",
                         trade_id, e)

        self._exit_in_progress.pop(trade_id, None)

        # Remove from routing dict so duplicate callbacks are ignored
        self._exit_orders.pop(order_id, None)

        # Re-arm protective stop
        effective_open = self._effective_open_shares(trade)
        if effective_open > 0:
            direction = trade.get('direction', 'long')
            stop_price = trade.get('stop_price', 0)
            new_stop = self._place_protective_stop(trade_id, stop_price,
                                                   effective_open, direction)
            if new_stop is None:
                self._set_degraded(
                    f"Stop re-arm after exit cancel failed for trade {trade_id}")
            # new_stop == -1 means timeout (in-flight) — recovery will reconcile
            logger.warning("Exit %d rejected/cancelled for trade %d — "
                           "stop re-armed (stop_id=%s)", order_id, trade_id,
                           new_stop)

    def _on_stop_terminal(self, order_id: int, status: str):
        """Handle terminal status for a protective stop (unexpected cancel)."""
        ctx = self._stop_orders.get(order_id, {})
        trade_id = ctx.get('trade_id')
        if not trade_id:
            return

        # Guard 1: intentional cancel (exit in progress)?
        if self._exit_in_progress.get(trade_id):
            return

        trade = self._get_trade(trade_id)
        if not trade:
            return

        # Guard 2: idempotency — is this still the current stop?
        if trade.get('ib_stop_order_id') != order_id:
            return

        # Guard 3: prevent re-arm storm — NULL the DB stop_id BEFORE re-arming
        # so duplicate callbacks for the same order_id are blocked by Guard 2
        try:
            self.db.update_trade_state(trade_id, ib_stop_order_id=None)
        except Exception as e:
            logger.error("Failed to NULL stop_id for trade %d during re-arm: %s",
                         trade_id, e)
            return

        # Remove from routing dict so further callbacks are ignored
        self._stop_orders.pop(order_id, None)

        # Unexpected stop cancellation — re-arm
        effective_open = self._effective_open_shares(trade)
        if effective_open > 0:
            direction = trade.get('direction', 'long')
            stop_price = trade.get('stop_price', 0)
            logger.warning("Stop %d unexpectedly cancelled for trade %d — re-arming",
                           order_id, trade_id)
            new_stop = self._place_protective_stop(
                trade_id, stop_price, effective_open, direction)
            if new_stop is None:
                self._set_degraded(f"Stop re-arm failed for trade {trade_id}")
            # new_stop == -1 means timeout (in-flight) — don't set degraded

    # ── Drain Buffers ───────────────────────────────────────────────

    def _drain_pending_entry(self, order_id: int, trade_id: int):
        """Drain buffered entry fills and terminal statuses after DB row exists."""
        ctx = self._entry_orders.get(order_id, {})

        # Drain fills first
        with self._buffer_lock:
            fills = self._pending_fills.pop(order_id, [])
            terminal = self._pending_terminal.pop(order_id, None)

        for fill in fills:
            self._apply_entry_fill(trade_id, fill, ctx)

        # Then drain terminal
        if terminal:
            self._on_entry_terminal(order_id, terminal.status)

    def _drain_pending_exit(self, exit_order_id: int, trade_id: int,
                            exit_reason: str):
        """Drain buffered exit fills after DB row is updated."""
        ctx = {'trade_id': trade_id, 'exit_reason': exit_reason}

        with self._buffer_lock:
            fills = self._pending_exit_fills.pop(exit_order_id, [])
            terminal = self._pending_exit_terminal.pop(exit_order_id, None)

        for fill in fills:
            self._apply_exit_fill(trade_id, fill, ctx)

        if terminal:
            self._on_exit_terminal(exit_order_id, terminal.status)

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_trade(self, trade_id: int) -> Optional[dict]:
        """Get trade from DB by ID."""
        try:
            trades = self.db.get_open_trades(source='ib', include_pending=True)
            for t in trades:
                if t['id'] == trade_id:
                    return t
            # Also check closed trades
            closed = self.db.get_closed_trades(source='ib', limit=100)
            for t in closed:
                if t['id'] == trade_id:
                    return t
        except Exception as e:
            logger.error("Failed to lookup trade %d: %s", trade_id, e)
        return None

    def _effective_open_shares(self, trade: dict) -> int:
        """Compute effective open shares including unapplied fills.

        Uses filled_shares only (not shares) — a pending entry with 0 fills
        has 0 effective exposure, not the requested share count.
        """
        filled = trade.get('filled_shares', 0) or 0
        exit_filled = trade.get('exit_filled_shares', 0) or 0
        return max(filled - exit_filled, 0)
