"""
IB Recovery — Startup recovery for in-flight orders, unlinked orders, and reconciliation.

Called during startup AFTER IB connects but BEFORE execDetailsEvent is wired.
Recovery applies historical fills from reqExecutions/reqCompletedOrders,
then seeds seen_exec_ids to prevent double-counting on reconnect.
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
ET = ZoneInfo('US/Eastern')


def _broker_time_to_et(t) -> str:
    """Convert a broker timestamp to US/Eastern ISO 8601."""
    if not t:
        return datetime.now(ET).isoformat()
    try:
        if hasattr(t, 'astimezone'):
            return t.astimezone(ET).isoformat()
        return str(t)
    except Exception:
        return datetime.now(ET).isoformat()


def scan_unlinked_orders(state):
    """Step 6c: Scan ALL broker orders for entries/exits not linked to DB rows.

    Catches the placeOrder -> crash -> restart window. Runs BEFORE recovery
    re-arms stops to avoid duplicates.
    """
    if not state.ib_client or not state.ib_client.is_connected():
        return

    ib = state.ib_client
    db = state.trade_db
    handler = state.ib_order_handler

    # Get all open orders from IB
    try:
        open_trades = ib.ib.openTrades()
    except Exception as e:
        logger.error("scan_unlinked_orders: failed to get open trades: %s", e)
        return

    # Also scan completed orders (catches fast fills during app downtime)
    completed_trades = []
    try:
        completed_trades = ib.ib.trades()  # includes open + recently completed
    except Exception as e:
        logger.warning("scan_unlinked_orders: failed to get completed trades: %s", e)

    # Merge: use all trades, dedup by orderId
    seen_order_ids = set()
    all_trades = []
    for t in list(open_trades) + list(completed_trades):
        oid = t.order.orderId
        if oid not in seen_order_ids:
            seen_order_ids.add(oid)
            all_trades.append(t)

    for trade in all_trades:
        order = trade.order
        order_ref = getattr(order, 'orderRef', '') or ''
        order_id = order.orderId
        perm_id = order.permId

        if not order_ref:
            continue  # Foreign order — log but don't touch

        if order_ref.startswith('entry:'):
            # Check if DB has a row with this order_id
            existing = db.get_trade_by_order_id(order_id, side='entry')
            if not existing:
                logger.warning("Unlinked entry order %d (ref=%s) — "
                               "checking fills", order_id, order_ref)
                # Parse orderRef: entry:{algo_id}:{direction}:{stop_price}:{tp_price}
                parts = order_ref.split(':')
                if len(parts) >= 5:
                    algo_id = parts[1]
                    direction = parts[2]
                    try:
                        stop_price = float(parts[3])
                    except (ValueError, IndexError) as e:
                        logger.error("Failed to parse stop_price from orderRef '%s': %s",
                                     order_ref, e)
                        stop_price = 0
                    try:
                        tp_price = float(parts[4])
                    except (ValueError, IndexError) as e:
                        logger.error("Failed to parse tp_price from orderRef '%s': %s",
                                     order_ref, e)
                        tp_price = 0

                    # Check if order has fills
                    filled = int(trade.orderStatus.filled or 0)
                    avg_price = float(trade.orderStatus.avgFillPrice or 0)

                    if filled > 0:
                        # Create DB row for the filled entry
                        logger.warning("Unlinked entry %d has %d fills @ $%.2f — "
                                       "creating DB row", order_id, filled, avg_price)
                        try:
                            fill_time = datetime.now(ET).isoformat()
                            if trade.fills:
                                first_exec = trade.fills[0].execution
                                fill_time = _broker_time_to_et(
                                    getattr(first_exec, 'time', None))

                            trade_id = db.open_trade(
                                source='ib', algo_id=algo_id, symbol='TSLA',
                                direction=direction, entry_time=fill_time,
                                entry_price=avg_price, shares=filled,
                                stop_price=stop_price, tp_price=tp_price,
                                confidence=0.5, signal_type='recovered',
                                ib_entry_order_id=order_id, ib_perm_id=perm_id,
                                ib_fill_status='filled',
                                filled_shares=filled, open_shares=filled,
                                avg_fill_price=avg_price,
                            )
                            logger.info("Created DB row %d for unlinked entry %d",
                                        trade_id, order_id)

                            # Place protective stop
                            if stop_price > 0:
                                handler.place_or_resize_stop(
                                    trade_id, stop_price, filled, direction)
                        except Exception as e:
                            logger.error("Failed to create DB row for unlinked "
                                         "entry %d: %s", order_id, e)
                            handler._set_degraded(
                                f"Unlinked entry {order_id} recovery failed")
                    else:
                        # No fills — order is pending, create pending DB row
                        try:
                            trade_id = db.open_trade(
                                source='ib', algo_id=algo_id, symbol='TSLA',
                                direction=direction,
                                entry_time=datetime.now(ET).isoformat(),
                                entry_price=0, shares=int(order.totalQuantity),
                                stop_price=stop_price, tp_price=tp_price,
                                confidence=0.5, signal_type='recovered',
                                ib_entry_order_id=order_id, ib_perm_id=perm_id,
                                ib_fill_status='pending',
                                filled_shares=0, open_shares=0,
                            )
                            handler._entry_orders[order_id] = {
                                'trade_id': trade_id, 'algo_id': algo_id,
                                'direction': direction, 'stop_price': stop_price,
                                'tp_price': tp_price,
                            }
                        except Exception as e:
                            logger.error("Failed to create pending row for "
                                         "unlinked entry %d: %s", order_id, e)
                            if handler and hasattr(handler, '_set_degraded'):
                                handler._set_degraded(
                                    f"Unlinked broker entry {order_id} is untracked")
                            else:
                                state.ib_degraded = True

        elif order_ref.startswith('exit:'):
            # Check if DB has the exit linked
            try:
                trade_id = int(order_ref.split(':')[1])
            except (ValueError, IndexError) as e:
                logger.error("Failed to parse trade_id from exit orderRef '%s': %s",
                             order_ref, e)
                continue
            # Register for fill tracking
            handler._exit_orders[order_id] = {
                'trade_id': trade_id, 'exit_reason': 'recovered'}

        elif order_ref.startswith('stop:') or order_ref.startswith('emstop:'):
            try:
                trade_id = int(order_ref.split(':')[1])
            except (ValueError, IndexError) as e:
                logger.error("Failed to parse trade_id from stop orderRef '%s': %s",
                             order_ref, e)
                continue
            handler._stop_orders[order_id] = {'trade_id': trade_id}

    logger.info("scan_unlinked_orders complete: processed %d trades (%d open, %d total)",
                len(all_trades), len(open_trades), len(all_trades))


def recover_inflight_orders(state):
    """Step 6d: Recover pending/partial entries + exits + stops.

    Queries DB for open IB trades and reconciles with broker state.
    Checks IB for fills that occurred while the app was down (e.g. stop
    triggered during a restart). execDetailsEvent is NOT wired yet —
    safe to apply historical fills.
    """
    if not state.ib_client or not state.ib_client.is_connected():
        return

    db = state.trade_db
    handler = state.ib_order_handler

    # Build lookup: orderId → IB Trade object (open + recently completed)
    all_broker_trades = {}
    try:
        for t in state.ib_client.ib.trades():
            all_broker_trades[t.order.orderId] = t
    except Exception as e:
        logger.error("Recovery: failed to get broker trades: %s — aborting recovery", e)
        if handler and hasattr(handler, '_set_degraded'):
            handler._set_degraded("Recovery aborted — cannot load broker trades")
        else:
            state.ib_degraded = True
        return

    # Get all open IB trades (including pending)
    open_trades = db.get_open_trades(source='ib', include_pending=True)
    logger.info("Recovery: %d open IB trades to check (broker has %d trades)",
                len(open_trades), len(all_broker_trades))

    now_et = datetime.now(ET).isoformat()

    for trade in open_trades:
        trade_id = trade['id']
        fill_status = trade.get('ib_fill_status', 'filled')
        direction = trade.get('direction', 'long')
        open_shares = trade.get('open_shares') or 0

        # ── Check pending entries against broker state ──
        entry_oid = trade.get('ib_entry_order_id')
        if entry_oid and fill_status == 'pending' and entry_oid in all_broker_trades:
            bt = all_broker_trades[entry_oid]
            broker_status = bt.orderStatus.status
            broker_filled = int(bt.orderStatus.filled or 0)
            broker_avg = float(bt.orderStatus.avgFillPrice or 0)

            if broker_status in ('Cancelled', 'Inactive', 'ApiCancelled') and broker_filled == 0:
                # Entry was cancelled with no fills — delete the pending row
                logger.warning("Recovery: pending entry %d is %s with 0 fills — "
                               "deleting trade %d", entry_oid, broker_status, trade_id)
                try:
                    db.delete_trade(trade_id)
                except Exception as e:
                    logger.error("Recovery: failed to delete cancelled trade %d: %s",
                                 trade_id, e)
                continue

            if broker_filled > 0 and fill_status == 'pending':
                # Entry filled while app was down — apply to DB
                logger.warning("Recovery: pending entry %d filled %d @ $%.2f — "
                               "updating trade %d",
                               entry_oid, broker_filled, broker_avg, trade_id)
                try:
                    fill_time = now_et
                    if bt.fills:
                        first_exec = bt.fills[0].execution
                        fill_time = _broker_time_to_et(
                            getattr(first_exec, 'time', None))
                    db.update_trade_state(
                        trade_id,
                        entry_price=broker_avg, avg_fill_price=broker_avg,
                        filled_shares=broker_filled, open_shares=broker_filled,
                        ib_fill_status='filled', entry_time=fill_time,
                    )
                    # Update local state for subsequent stop/exit checks
                    fill_status = 'filled'
                    open_shares = broker_filled
                except Exception as e:
                    logger.error("Recovery: failed to apply entry fills for trade %d: %s",
                                 trade_id, e)

        # ── Check stops against broker state ──
        stop_oid = trade.get('ib_stop_order_id')
        if stop_oid and stop_oid not in all_broker_trades:
            # Order not in IB at all (old session, different clientId) — stale
            logger.warning("Recovery: stop %d for trade %d not found at IB — "
                           "clearing stale DB ref", stop_oid, trade_id)
            try:
                db.update_trade_state(trade_id, ib_stop_order_id=None)
            except Exception as e:
                logger.error("Recovery: failed to clear stop ref for trade %d: %s",
                             trade_id, e)
            stop_oid = None  # Force re-arm below

        if stop_oid and stop_oid in all_broker_trades:
            bt = all_broker_trades[stop_oid]
            broker_status = bt.orderStatus.status
            broker_filled = int(bt.orderStatus.filled or 0)

            if broker_filled > 0:
                # Stop fired while app was down — close the trade
                exit_price = float(bt.orderStatus.avgFillPrice or 0)
                exit_time = now_et
                if bt.fills:
                    last_exec = bt.fills[-1].execution
                    exit_time = _broker_time_to_et(
                        getattr(last_exec, 'time', None))
                logger.warning("Recovery: stop %d FILLED @ $%.2f for trade %d — "
                               "closing in DB", stop_oid, exit_price, trade_id)
                try:
                    db.close_trade(trade_id, exit_time=exit_time,
                                   exit_price=exit_price,
                                   exit_reason='stop_filled_during_restart')
                except Exception as e:
                    logger.error("Recovery: failed to close trade %d after stop fill: %s",
                                 trade_id, e)
                continue  # Trade is closed — don't re-register

            if broker_status in ('Cancelled', 'ApiCancelled') and broker_filled == 0:
                # Stop was cancelled (e.g. by TWS user) — clear and re-arm below
                logger.warning("Recovery: stop %d cancelled for trade %d — "
                               "will re-arm", stop_oid, trade_id)
                try:
                    db.update_trade_state(trade_id, ib_stop_order_id=None)
                except Exception as e:
                    logger.error("Recovery: failed to clear stop_id for trade %d: %s",
                                 trade_id, e)
                stop_oid = None  # Force re-arm below

        # ── Check exits against broker state ──
        exit_oid = trade.get('ib_exit_order_id')
        if exit_oid and exit_oid not in all_broker_trades:
            # Order not in IB at all (old session, IB no longer reports it) — stale
            logger.warning("Recovery: exit %d for trade %d not found at IB — "
                           "clearing stale DB ref", exit_oid, trade_id)
            try:
                db.update_trade_state(trade_id, ib_exit_order_id=None,
                                      ib_exit_perm_id=None)
            except Exception as e:
                logger.error("Recovery: failed to clear exit ref for trade %d: %s",
                             trade_id, e)
            exit_oid = None

        if exit_oid and exit_oid in all_broker_trades:
            bt = all_broker_trades[exit_oid]
            broker_filled = int(bt.orderStatus.filled or 0)
            broker_status = bt.orderStatus.status

            if broker_filled > 0:
                # Exit filled while app was down — close the trade
                exit_price = float(bt.orderStatus.avgFillPrice or 0)
                exit_time = now_et
                if bt.fills:
                    last_exec = bt.fills[-1].execution
                    exit_time = _broker_time_to_et(
                        getattr(last_exec, 'time', None))
                logger.warning("Recovery: exit %d FILLED @ $%.2f for trade %d — "
                               "closing in DB", exit_oid, exit_price, trade_id)
                try:
                    db.close_trade(trade_id, exit_time=exit_time,
                                   exit_price=exit_price,
                                   exit_reason='exit_filled_during_restart')
                except Exception as e:
                    logger.error("Recovery: failed to close trade %d after exit fill: %s",
                                 trade_id, e)
                continue  # Trade is closed — don't re-register

            if broker_status in ('Cancelled', 'ApiCancelled', 'Inactive'):
                # Exit order died (e.g. DAY expiry) — clear stale ref
                logger.warning("Recovery: exit %d for trade %d is %s — "
                               "clearing stale DB ref", exit_oid, trade_id,
                               broker_status)
                try:
                    db.update_trade_state(trade_id, ib_exit_order_id=None,
                                          ib_exit_perm_id=None)
                except Exception as e:
                    logger.error("Recovery: failed to clear exit ref for trade %d: %s",
                                 trade_id, e)
                exit_oid = None  # So it's not re-registered below

        # ── Register for fill routing (existing logic) ──
        if entry_oid:
            handler._entry_orders[entry_oid] = {
                'trade_id': trade_id,
                'algo_id': trade.get('algo_id', ''),
                'direction': direction,
                'stop_price': trade.get('stop_price') or 0,
                'tp_price': trade.get('tp_price') or 0,
            }

        if exit_oid:
            handler._exit_orders[exit_oid] = {
                'trade_id': trade_id,
                'exit_reason': 'recovered',
            }

        if stop_oid:
            handler._stop_orders[stop_oid] = {'trade_id': trade_id}

        # ── Refresh best_price from current market price ──
        # DB best_price may be stale (from last bar of old session).
        # Use current IB price to ensure trailing stops compute correctly.
        if open_shares > 0:
            current_price = getattr(state, 'tsla_price', None) or 0
            if current_price > 0:
                db_best = trade.get('best_price') or trade.get('entry_price') or 0
                if direction == 'long' and current_price > db_best:
                    try:
                        db.update_trade_state(trade_id, best_price=current_price)
                        logger.info("Recovery: trade %d best_price updated $%.2f → $%.2f "
                                    "(current market)", trade_id, db_best, current_price)
                    except Exception as e:
                        logger.error("Recovery: best_price update failed for trade %d: %s",
                                     trade_id, e)
                elif direction == 'short' and current_price < db_best:
                    try:
                        db.update_trade_state(trade_id, best_price=current_price)
                        logger.info("Recovery: trade %d best_price updated $%.2f → $%.2f "
                                    "(current market)", trade_id, db_best, current_price)
                    except Exception as e:
                        logger.error("Recovery: best_price update failed for trade %d: %s",
                                     trade_id, e)

        # Check stop protection
        if open_shares > 0 and not stop_oid and not exit_oid:
            # Skip re-arm for manual trades — user manages their own stops
            if trade.get('management_mode') == 'manual':
                logger.warning("Trade %d is MANUAL with %d open shares and "
                               "NO stop/exit — UNPROTECTED (user responsibility)",
                               trade_id, open_shares)
            else:
                # No stop and no exit — naked position, re-arm
                stop_price = trade.get('stop_price') or 0
                if stop_price > 0:
                    logger.warning("Trade %d has %d open shares with no stop/exit — "
                                   "re-arming stop @ $%.2f",
                                   trade_id, open_shares, stop_price)
                    result = handler._place_protective_stop(
                        trade_id, stop_price, open_shares, direction)
                    if result is None or result == -1:
                        if handler and hasattr(handler, '_set_degraded'):
                            handler._set_degraded(
                                f"Failed to re-arm stop for trade {trade_id} — "
                                f"position UNPROTECTED")
                        else:
                            state.ib_degraded = True
                            logger.error("CRITICAL: Failed to re-arm stop for "
                                         "trade %d — position UNPROTECTED", trade_id)

    logger.info("Recovery complete: %d trades processed", len(open_trades))


def seed_seen_exec_ids(state):
    """Step 6e (part 1): Seed seen_exec_ids from reqExecutions().

    Must run AFTER recovery has applied historical fills.
    """
    if not state.ib_client or not state.ib_client.is_connected():
        return

    handler = state.ib_order_handler
    try:
        # ib.fills() reads from already-populated cache (synchronous)
        fills = state.ib_client.ib.fills()

        for fill in fills:
            exec_id = fill.execution.execId
            handler.seen_exec_ids.add(exec_id)

        logger.info("Seeded seen_exec_ids: %d executions", len(handler.seen_exec_ids))
    except Exception as e:
        logger.error("Failed to seed seen_exec_ids: %s", e)
        raise


def wire_exec_details_callbacks(state):
    """Step 6e (part 2): Wire execDetailsEvent + statusEvent callbacks.

    Must run AFTER seeding seen_exec_ids. Any replayed executions will
    be deduplicated by the seen set.
    """
    if not state.ib_client or not state.ib_client.is_connected():
        return

    handler = state.ib_order_handler

    # Wire execDetailsEvent for fill tracking (global — persists across reconnects)
    state.ib_client.ib.execDetailsEvent += handler.on_exec_details

    # Register handler.on_order_status as a global status callback on IBClient.
    # This fires for ALL orders (existing + post-startup) via _on_order_status,
    # replacing the old per-trade wiring which missed post-startup orders.
    state.ib_client.register_order_status_callback(handler.on_order_status)

    # Wire IBClient._on_order_status on existing trades so the callback chain fires
    for trade in state.ib_client.ib.openTrades():
        trade.statusEvent += state.ib_client._on_order_status

    logger.info("execDetailsEvent + statusEvent wired (global callback)")


def reconcile_ib_db(state):
    """Step 7: IB/DB quantity-based reconciliation.

    Compares broker positions vs DB open trades per algo.
    Sets ib_degraded on mismatch.
    """
    if not state.ib_client or not state.ib_client.is_connected():
        return

    db = state.trade_db

    # Get broker positions (synchronous cache read)
    try:
        broker_positions = state.ib_client.ib.positions()
    except Exception as e:
        logger.error("Reconciliation failed: could not get broker positions: %s — setting ib_degraded", e)
        state.ib_degraded = True
        try:
            state.trade_db.set_metadata('ib_degraded', '1')
        except Exception as e2:
            logger.error("Failed to persist ib_degraded: %s", e2)
        return

    # Sum broker TSLA position
    broker_net = 0
    for pos in broker_positions:
        if pos.contract.symbol == 'TSLA':
            broker_net += int(pos.position)

    # Sum DB open positions
    open_trades = db.get_open_trades(source='ib')
    db_net = 0
    for t in open_trades:
        open_shares = t.get('open_shares') or 0
        direction = t.get('direction', 'long')
        if direction == 'long':
            db_net += open_shares
        else:
            db_net -= open_shares

    # Per-algo breakdown for visibility (helps debug multi-algo conflicts)
    algo_ids = set(t.get('algo_id') for t in open_trades if t.get('algo_id'))
    for algo_id in sorted(algo_ids):
        algo_trades = [t for t in open_trades if t.get('algo_id') == algo_id]
        algo_net = sum(
            ((t.get('open_shares') or 0) if t.get('direction') == 'long'
             else -(t.get('open_shares') or 0))
            for t in algo_trades)
        logger.info("  Algo %s: DB net=%d (%d trades)", algo_id, algo_net, len(algo_trades))

    if db_net == broker_net:
        logger.info("Reconciliation OK: broker=%d, DB=%d", broker_net, db_net)
        # Auto-clear ib_degraded on successful reconciliation
        if state.ib_degraded:
            state.ib_degraded = False
            try:
                db.set_metadata('ib_degraded', '0')
                logger.info("ib_degraded auto-cleared after successful reconciliation")
            except Exception as e:
                logger.error("Failed to clear ib_degraded: %s", e)
        return

    # Mismatch
    logger.error("RECONCILIATION MISMATCH: broker=%d, DB=%d (delta=%d)",
                 broker_net, db_net, broker_net - db_net)

    if db_net != 0 and broker_net == 0:
        # DB says open, broker is flat — close all orphaned trades
        # This happens when stops fill during restart or orders complete while app is down
        logger.error("DB has positions but broker is flat — closing orphaned trades")
        now_et = datetime.now(ET).isoformat()
        for t in open_trades:
            try:
                # Try to find the fill price from IB completed orders
                exit_price = t.get('stop_price') or t.get('entry_price') or 0
                try:
                    completed = state.ib_client.ib.trades()
                    stop_oid = t.get('ib_stop_order_id')
                    for ct in completed:
                        if ct.order.orderId == stop_oid and ct.orderStatus.filled > 0:
                            exit_price = float(ct.orderStatus.avgFillPrice)
                            break
                except Exception:
                    pass  # Use stop_price as fallback

                db.close_trade(
                    t['id'], exit_time=now_et,
                    exit_price=exit_price,
                    exit_reason='stop_filled_during_restart')
                logger.warning("Closed orphaned trade %d: exit @ $%.2f (stop filled while app was down)",
                               t['id'], exit_price)
            except Exception as e:
                logger.error("Failed to close orphaned trade %d: %s", t['id'], e)

        # Broker is flat and we closed all DB trades — reconciliation is now clean
        logger.info("All orphaned trades closed — reconciliation resolved (broker=0, DB=0)")
        if state.ib_degraded:
            state.ib_degraded = False
            try:
                db.set_metadata('ib_degraded', '0')
                logger.info("ib_degraded auto-cleared after orphaned trade cleanup")
            except Exception as e:
                logger.error("Failed to clear ib_degraded: %s", e)
        return

    # Set degraded for other mismatches (non-zero on both sides)
    state.ib_degraded = True
    try:
        db.set_metadata('ib_degraded', '1')
    except Exception as e:
        logger.error("Failed to persist ib_degraded after reconciliation mismatch: %s", e)
    logger.error("ib_degraded=True set from reconciliation mismatch")
