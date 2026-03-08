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

    # Get all open + completed orders from IB
    try:
        open_trades = ib.ib.openTrades()
    except Exception as e:
        logger.error("scan_unlinked_orders: failed to get open trades: %s", e)
        return

    for trade in open_trades:
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
                    except (ValueError, IndexError):
                        stop_price = 0
                    try:
                        tp_price = float(parts[4])
                    except (ValueError, IndexError):
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

        elif order_ref.startswith('exit:'):
            # Check if DB has the exit linked
            try:
                trade_id = int(order_ref.split(':')[1])
            except (ValueError, IndexError):
                continue
            # Register for fill tracking
            handler._exit_orders[order_id] = {
                'trade_id': trade_id, 'exit_reason': 'recovered'}

        elif order_ref.startswith('stop:'):
            try:
                trade_id = int(order_ref.split(':')[1])
            except (ValueError, IndexError):
                continue
            handler._stop_orders[order_id] = {'trade_id': trade_id}

    logger.info("scan_unlinked_orders complete: processed %d open trades",
                len(open_trades))


def recover_inflight_orders(state):
    """Step 6d: Recover pending/partial entries + exits + stops.

    Queries DB for open IB trades and reconciles with broker state.
    execDetailsEvent is NOT wired yet — safe to apply historical fills.
    """
    if not state.ib_client or not state.ib_client.is_connected():
        return

    db = state.trade_db
    handler = state.ib_order_handler

    # Get all open IB trades (including pending)
    open_trades = db.get_open_trades(source='ib', include_pending=True)
    logger.info("Recovery: %d open IB trades to check", len(open_trades))

    for trade in open_trades:
        trade_id = trade['id']
        fill_status = trade.get('ib_fill_status', 'filled')
        direction = trade.get('direction', 'long')
        open_shares = trade.get('open_shares', 0)

        # Register entry/exit/stop orders for fill routing
        entry_oid = trade.get('ib_entry_order_id')
        if entry_oid:
            handler._entry_orders[entry_oid] = {
                'trade_id': trade_id,
                'algo_id': trade.get('algo_id', ''),
                'direction': direction,
                'stop_price': trade.get('stop_price', 0),
                'tp_price': trade.get('tp_price', 0),
            }

        exit_oid = trade.get('ib_exit_order_id')
        if exit_oid:
            handler._exit_orders[exit_oid] = {
                'trade_id': trade_id,
                'exit_reason': 'recovered',
            }

        stop_oid = trade.get('ib_stop_order_id')
        if stop_oid:
            handler._stop_orders[stop_oid] = {'trade_id': trade_id}

        # Check stop protection
        if open_shares > 0 and not stop_oid and not exit_oid:
            # No stop and no exit — naked position, re-arm
            stop_price = trade.get('stop_price', 0)
            if stop_price > 0:
                logger.warning("Trade %d has %d open shares with no stop/exit — "
                               "re-arming stop @ $%.2f",
                               trade_id, open_shares, stop_price)
                handler._place_protective_stop(
                    trade_id, stop_price, open_shares, direction)

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

    # Wire execDetailsEvent for fill tracking
    state.ib_client.ib.execDetailsEvent += handler.on_exec_details

    # Wire statusEvent on existing trades for stop monitoring
    for trade in state.ib_client.ib.openTrades():
        trade.statusEvent += handler.on_order_status

    logger.info("execDetailsEvent + statusEvent wired")


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
        except Exception:
            pass
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
        open_shares = t.get('open_shares', 0)
        direction = t.get('direction', 'long')
        if direction == 'long':
            db_net += open_shares
        else:
            db_net -= open_shares

    if db_net == broker_net:
        logger.info("Reconciliation OK: broker=%d, DB=%d", broker_net, db_net)
        return

    # Mismatch
    logger.error("RECONCILIATION MISMATCH: broker=%d, DB=%d (delta=%d)",
                 broker_net, db_net, broker_net - db_net)

    if db_net != 0 and broker_net == 0:
        # DB says open, broker is flat
        logger.error("DB has positions but broker is flat — marking orphaned")
        for t in open_trades:
            try:
                db.update_trade_state(t['id'], ib_fill_status='orphaned')
            except Exception as e:
                logger.error("Failed to mark trade %d orphaned: %s", t['id'], e)

    # Set degraded
    state.ib_degraded = True
    try:
        db.set_metadata('ib_degraded', '1')
    except Exception:
        pass
    logger.error("ib_degraded=True set from reconciliation mismatch")
