"""
Background loop functions — extracted from state.py.

Each loop runs in a daemon thread. Functions take `state` as their first argument
and access state.price_manager, state.trade_db, state.ib_scanner_manager, etc.
"""

import logging
import time
import traceback

import pandas as pd

logger = logging.getLogger(__name__)


def ib_price_loop(state):
    """IB tick-driven price loop (~100ms throttle).

    Updates PriceManager from IB ticks and runs exit checks.
    """
    min_interval = 0.1  # 100ms throttle
    last_update = 0.0

    while True:
        # Wait for IB tick or timeout at 2s
        if state.ib_client and hasattr(state.ib_client, 'tick_event'):
            state.ib_client.tick_event.wait(timeout=2.0)
            state.ib_client.tick_event.clear()
        else:
            time.sleep(2)

        now = time.time()
        if now - last_update < min_interval:
            continue
        last_update = now

        try:
            _update_ib_prices(state)
        except Exception as e:
            logger.error("IB price loop error: %s", e)


def _update_ib_prices(state):
    """Core IB price update + exit checking."""
    price = 0.0
    source_label = 'NONE'

    if state.ib_client and state.ib_client.is_connected():
        ib_price = state.ib_client.get_last_price('TSLA')
        if ib_price > 0:
            price = ib_price
            source_label = 'IB LIVE'
            if not state.ib_connected:
                state.ib_connected = True

            # Update PriceManager
            if state.price_manager:
                bid = getattr(state.ib_client, 'get_bid', lambda s: 0.0)('TSLA')
                ask = getattr(state.ib_client, 'get_ask', lambda s: 0.0)('TSLA')
                state.price_manager.update_ib('TSLA', price, bid, ask)

        # SPY + VIX
        for sym in ['SPY', 'VIX']:
            p = state.ib_client.get_last_price(sym)
            if p > 0:
                if state.price_manager:
                    state.price_manager.update_ib(sym, p)
                if sym == 'SPY' and p != state.spy_price:
                    state.spy_price = p
                elif sym == 'VIX':
                    state.vix_price = p

    elif state.ib_client and not state.ib_client.is_connected():
        if state.ib_connected:
            state.ib_connected = False
            logger.error("IB DISCONNECTED")

    if price == 0.0:
        if state.price_manager:
            state.price_manager.record_error('TSLA')
            err_count = state.price_manager.get_error_count('TSLA')
            if err_count <= 5 or err_count % 10 == 0:
                logger.error("NO LIVE PRICE (count=%d)", err_count)
            if err_count >= 10 and state.ib_connected:
                state.ib_connected = False

    if source_label != state.price_source:
        state.price_source = source_label

    if price > 0:
        price_changed = price != state.tsla_price
        if state._prev_price > 0:
            new_delta = price - state._prev_price
            if new_delta != state.price_delta:
                state.price_delta = new_delta
        state._prev_price = state.tsla_price if state.tsla_price > 0 else price
        if price_changed:
            state.tsla_price = price

        # Check exits for IB trades
        if (hasattr(state, 'ib_scanner_manager') and state.ib_scanner_manager
                and not getattr(state, 'migration_failed', False)):
            try:
                bid = state.price_manager.get('TSLA', 'ib').bid if state.price_manager else price
                ask = state.price_manager.get('TSLA', 'ib').ask if state.price_manager else price
                exits = state.ib_scanner_manager.check_all_exits(price, bid, ask)
                if exits:
                    _handle_exits(state, exits, source='ib')
            except Exception as e:
                logger.warning("IB exit check failed: %s", e)

        # Update trailing stops for IB trades
        if (hasattr(state, 'ib_scanner_manager') and state.ib_scanner_manager
                and not getattr(state, 'migration_failed', False)):
            try:
                updates = state.ib_scanner_manager.update_all_trailing(price)
                for trade_id, changes in updates:
                    state.trade_db.update_trade_state(trade_id, **changes)
                    # If stop_price changed, modify the resting IB stop
                    if ('stop_price' in changes
                            and hasattr(state, 'ib_order_handler')
                            and state.ib_order_handler):
                        state.ib_order_handler.modify_trailing_stop(
                            trade_id, changes['stop_price'])
            except Exception as e:
                logger.warning("IB trailing update failed: %s", e)

        # Bump position version for live P&L
        if price_changed:
            has_positions = bool(state.trade_db.get_open_trades(source='ib'))
            if has_positions:
                state.positions_version += 1


def yf_price_loop(state):
    """yfinance 2s REST price polling loop.

    Polls yf.Ticker('TSLA').fast_info['lastPrice'] for live P&L display.
    """
    while True:
        time.sleep(2)
        try:
            import yfinance as yf
            ticker = yf.Ticker('TSLA')
            info = ticker.fast_info
            price = info.get('lastPrice', 0) or info.get('last_price', 0)
            if price and price > 0:
                if state.price_manager:
                    state.price_manager.update_yf('TSLA', price)

                # Check exits for yf trades
                if (hasattr(state, 'yf_scanner_manager') and state.yf_scanner_manager
                        and not getattr(state, 'migration_failed', False)):
                    exits = state.yf_scanner_manager.check_all_exits(price)
                    if exits:
                        _handle_exits(state, exits, source='yf')

                    # Update trailing stops for yf trades
                    updates = state.yf_scanner_manager.update_all_trailing(price)
                    for trade_id, changes in updates:
                        state.trade_db.update_trade_state(trade_id, **changes)
        except Exception as e:
            logger.warning("yf price loop error: %s", e)


def analysis_loop(state):
    """Analysis loop — triggers on 5-min bar close or 60s timeout."""
    time.sleep(30)  # Initial delay
    while True:
        if hasattr(state, '_bar_aggregator') and state._bar_aggregator:
            state._bar_aggregator.bar_close_event.wait(timeout=60)
            state._bar_aggregator.bar_close_event.clear()
        else:
            time.sleep(60)

        try:
            _run_analysis(state)
        except Exception as e:
            logger.error("Analysis loop error: %s\n%s", e, traceback.format_exc())


def tf_refresh_loop(state):
    """Refresh higher TF bars from IB every 30 min."""
    while True:
        time.sleep(1800)
        if (state.ib_client and state.ib_client.is_connected()
                and state.data_source == 'IB'):
            try:
                new_data = state._load_ib_historical()
                if new_data and 'TSLA' in new_data:
                    state.native_tf_data = new_data
                    logger.info("Higher TF data refreshed from IB")
            except Exception as e:
                logger.warning("Higher TF refresh failed: %s", e)


def _handle_exits(state, exit_signals, source='ib'):
    """Process exit signals — close trades in DB (yf) or place IB orders (ib).

    For source='yf': instant close at signal price.
    For source='ib': two-phase commit (order placement + fill callback).
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo
    ET = ZoneInfo('US/Eastern')

    for exit_sig in exit_signals:
        trade_id = exit_sig.trade_id
        exit_price = exit_sig.exit_price
        exit_reason = exit_sig.exit_reason

        if source == 'yf':
            # Instant close for yfinance
            try:
                now_et = datetime.now(ET).isoformat()
                state.trade_db.close_trade(
                    trade_id, exit_time=now_et,
                    exit_price=exit_price, exit_reason=exit_reason)
                state.positions_version += 1
                state.trades_version += 1
                logger.info("yf trade %d closed: %s @ $%.2f",
                            trade_id, exit_reason, exit_price)
            except Exception as e:
                logger.error("Failed to close yf trade %d: %s", trade_id, e)
        else:
            # IB exit: two-phase commit via IBOrderHandler
            if hasattr(state, 'ib_order_handler') and state.ib_order_handler:
                try:
                    state.ib_order_handler.place_exit(
                        trade_id, exit_reason, exit_price)
                except Exception as e:
                    logger.error("IB exit failed for trade %d: %s",
                                 trade_id, e)
            else:
                logger.warning("No IB order handler — exit signal for trade %d "
                               "not placed", trade_id)


def _run_analysis(state):
    """Run IB-based analysis (thin wrapper for existing logic)."""
    if not state.native_tf_data:
        return
    if getattr(state, '_analysis_running', False):
        return

    # Delegate to existing analysis method on state
    if hasattr(state, '_run_analysis_bg'):
        state._run_analysis_bg()


def start_all_loops(state):
    """Start all background loops as daemon threads."""
    import threading

    loops = [
        (ib_price_loop, 'ib-price'),
        (yf_price_loop, 'yf-price'),
        (analysis_loop, 'analysis'),
        (tf_refresh_loop, 'tf-refresh'),
    ]

    for fn, name in loops:
        t = threading.Thread(target=fn, args=(state,), daemon=True,
                             name=f'x14-{name}')
        t.start()

    logger.info("Background loops started: %s",
                ', '.join(name for _, name in loops))
