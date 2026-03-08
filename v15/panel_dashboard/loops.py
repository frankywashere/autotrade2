"""
Background loop functions — extracted from state.py.

Each loop runs in a daemon thread. Functions take `state` as their first argument
and access state.price_manager, state.trade_db, etc.
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

        # Exit checks + trailing handled by LiveEngine via bar events

        # Bump position version for live P&L
        if price_changed:
            has_positions = bool(state.trade_db.get_open_trades(source='ib'))
            if has_positions:
                state.positions_version += 1


def yf_price_loop(state):
    """yfinance 30s REST price polling loop.

    Polls yf.Ticker lastPrice for TSLA/SPY/VIX.
    Feeds prices to PriceManager + YfinanceDataProvider (for bar construction).
    Exit/trailing handled by yf LiveEngine via bar events.
    """
    while True:
        time.sleep(30)
        try:
            import yfinance as yf
            for symbol, yf_sym in [('TSLA', 'TSLA'), ('SPY', 'SPY'), ('VIX', '^VIX')]:
                try:
                    ticker = yf.Ticker(yf_sym)
                    info = ticker.fast_info
                    price = info.get('lastPrice', 0) or info.get('last_price', 0)
                    if price and price > 0:
                        if state.price_manager:
                            state.price_manager.update_yf(symbol, price)
                        # Feed to YfinanceDataProvider for synthetic bar construction
                        yf_data = getattr(state, 'yf_data_provider', None)
                        if yf_data:
                            yf_data.on_price_update(symbol, price)
                except Exception as e:
                    logger.debug("yf price fetch %s: %s", symbol, e)
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


def _run_analysis(state):
    """Run analysis to refresh channel chart UI."""
    if not state.native_tf_data:
        return
    state.run_analysis()


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
