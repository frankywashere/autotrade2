"""
Startup sequence — initializes TradeDB, IB, LiveEngine, scanners.

Follows the startup order from REBUILD_PLAN.md Part 8:
  1. Create DB
  3. Connect IB
  4. Load ML models
  5b. Create IBOrderHandler
  5c. Create LiveEngine (unified backtester algos + tick-to-bar feed)
  6. Reload ib_degraded, populate open-order cache, scan unlinked, recover, seed+wire
  7. Reconcile IB/DB
  8. Start background loops
"""

import logging
import random

logger = logging.getLogger(__name__)


def init_trade_db(state):
    """Step 1: Create TradeDB instance."""
    from v15.panel_dashboard.db.trade_db import TradeDB
    state.trade_db = TradeDB()
    logger.info("TradeDB created at %s", state.trade_db._db_path)



def connect_ib(state):
    """Step 3: Connect to IB Gateway.

    Does NOT wire execDetailsEvent yet — deferred until after recovery.
    """
    from v15.panel_dashboard.price_manager import PriceManager

    state.price_manager = PriceManager()

    try:
        from v15.ib.client import IBClient
        cid = random.randint(10, 99)
        state.ib_client = IBClient(host='127.0.0.1', port=4002, client_id=cid)
        state.ib_client.connect()
        state.ib_client.subscribe('TSLA')
        state.ib_client.subscribe('SPY')
        state.ib_client.subscribe('VIX')
        state.ib_connected = True
        state.price_manager._ib = state.ib_client
        logger.info("IB connected (client_id=%d)", cid)
    except Exception as e:
        logger.error("IB connection failed: %s", e)
        state.ib_client = None
        state.ib_connected = False


def load_models(state):
    """Step 4: Load ML models."""
    from v15.panel_dashboard.ml_loader import load_ml_models

    models = load_ml_models()
    state._ml_models = models
    state._ml_model = models.gbt_model
    state._ml_feature_names = models.gbt_feature_names
    state._el_model = models.el_model
    state._er_model = models.er_model
    state._intraday_ml_model = models.intraday_model
    state._intraday_ml_features = models.intraday_features

    if models.load_errors:
        logger.warning("ML model load errors: %s", models.load_errors)


def create_live_engine(state):
    """Step 5c: Create LiveEngine with unified backtester algos.

    Instantiates the same algo classes used by the backtester,
    configured for live execution. IB algos get live_orders=True.
    """
    from v15.panel_dashboard.live_data import LiveDataProvider
    from v15.panel_dashboard.live_engine import LiveEngine
    from v15.validation.unified_backtester.algo_base import AlgoConfig, CostModel
    from v15.validation.unified_backtester.algos.surfer_ml import SurferMLAlgo
    from v15.validation.unified_backtester.algos.intraday import IntradayAlgo
    from v15.validation.unified_backtester.algos.cs_combo import CSComboAlgo
    from v15.validation.unified_backtester.algos.oe_sig5 import OESig5Algo
    import datetime as dt

    try:
        # Create LiveDataProvider
        data = LiveDataProvider(ib_client=state.ib_client)
        state.live_data_provider = data

        # Create algo instances for IB execution
        cost = CostModel(slippage_pct=0.0, commission_per_share=0.0)

        ib_algos = [
            SurferMLAlgo(config=AlgoConfig(
                algo_id='c16-ml', live_orders=True,
                initial_equity=100_000.0, max_equity_per_trade=100_000.0,
                max_positions=2, primary_tf='5min', eval_interval=3,
                exit_check_tf='5min', cost_model=cost,
                params={
                    'flat_sizing': True, 'min_confidence': 0.01,
                    'max_hold_bars': 60, 'ou_half_life': 5.0,
                    'stop_pct': 0.015, 'tp_pct': 0.012,
                    'breakout_stop_mult': 1.00,
                },
            ), data=data),
            IntradayAlgo(config=AlgoConfig(
                algo_id='c16-intra', live_orders=True,
                initial_equity=100_000.0, max_equity_per_trade=100_000.0,
                max_positions=1, primary_tf='5min', eval_interval=1,
                exit_check_tf='5min', cost_model=cost,
                active_start=dt.time(9, 30), active_end=dt.time(15, 25),
                params={
                    'flat_sizing': True,
                    'max_trades_per_day': 30,
                    'trail_base': 0.006,
                    'trail_power': 6,
                    'trail_floor': 0.0,
                    'stop_pct': 0.008,
                    'tp_pct': 0.020,
                    'signal_params': {
                        'vwap_thresh': -0.10,
                        'd_min': 0.20,
                        'h1_min': 0.15,
                        'f5_thresh': 0.35,
                        'div_thresh': 0.20,
                        'div_f5_thresh': 0.35,
                        'min_vol_ratio': 0.8,
                        'stop': 0.008,
                        'tp': 0.020,
                    },
                },
            ), data=data),
            CSComboAlgo(config=AlgoConfig(
                algo_id='c16', live_orders=True,
                initial_equity=100_000.0, max_equity_per_trade=100_000.0,
                max_positions=1, primary_tf='daily', eval_interval=1,
                exit_check_tf='5min', cost_model=cost,
                params={
                    'signal_source': 'CS-5TF', 'flat_sizing': True,
                    'stop_pct': 0.02, 'tp_pct': 0.04,
                    'target_tfs': ['5min', '1h', '4h', 'daily', 'weekly', 'monthly'],
                    'trail_power': 12, 'trail_base': 0.025,
                    'max_hold_days': 10, 'cooldown_days': 2,
                    'min_confidence': 0.45,
                },
            ), data=data),
            CSComboAlgo(config=AlgoConfig(
                algo_id='c16-dw', live_orders=True,
                initial_equity=100_000.0, max_equity_per_trade=100_000.0,
                max_positions=1, primary_tf='daily', eval_interval=1,
                exit_check_tf='5min', cost_model=cost,
                params={
                    'signal_source': 'CS-DW', 'flat_sizing': True,
                    'stop_pct': 0.02, 'tp_pct': 0.04,
                    'target_tfs': ['daily', 'weekly'],
                    'trail_power': 12, 'trail_base': 0.025,
                    'max_hold_days': 10, 'cooldown_days': 0,
                    'min_confidence': 0.45,
                },
            ), data=data),
            OESig5Algo(config=AlgoConfig(
                algo_id='c16-oe', live_orders=True,
                initial_equity=100_000.0, max_equity_per_trade=100_000.0,
                max_positions=1, primary_tf='daily', eval_interval=1,
                exit_check_tf='5min', cost_model=cost,
                params={
                    'flat_sizing': True,
                    'stop_pct': 0.03, 'tp_pct': 0.04,
                    'default_confidence': 0.7,
                    'trail_power': 12, 'trail_base': 0.025,
                    'max_hold_days': 10, 'cooldown_days': 0,
                },
            ), data=data),
        ]

        engine = LiveEngine(
            algos=ib_algos,
            data=data,
            trade_db=state.trade_db,
            ib_order_handler=getattr(state, 'ib_order_handler', None),
        )

        # Load persisted enabled/equity state from DB metadata
        if state.trade_db:
            for algo in ib_algos:
                # Enabled state
                val = state.trade_db.get_metadata(f'enabled_{algo.algo_id}')
                if val is not None:
                    engine._algo_enabled[algo.algo_id] = (val == '1')
                    if val == '0':
                        logger.info("Loaded %s as DISABLED from DB", algo.algo_id)
                # Equity allocation
                eq_val = state.trade_db.get_metadata(f'equity_{algo.algo_id}')
                if eq_val is not None:
                    try:
                        algo.config.max_equity_per_trade = float(eq_val)
                        logger.info("Loaded %s equity=$%.0f from DB",
                                    algo.algo_id, float(eq_val))
                    except ValueError as e:
                        logger.warning("Invalid equity value for %s: %s",
                                       algo.algo_id, eq_val)

        # Recover state from DB
        engine.recover_after_restart()

        state.live_engine = engine

        # Create 1-min bar aggregators for all symbols and wire to LiveDataProvider
        if state.ib_client:
            _wire_tick_to_bar_feed(state, data)

        logger.info("LiveEngine created with %d algos: %s",
                     len(ib_algos),
                     [a.algo_id for a in ib_algos])
    except Exception as e:
        logger.error("CRITICAL: Failed to create LiveEngine — "
                     "NO algo execution possible: %s", e, exc_info=True)
        state.live_engine = None
        state.ib_degraded = True
        if state.trade_db:
            try:
                state.trade_db.set_metadata('ib_degraded', '1')
            except Exception as e:
                logger.error("Failed to persist ib_degraded: %s", e)


def _wire_tick_to_bar_feed(state, data):
    """Create 1-min bar aggregators for TSLA/SPY/VIX and feed to LiveDataProvider.

    Each symbol gets a 1-min LiveBarAggregator (via add_bar_aggregator to avoid
    overwriting the existing 5-min TSLA aggregator used by the old scanner system).
    A daemon thread watches for completed bars and routes them to
    LiveDataProvider.on_1min_close().

    Bar timestamps are converted from local time to naive ET (matching IB historical
    bar timestamps). This is critical when the server runs in CDT/PST/etc. —
    all boundary checks (RTH open at 9:31, daily close at 16:00, 4h at 13:00/16:00)
    assume Eastern time.
    """
    import threading
    import time as _time
    from zoneinfo import ZoneInfo
    import pandas as pd

    ET = ZoneInfo('US/Eastern')
    symbols = ['TSLA', 'SPY', 'VIX']
    aggregators = {}
    for symbol in symbols:
        agg = state.ib_client.add_bar_aggregator(symbol, 1)
        aggregators[symbol] = agg

    state._1min_aggregators = aggregators

    def _feed_loop():
        """Watch all 1-min aggregators, feed completed bars to LiveDataProvider."""
        consumed = {s: 0 for s in symbols}
        while True:
            any_new = False
            for symbol in symbols:
                agg = aggregators[symbol]
                with agg._lock:
                    n = len(agg._completed_bars)
                if n > consumed[symbol]:
                    with agg._lock:
                        new_bars = list(agg._completed_bars[consumed[symbol]:])
                    for bar in new_bars:
                        # bar['time'] is naive local datetime from datetime.now().
                        # Convert to tz-aware ET (matching IB historical bars
                        # which are tz-aware from ib_async).
                        bar_local = bar['time']
                        bar_et = bar_local.astimezone().astimezone(ET)
                        bar_time = pd.Timestamp(bar_et) + pd.Timedelta(minutes=1)
                        try:
                            data.on_1min_close(symbol, bar_time, bar)
                        except Exception as e:
                            logger.error("Feed %s 1-min bar failed: %s",
                                         symbol, e)
                    consumed[symbol] = n
                    any_new = True
            if not any_new:
                _time.sleep(0.5)

    t = threading.Thread(target=_feed_loop, daemon=True, name='TickToBarFeed')
    t.start()
    logger.info("Tick-to-bar feed started for %s", symbols)

    # Register reconnect callback: re-seed exec_ids + backfill data gaps
    def _on_ib_reconnect():
        logger.info("IB reconnected — running mid-session recovery")
        handler = getattr(state, 'ib_order_handler', None)
        if handler:
            try:
                fills = state.ib_client.ib.fills()
                for fill in fills:
                    handler.seen_exec_ids.add(fill.execution.execId)
                logger.info("Re-seeded seen_exec_ids: %d", len(handler.seen_exec_ids))
            except Exception as e:
                logger.error("Mid-session re-seed failed: %s", e)
        # Backfill 1-min bars for any gap
        for sym in symbols:
            try:
                data.backfill_gap(sym, pd.Timestamp.now() - pd.Timedelta(hours=1))
            except Exception as e:
                logger.error("Backfill %s failed: %s", sym, e)

    state.ib_client.register_reconnect_callback(_on_ib_reconnect)


def reload_degraded_state(state):
    """Step 6: Reload persisted ib_degraded flag from DB metadata."""
    try:
        if state.trade_db.get_metadata('ib_degraded') == '1':
            state.ib_degraded = True
            logger.warning("ib_degraded=True loaded from DB metadata")
    except Exception as e:
        logger.error("Failed to reload ib_degraded: %s", e)


def create_order_handler(state):
    """Step 5b: Create the IBOrderHandler (shared between scanners and manual)."""
    from v15.panel_dashboard.ib_order_handler import IBOrderHandler
    state.ib_order_handler = IBOrderHandler(state)
    logger.info("IBOrderHandler created")


def run_ib_recovery(state):
    """Steps 6b-6e: Open-order cache, unlinked scan, recovery, seed+wire."""
    if not state.ib_client or not state.ib_connected:
        logger.info("IB not connected — skipping recovery")
        return
    if not hasattr(state, 'ib_order_handler') or not state.ib_order_handler:
        logger.warning("No order handler — skipping recovery")
        return

    from v15.panel_dashboard.ib_recovery import (
        scan_unlinked_orders, recover_inflight_orders,
        seed_seen_exec_ids, wire_exec_details_callbacks,
    )

    # 6b. Populate open-order cache
    try:
        state.ib_client.sync_orders()
        logger.info("Open-order cache populated")
    except Exception as e:
        logger.error("Failed to populate open-order cache: %s", e)

    # 6c. Scan unlinked orders
    try:
        scan_unlinked_orders(state)
    except Exception as e:
        logger.error("CRITICAL: scan_unlinked_orders failed: %s — setting ib_degraded", e)
        state.ib_degraded = True
        state.trade_db.set_metadata('ib_degraded', '1')

    # 6d. Recover in-flight orders
    try:
        recover_inflight_orders(state)
    except Exception as e:
        logger.error("CRITICAL: recover_inflight_orders failed: %s — setting ib_degraded", e)
        state.ib_degraded = True
        state.trade_db.set_metadata('ib_degraded', '1')

    # 6e. Seed seen_exec_ids + wire callbacks
    try:
        seed_seen_exec_ids(state)
        wire_exec_details_callbacks(state)
    except Exception as e:
        logger.error("CRITICAL: Seed/wire failed: %s — setting ib_degraded", e)
        state.ib_degraded = True
        state.trade_db.set_metadata('ib_degraded', '1')


def run_reconciliation(state):
    """Step 7: IB/DB reconciliation."""
    if not state.ib_client or not state.ib_connected:
        logger.info("IB not connected — skipping reconciliation")
        return

    from v15.panel_dashboard.ib_recovery import reconcile_ib_db

    try:
        reconcile_ib_db(state)
    except Exception as e:
        logger.error("CRITICAL: Reconciliation failed: %s — setting ib_degraded", e)
        state.ib_degraded = True
        state.trade_db.set_metadata('ib_degraded', '1')


def start_loops(state):
    """Step 8: Start background loops."""
    from v15.panel_dashboard.loops import start_all_loops
    start_all_loops(state)


def full_init(state):
    """Run the complete startup sequence.

    Call this from app.py or state.load_market_data().
    """
    init_trade_db(state)        # 1
    connect_ib(state)           # 3
    load_models(state)          # 4
    create_order_handler(state) # 5b
    create_live_engine(state)   # 5c
    reload_degraded_state(state)  # 6
    run_ib_recovery(state)      # 6b-6e
    run_reconciliation(state)   # 7
    start_loops(state)          # 8

    logger.info("Startup complete (ib_degraded=%s, ib_connected=%s)",
                getattr(state, 'ib_degraded', False),
                state.ib_connected)
