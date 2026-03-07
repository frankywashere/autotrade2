"""
Startup sequence — initializes TradeDB, migration, IB, adapters, scanners.

Follows the exact startup order from REBUILD_PLAN.md Part 8:
  1. Create DB
  2. Migrate JSON -> DB
  3. Connect IB
  4. Load ML models
  5. Create adapters, register with ScannerManagers
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


def run_migration(state):
    """Step 2: Migrate JSON state files to SQLite.

    MUST run before anything else touches DB.
    """
    from v15.panel_dashboard.db.migration import run_migration as _migrate

    try:
        migrated = _migrate(state.trade_db)
        if migrated:
            logger.info("Migration complete")
        else:
            logger.info("Migration already done")
    except Exception as e:
        logger.error("Migration FAILED: %s", e, exc_info=True)
        state.migration_failed = True


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


def create_adapters(state):
    """Step 5: Create algo adapters and register with ScannerManagers."""
    from v15.panel_dashboard.algos.cs_combo import CSComboAdapter
    from v15.panel_dashboard.algos.surfer_ml import SurferMLAdapter
    from v15.panel_dashboard.algos.intraday import IntradayAdapter
    from v15.panel_dashboard.algos.oe_sig5 import OESig5Adapter
    from v15.panel_dashboard.algos.scanner_manager import ScannerManager

    # --- IB ScannerManager ---
    ib_manager = ScannerManager(
        trade_db=state.trade_db,
        source='ib',
        ib_client=state.ib_client,
    )

    # c16 generation: flat $100K, trail^12
    ib_manager.register(CSComboAdapter('c16', config={
        'signal_source': 'CS-5TF', 'equity': 100_000, 'trail_power': 12,
    }))
    ib_manager.register(CSComboAdapter('c16-dw', config={
        'signal_source': 'CS-DW', 'equity': 100_000, 'trail_power': 12,
    }))
    ib_manager.register(SurferMLAdapter('c16-ml', config={
        'equity': 100_000,
    }))
    ib_manager.register(IntradayAdapter('c16-intra', config={
        'equity': 100_000, 'trail_power': 12,
    }))
    ib_manager.register(OESig5Adapter('c16-oe', config={
        'equity': 100_000, 'trail_power': 12,
    }))

    state.ib_scanner_manager = ib_manager

    # --- yfinance ScannerManager ---
    yf_manager = ScannerManager(
        trade_db=state.trade_db,
        source='yf',
        ib_client=None,
    )

    # Same adapters, separate instances (state isolation)
    yf_manager.register(CSComboAdapter('c16', config={
        'signal_source': 'CS-5TF', 'equity': 100_000, 'trail_power': 12,
    }))
    yf_manager.register(CSComboAdapter('c16-dw', config={
        'signal_source': 'CS-DW', 'equity': 100_000, 'trail_power': 12,
    }))
    yf_manager.register(SurferMLAdapter('c16-ml', config={
        'equity': 100_000,
    }))
    yf_manager.register(IntradayAdapter('c16-intra', config={
        'equity': 100_000, 'trail_power': 12,
    }))
    yf_manager.register(OESig5Adapter('c16-oe', config={
        'equity': 100_000, 'trail_power': 12,
    }))

    # c14a generation for yf (different config)
    yf_manager.register(CSComboAdapter('c14a', config={
        'signal_source': 'CS-5TF', 'equity': 100_000,
        'trail_power': 8, 'flat_sizing': False,
    }))
    yf_manager.register(CSComboAdapter('c14a-dw', config={
        'signal_source': 'CS-DW', 'equity': 100_000,
        'trail_power': 8, 'flat_sizing': False,
    }))
    yf_manager.register(SurferMLAdapter('c14a-ml', config={
        'equity': 100_000,
    }))
    yf_manager.register(IntradayAdapter('c14a-intra', config={
        'equity': 100_000, 'trail_power': 8,
    }))

    state.yf_scanner_manager = yf_manager

    logger.info("Adapters registered: IB=%d, yf=%d",
                len(ib_manager.adapters), len(yf_manager.adapters))


def reload_degraded_state(state):
    """Step 6: Reload persisted ib_degraded flag from DB metadata."""
    if getattr(state, 'migration_failed', False):
        return

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
    if getattr(state, 'migration_failed', False):
        return
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
        logger.error("scan_unlinked_orders failed: %s", e)

    # 6d. Recover in-flight orders
    try:
        recover_inflight_orders(state)
    except Exception as e:
        logger.error("recover_inflight_orders failed: %s", e)

    # 6e. Seed seen_exec_ids + wire callbacks
    try:
        seed_seen_exec_ids(state)
        wire_exec_details_callbacks(state)
    except Exception as e:
        logger.error("Seed/wire failed: %s", e)


def run_reconciliation(state):
    """Step 7: IB/DB reconciliation."""
    if getattr(state, 'migration_failed', False):
        return
    if not state.ib_client or not state.ib_connected:
        logger.info("IB not connected — skipping reconciliation")
        return

    from v15.panel_dashboard.ib_recovery import reconcile_ib_db

    try:
        reconcile_ib_db(state)
    except Exception as e:
        logger.error("Reconciliation failed: %s", e)


def start_loops(state):
    """Step 8: Start background loops."""
    if getattr(state, 'migration_failed', False):
        logger.error("Migration failed — NOT starting background loops")
        return

    from v15.panel_dashboard.loops import start_all_loops
    start_all_loops(state)


def full_init(state):
    """Run the complete startup sequence.

    Call this from app.py or state.load_market_data().
    """
    init_trade_db(state)        # 1
    run_migration(state)        # 2
    connect_ib(state)           # 3
    load_models(state)          # 4
    create_adapters(state)      # 5
    create_order_handler(state) # 5b
    reload_degraded_state(state)  # 6
    run_ib_recovery(state)      # 6b-6e
    run_reconciliation(state)   # 7
    start_loops(state)          # 8

    logger.info("Startup complete (migration_failed=%s, ib_degraded=%s, ib_connected=%s)",
                getattr(state, 'migration_failed', False),
                getattr(state, 'ib_degraded', False),
                state.ib_connected)
