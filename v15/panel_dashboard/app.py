"""X14 Panel Dashboard — entry point.

Local:  panel serve v15/panel_dashboard/app.py --show --autoreload
Docker: see Dockerfile
"""

import sys
import os
import logging
import logging.handlers
import traceback

# Ensure project root is on sys.path so v15.* imports work
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import panel as pn

pn.extension('plotly', 'tabulator')

_log_dir = os.path.join(_project_root, 'logs')
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, 'dashboard.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.handlers.RotatingFileHandler(
            _log_file, maxBytes=5_000_000, backupCount=3,
        ),
    ],
)
logger = logging.getLogger('x14.panel')

# Also capture unhandled exceptions to log file
def _exc_hook(exc_type, exc_value, exc_tb):
    logger.error("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))
    sys.__excepthook__(exc_type, exc_value, exc_tb)
sys.excepthook = _exc_hook


# ── Singleton state — initialized once, shared across all sessions ────────────
_state = None


def _init_state():
    """Initialize DashboardState once (data load, scanners, background loops)."""
    global _state
    if _state is not None:
        return _state

    from v15.panel_dashboard.state import DashboardState

    logger.info("Initializing DashboardState (one-time)...")
    _state = DashboardState()

    _state.load_market_data()
    logger.info("Market data loaded. TSLA=%.2f", _state.tsla_price)

    # Log IB status
    ib_ok = getattr(_state, 'ib_connected', False)
    if ib_ok:
        logger.info("IB Gateway: CONNECTED")
    else:
        logger.error("IB Gateway: NOT CONNECTED — no live price source available!")

    # Initialize infrastructure (TradeDB, adapters, order handler, LiveEngine, recovery, loops)
    _init_new_infra(_state)

    return _state


def _init_new_infra(state):
    """Initialize DB-backed infrastructure: TradeDB, adapters, LiveEngine, recovery, loops."""
    try:
        from v15.panel_dashboard.startup import (
            init_trade_db,
            create_order_handler, create_live_engine,
            reload_degraded_state,
            run_ib_recovery, run_reconciliation,
            start_loops,
        )

        # 1. Create DB
        init_trade_db(state)

        # 2. Create order handler + live engine
        create_order_handler(state)
        create_live_engine(state)
        reload_degraded_state(state)

        # 3. Recovery + reconciliation (if IB connected)
        if state.ib_connected:
            run_ib_recovery(state)
            run_reconciliation(state)

        # 4. Start background loops (price, exit checks, analysis, TF refresh)
        start_loops(state)

        logger.info("Infrastructure initialized (trade_db=%s, ib_handler=%s, "
                     "live_engine=%s, ib_degraded=%s)",
                     'OK' if state.trade_db else 'NONE',
                     'OK' if getattr(state, 'ib_order_handler', None) else 'NONE',
                     'OK' if getattr(state, 'live_engine', None) else 'NONE',
                     state.ib_degraded)

        # Send startup notification
        ib_ok = getattr(state, 'ib_connected', False)
        engine_ok = getattr(state, 'live_engine', None) is not None
        startup_msg = (
            f"LiveEngine: {'OK' if engine_ok else 'FAIL'}\n"
            f"TSLA price: ${state.tsla_price:.2f}\n"
            f"IB Gateway: {'CONNECTED' if ib_ok else 'NOT CONNECTED'}\n"
            f"ib_degraded: {state.ib_degraded}"
        )
        state.send_notification(startup_msg, title='c17 Startup')

    except Exception as e:
        logger.error("Infrastructure init failed (non-fatal): %s\n%s",
                     e, traceback.format_exc())


def create_app():
    logger.info("create_app() called (session factory)")
    try:
        from v15.panel_dashboard.tabs.ib_live import ib_live_tab, _kill_switch_panel
    except Exception:
        logger.error("Tab import failed:\n%s", traceback.format_exc())
        raise

    state = _init_state()

    # UI refresh only — triggers re-render when browser is connected
    pn.state.add_periodic_callback(lambda: None, period=2000)

    # Sidebar controls
    run_btn = pn.widgets.Button(name='Run Analysis Now', button_type='primary')
    run_btn.on_click(lambda e: state.run_analysis())

    ntfy_test_btn = pn.widgets.Button(name='Test Notification', button_type='warning')
    ntfy_status = pn.pane.Markdown('', width=200)

    def _test_ntfy(e):
        ntfy_status.object = '*Sending...*'
        result = state.send_test_notification()
        if result == 'OK':
            ntfy_status.object = '**Sent OK**'
        else:
            ntfy_status.object = f'**Failed:** {result}'

    ntfy_test_btn.on_click(_test_ntfy)

    # Last analysis timestamp display
    last_analysis_display = pn.bind(
        lambda ts: pn.pane.Markdown(f"Last analysis: **{ts}**" if ts else "*No analysis yet*"),
        state.param.last_analysis,
    )

    # IB connection status (reactive)
    ib_status_display = pn.bind(
        lambda connected: pn.pane.HTML(
            f'<span style="color:{"#00c853" if connected else "#ff5252"}">'
            f'{"&#9679;" if connected else "&#9675;"}</span> '
            f'IB: <b>{"CONNECTED" if connected else "DISCONNECTED"}</b>',
            width=200,
        ),
        state.param.ib_connected,
    )

    # IB Reconnect button — disabled when connected
    ib_reconnect_btn = pn.widgets.Button(
        name='Reconnect IB', button_type='warning', width=200,
        disabled=state.ib_connected,
    )
    ib_reconnect_status = pn.pane.Markdown('', width=200)

    def _sync_ib_btn(event):
        ib_reconnect_btn.disabled = event.new
        if event.new:
            ib_reconnect_status.object = ''
    state.param.watch(_sync_ib_btn, 'ib_connected')

    def _reconnect_ib(e):
        ib_reconnect_status.object = '*Reconnecting...*'
        if state.ib_client:
            state.ib_client.reconnect()
            import time
            for _ in range(30):
                time.sleep(0.5)
                if state.ib_client.is_connected():
                    # Wait for prices to actually flow
                    for _ in range(20):
                        time.sleep(0.5)
                        ib_price = state.ib_client.get_last_price('TSLA')
                        if ib_price > 0:
                            state.ib_connected = True
                            state._price_err_count = 0
                            # Re-init bar aggregator for 5-min analysis triggers
                            try:
                                state._bar_aggregator = state.ib_client.create_bar_aggregator('TSLA', 5)
                            except Exception as ex:
                                logger.warning("Bar aggregator re-init failed: %s", ex)
                            ib_reconnect_status.object = '**Connected!**'
                            return
                    # Socket up but no prices
                    state.ib_connected = False
                    ib_reconnect_status.object = '**Socket OK but no prices** — try again'
                    return
            state.ib_connected = False
            ib_reconnect_status.object = '**Failed** — check IB Gateway'
        else:
            ib_reconnect_status.object = '**No IB client**'

    ib_reconnect_btn.on_click(_reconnect_ib)

    # Build template
    template = pn.template.FastListTemplate(
        title='c16 Trading Dashboard',
        theme='dark',
        accent_base_color='#00c853',
        header_background='#111',
        sidebar=[
            pn.pane.Markdown("### Controls"),
            run_btn,
            _kill_switch_panel(state),
            pn.layout.Divider(),
            last_analysis_display,
            pn.pane.Markdown("### Data Sources"),
            ib_status_display,
            ib_reconnect_btn,
            ib_reconnect_status,
            pn.layout.Divider(),
            ntfy_test_btn,
            ntfy_status,
        ],
        main=[
            ib_live_tab(state),
        ],
    )
    logger.info("c14a Dashboard template built successfully — ready to serve")
    return template


if __name__.startswith('bokeh'):
    # When served via `panel serve app.py`
    logger.info("Bokeh entry point — calling create_app().servable()")
    try:
        create_app().servable()
        logger.info("servable() completed — app should be available at /app")
    except Exception:
        logger.error("FATAL: create_app() failed:\n%s", traceback.format_exc())
        pn.pane.HTML(
            f"<h1>c14a Startup Error</h1><pre>{traceback.format_exc()}</pre>",
            sizing_mode='stretch_width',
        ).servable()
elif __name__ == '__main__':
    # When run directly: `python app.py`
    # Initialize state once before starting server
    _init_state()
    pn.serve(
        create_app,
        port=int(os.environ.get('PORT', 7860)),
        address='0.0.0.0',
        allow_websocket_origin=['*'],
        title='c14 Dashboard',
        show=False,
    )
