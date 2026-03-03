"""X14 Panel Dashboard — entry point.

Local:  panel serve v15/panel_dashboard/app.py --show --autoreload
Docker: see Dockerfile
"""

import sys
import os
import logging
import traceback

# Ensure project root is on sys.path so v15.* imports work
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import panel as pn

pn.extension('plotly', 'tabulator')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('x14.panel')


def create_app():
    logger.info("create_app() called — importing modules...")
    try:
        from v15.panel_dashboard.state import DashboardState
        logger.info("  DashboardState imported")
        from v15.panel_dashboard.channel_surfer import channel_surfer_tab
        logger.info("  channel_surfer_tab imported")
        from v15.panel_dashboard.model_compare import model_comparisons_tab
        logger.info("  model_comparisons_tab imported")
    except Exception:
        logger.error("Import failed:\n%s", traceback.format_exc())
        raise

    state = DashboardState()

    # Load market data (blocking on startup)
    logger.info("Starting X14 c14 Panel Dashboard...")
    state.load_market_data()
    logger.info("Market data loaded. TSLA price=%.2f, scanner=%s, scanner_dw=%s, scanner_ml=%s, scanner_intra=%s",
                state.tsla_price,
                "OK" if state.scanner else "NONE",
                "OK" if state.scanner_dw else "NONE",
                "OK" if state.scanner_ml else "NONE",
                "OK" if state.scanner_intra else "NONE")

    # Log ML model status
    if hasattr(state, '_ml_model') and state._ml_model is not None:
        logger.info("ML model: LOADED (%d features)", len(state._ml_feature_names or []))
    else:
        logger.warning("ML model: NOT LOADED — Surfer ML signals will be skipped")

    # Load initial model comparison data
    state.load_model_data()
    logger.info("Model data loaded. Keys=%d", len(state.model_data))

    # Periodic callbacks
    logger.info("Registering periodic callbacks...")
    pn.state.add_periodic_callback(state.update_prices, period=500)
    pn.state.add_periodic_callback(state.run_analysis, period=150_000)   # 2.5 min
    pn.state.add_periodic_callback(state.load_model_data, period=3_600_000)  # 1 hour

    # Sidebar controls
    run_btn = pn.widgets.Button(name='Run Analysis Now', button_type='primary')
    run_btn.on_click(lambda e: state.run_analysis())

    tg_test_btn = pn.widgets.Button(name='Test Telegram', button_type='warning')
    tg_status = pn.pane.Markdown('', width=200)

    def _test_telegram(e):
        tg_status.object = '*Sending...*'
        result = state.send_test_telegram()
        if result == 'OK':
            tg_status.object = '**Sent OK**'
        else:
            tg_status.object = f'**Failed:** {result}'

    tg_test_btn.on_click(_test_telegram)

    reset_btn = pn.widgets.Button(name='Reset Scanner', button_type='danger')

    def _reset_scanner(e):
        if state.scanner:
            state.scanner.reset()
            state.positions_version += 1
            state.trades_version += 1

    reset_btn.on_click(_reset_scanner)

    capital_input = pn.widgets.FloatInput.from_param(
        state.param.scanner_capital,
        name='Scanner Capital ($)',
        step=10_000,
    )
    kill_switch = pn.widgets.Toggle.from_param(
        state.param.kill_switch,
        name='Kill Switch',
    )

    # Sync kill switch to scanner
    def _sync_kill(event):
        if state.scanner:
            state.scanner.config.kill_switch = event.new
    state.param.watch(_sync_kill, 'kill_switch')

    # Last analysis timestamp display
    last_analysis_display = pn.bind(
        lambda ts: pn.pane.Markdown(f"Last analysis: **{ts}**" if ts else "*No analysis yet*"),
        state.param.last_analysis,
    )

    # Build template
    template = pn.template.FastListTemplate(
        title='c14 Trading Dashboard',
        theme='dark',
        accent_base_color='#00c853',
        header_background='#111',
        sidebar=[
            pn.pane.Markdown("### Controls"),
            run_btn,
            pn.layout.Divider(),
            capital_input,
            kill_switch,
            pn.layout.Divider(),
            last_analysis_display,
            pn.layout.Divider(),
            tg_test_btn,
            tg_status,
            pn.layout.Divider(),
            reset_btn,
        ],
        main=[
            pn.Tabs(
                ('Channel Surfer', channel_surfer_tab(state)),
                ('Model Comparisons', model_comparisons_tab(state)),
                dynamic=True,
            ),
        ],
    )
    logger.info("c14 Dashboard template built successfully — ready to serve")
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
            f"<h1>c14 Startup Error</h1><pre>{traceback.format_exc()}</pre>",
            sizing_mode='stretch_width',
        ).servable()
elif __name__ == '__main__':
    # When run directly: `python app.py`
    pn.serve(
        create_app,
        port=int(os.environ.get('PORT', 7860)),
        address='0.0.0.0',
        allow_websocket_origin=['*'],
        title='c14 Dashboard',
        show=False,
    )
