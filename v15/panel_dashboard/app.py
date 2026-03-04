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
    logger.info("Starting X14 c14a Panel Dashboard...")
    state.load_market_data()
    logger.info("Market data loaded. TSLA price=%.2f, scanner=%s, scanner_dw=%s, scanner_ml=%s, scanner_intra=%s",
                state.tsla_price,
                "OK" if state.scanner else "NONE",
                "OK" if state.scanner_dw else "NONE",
                "OK" if state.scanner_ml else "NONE",
                "OK" if state.scanner_intra else "NONE")

    # Log ML model status + send startup notification
    gbt_ok = hasattr(state, '_ml_model') and state._ml_model is not None
    intra_ok = hasattr(state, '_intraday_ml_model') and state._intraday_ml_model is not None

    # Build detailed status messages
    gbt_msg = f"LOADED ({len(state._ml_feature_names or [])} features)" if gbt_ok else "NOT LOADED"
    intra_msg = (f"LOADED ({len(getattr(state, '_intraday_ml_features', []) or [])} features, "
                 f"threshold={getattr(state, '_intraday_ml_threshold', 0.5):.2f})") if intra_ok else "NOT LOADED"

    # Check file existence for diagnostics
    from pathlib import Path
    gbt_path = Path('surfer_models/gbt_model.pkl')
    intra_path = Path('surfer_models/intraday_ml_model.pkl')
    gbt_diag = ""
    intra_diag = ""
    if not gbt_ok:
        if gbt_path.exists():
            import os
            gbt_diag = f" (file exists, {os.path.getsize(gbt_path)} bytes — load error)"
        else:
            gbt_diag = f" (file missing at {gbt_path.resolve()})"
    if not intra_ok:
        if intra_path.exists():
            import os
            intra_diag = f" (file exists, {os.path.getsize(intra_path)} bytes — load error)"
        else:
            intra_diag = f" (file missing at {intra_path.resolve()})"

    if gbt_ok:
        logger.info("ML model (GBT): %s", gbt_msg)
    else:
        logger.warning("ML model (GBT): %s%s — c14-ml signals will be skipped", gbt_msg, gbt_diag)
    if intra_ok:
        logger.info("ML model (Intraday): %s", intra_msg)
    else:
        logger.warning("ML model (Intraday): %s%s — c14-intra ML filter disabled", intra_msg, intra_diag)

    # Send startup status via Telegram
    scanner_err = getattr(state, '_scanner_init_error', '')
    gbt_err = getattr(state, '_gbt_load_error', '')
    startup_msg = (
        f"Scanners: CS-5TF={'OK' if state.scanner else 'FAIL'}, "
        f"CS-DW={'OK' if state.scanner_dw else 'FAIL'}, "
        f"Surfer ML={'OK' if state.scanner_ml else 'FAIL'}, "
        f"Intraday={'OK' if state.scanner_intra else 'FAIL'}\n"
        f"GBT model: {gbt_msg}{gbt_diag}\n"
        f"Intraday model: {intra_msg}{intra_diag}\n"
        f"TSLA price: ${state.tsla_price:.2f}"
    )
    if scanner_err:
        startup_msg += f"\nSCANNER ERROR: {scanner_err[:200]}"
    if gbt_err:
        startup_msg += f"\nGBT ERROR: {gbt_err[:200]}"
    state.send_notification(startup_msg, title='c14a Startup')

    # Load initial model comparison data
    state.load_model_data()
    logger.info("Model data loaded. Keys=%d", len(state.model_data))

    # Start background loops (run even without browser)
    logger.info("Starting background loops...")
    state.start_background_loops()

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

    # ML model status display
    gbt_ok = hasattr(state, '_ml_model') and state._ml_model is not None
    intra_ok = hasattr(state, '_intraday_ml_model') and state._intraday_ml_model is not None
    gbt_info = f"{len(state._ml_feature_names or [])}f" if gbt_ok else "MISSING"
    intra_info = (f"{len(getattr(state, '_intraday_ml_features', []) or [])}f"
                  if intra_ok else "MISSING")
    ml_status = pn.pane.Markdown(
        f"GBT: **{'OK' if gbt_ok else 'MISSING'}** ({gbt_info})  \n"
        f"Intra: **{'OK' if intra_ok else 'MISSING'}** ({intra_info})",
        width=200,
    )

    # Build template
    template = pn.template.FastListTemplate(
        title='c14a Trading Dashboard',
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
            pn.pane.Markdown("### ML Models"),
            ml_status,
            pn.layout.Divider(),
            ntfy_test_btn,
            ntfy_status,
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
    pn.serve(
        create_app,
        port=int(os.environ.get('PORT', 7860)),
        address='0.0.0.0',
        allow_websocket_origin=['*'],
        title='c14 Dashboard',
        show=False,
    )
