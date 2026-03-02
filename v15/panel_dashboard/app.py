"""X14 Panel Dashboard — entry point.

Local:  panel serve v15/panel_dashboard/app.py --show --autoreload
Docker: see Dockerfile
"""

import sys
import os
import logging

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
    from v15.panel_dashboard.state import DashboardState
    from v15.panel_dashboard.channel_surfer import channel_surfer_tab
    from v15.panel_dashboard.model_compare import model_comparisons_tab

    state = DashboardState()

    # Load market data (blocking on startup)
    logger.info("Starting X14 Panel Dashboard...")
    state.load_market_data()

    # Load initial model comparison data
    state.load_model_data()

    # Periodic callbacks
    pn.state.add_periodic_callback(state.update_prices, period=500)
    pn.state.add_periodic_callback(state.run_analysis, period=300_000)   # 5 min
    pn.state.add_periodic_callback(state.load_model_data, period=3_600_000)  # 1 hour

    # Sidebar controls
    run_btn = pn.widgets.Button(name='Run Analysis Now', button_type='primary')
    run_btn.on_click(lambda e: state.run_analysis())

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
        title='c13a Trading Dashboard',
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
    return template


if __name__.startswith('bokeh'):
    # When served via `panel serve app.py`
    create_app().servable()
elif __name__ == '__main__':
    # When run directly: `python app.py`
    pn.serve(
        create_app,
        port=int(os.environ.get('PORT', 7860)),
        address='0.0.0.0',
        allow_websocket_origin=['*'],
        title='c13a Dashboard',
        show=False,
    )
