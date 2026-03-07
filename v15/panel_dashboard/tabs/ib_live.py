"""
IB Live Tab — Real-time IB trading view with P&L, positions, order entry.

Layout (top to bottom):
  - Price banner (tick-driven)
  - Per-algo P&L summary with on/off toggles
  - Open positions with live P&L and trailing stop visualization
  - Manual order entry panel
  - Market insights (CS analysis)
  - Trade history (from DB)
"""

import logging
from datetime import datetime

import panel as pn
import param

logger = logging.getLogger(__name__)


def _price_banner(state):
    """Live price banner bound to tsla_price param."""
    def _render(tsla_price, price_delta, price_source, ib_connected):
        pct = (price_delta / (tsla_price - price_delta) * 100
               if tsla_price > 0 and (tsla_price - price_delta) > 0 else 0)
        arrow = '\u25b2' if price_delta >= 0 else '\u25bc'
        color = '#00e676' if price_delta >= 0 else '#ff5252'
        source_color = '#00e676' if price_source == 'IB LIVE' else '#ff5252'
        source_dot = '\u25cf'

        return pn.pane.HTML(f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:12px 20px; background:#1a1a2e; border-radius:8px;
                    border:1px solid #333; margin-bottom:8px;">
            <div>
                <span style="font-size:24px; font-weight:bold; color:white;">
                    TSLA ${tsla_price:,.2f}
                </span>
                <span style="font-size:16px; color:{color}; margin-left:12px;">
                    {arrow} ${abs(price_delta):,.2f} ({pct:+.1f}%)
                </span>
            </div>
            <div style="display:flex; align-items:center; gap:16px;">
                <span style="color:{source_color}; font-size:14px;">
                    {source_dot} {price_source}
                </span>
            </div>
        </div>
        """, sizing_mode='stretch_width')

    return pn.bind(_render,
                   tsla_price=state.param.tsla_price,
                   price_delta=state.param.price_delta,
                   price_source=state.param.price_source,
                   ib_connected=state.param.ib_connected)


def _algo_pnl_summary(state):
    """Per-algo P&L summary with on/off toggles. Bound to positions_version."""
    def _render(positions_version):
        if not hasattr(state, 'trade_db') or state.trade_db is None:
            return pn.pane.HTML(
                '<div style="color:#888; padding:8px;">No trade database</div>')

        db = state.trade_db
        rows_html = []

        # Get algo summaries for IB source
        algos = ['c16', 'c16-dw', 'c16-ml', 'c16-intra', 'c16-oe', 'manual']
        for algo_id in algos:
            open_trades = db.get_open_trades(source='ib', algo_id=algo_id)
            closed = db.get_closed_trades(source='ib', algo_id=algo_id)

            total_pnl = sum(t.get('pnl', 0) for t in closed)
            day_pnl = 0  # TODO: filter by today's date
            n_trades = len(closed)
            n_open = len(open_trades)
            wins = sum(1 for t in closed if t.get('pnl', 0) > 0)
            wr = (wins / n_trades * 100) if n_trades > 0 else 0

            # Live unrealized P&L
            unrealized = 0
            if state.tsla_price > 0:
                for t in open_trades:
                    entry = t.get('entry_price', 0)
                    shares = t.get('open_shares', t.get('shares', 0))
                    if t.get('direction', 'long') == 'long':
                        unrealized += (state.tsla_price - entry) * shares
                    else:
                        unrealized += (entry - state.tsla_price) * shares

            pnl_color = '#00e676' if total_pnl >= 0 else '#ff5252'
            unr_color = '#00e676' if unrealized >= 0 else '#ff5252'

            # Enabled toggle state
            enabled = True
            if hasattr(state, 'ib_scanner_manager') and state.ib_scanner_manager:
                adapter = state.ib_scanner_manager.get_adapter(algo_id)
                if adapter:
                    enabled = adapter.enabled

            dot = '\u25cf' if enabled else '\u25cb'
            dot_color = '#00e676' if enabled else '#666'

            rows_html.append(f"""
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:6px 12px; color:{dot_color};">{dot} {algo_id}</td>
                <td style="padding:6px 12px; color:{pnl_color};">${total_pnl:,.0f}</td>
                <td style="padding:6px 12px; color:{unr_color};">${unrealized:,.0f}</td>
                <td style="padding:6px 12px; color:#aaa;">{n_trades}</td>
                <td style="padding:6px 12px; color:#aaa;">{n_open}</td>
                <td style="padding:6px 12px; color:#aaa;">{wr:.0f}%</td>
            </tr>""")

        html = f"""
        <div style="background:#1a1a2e; border-radius:8px; border:1px solid #333;
                    padding:12px; margin-bottom:8px;">
            <div style="font-size:14px; font-weight:bold; color:white; margin-bottom:8px;">
                IB ALGO P&L SUMMARY
            </div>
            <table style="width:100%; border-collapse:collapse; font-family:monospace; font-size:13px;">
                <tr style="border-bottom:2px solid #555; color:#888;">
                    <th style="text-align:left; padding:4px 12px;">Algo</th>
                    <th style="text-align:left; padding:4px 12px;">Total P&L</th>
                    <th style="text-align:left; padding:4px 12px;">Unrealized</th>
                    <th style="text-align:left; padding:4px 12px;">Trades</th>
                    <th style="text-align:left; padding:4px 12px;">Open</th>
                    <th style="text-align:left; padding:4px 12px;">WR</th>
                </tr>
                {''.join(rows_html)}
            </table>
        </div>
        """
        return pn.pane.HTML(html, sizing_mode='stretch_width')

    return pn.bind(_render, positions_version=state.param.positions_version)


def _open_positions(state):
    """Open IB positions with live P&L. Bound to positions_version."""
    def _render(positions_version):
        if not hasattr(state, 'trade_db') or state.trade_db is None:
            return pn.pane.HTML('')

        trades = state.trade_db.get_open_trades(source='ib')
        if not trades:
            return pn.pane.HTML(
                '<div style="color:#888; padding:8px; background:#1a1a2e; '
                'border-radius:8px; border:1px solid #333; margin-bottom:8px;">'
                'No open IB positions</div>')

        cards = []
        for t in trades:
            entry = t.get('entry_price', 0)
            shares = t.get('open_shares', t.get('shares', 0))
            direction = t.get('direction', 'long')
            stop = t.get('stop_price', 0)
            tp = t.get('tp_price', 0)
            best = t.get('best_price', entry)
            algo = t.get('algo_id', '?')
            fill_status = t.get('ib_fill_status', 'filled')

            # P&L
            price = state.tsla_price if state.tsla_price > 0 else entry
            if direction == 'long':
                pnl = (price - entry) * shares
                pnl_pct = ((price - entry) / entry * 100) if entry > 0 else 0
            else:
                pnl = (entry - price) * shares
                pnl_pct = ((entry - price) / entry * 100) if entry > 0 else 0

            pnl_color = '#00e676' if pnl >= 0 else '#ff5252'
            dir_color = '#00e676' if direction == 'long' else '#ff5252'
            dir_arrow = '\u25b2' if direction == 'long' else '\u25bc'

            status_badge = ''
            if fill_status == 'pending':
                status_badge = '<span style="background:#ffab00; color:black; padding:2px 6px; border-radius:4px; font-size:11px;">PENDING</span>'
            elif fill_status == 'partial':
                status_badge = '<span style="background:#ff9800; color:black; padding:2px 6px; border-radius:4px; font-size:11px;">PARTIAL</span>'

            cards.append(f"""
            <div style="background:#1a1a2e; border:1px solid #333; border-radius:8px;
                        padding:12px; margin-bottom:6px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="color:{dir_color}; font-weight:bold;">
                            {dir_arrow} {direction.upper()}
                        </span>
                        <span style="color:#aaa; margin-left:8px;">{algo}</span>
                        {status_badge}
                    </div>
                    <div style="color:{pnl_color}; font-weight:bold; font-size:16px;">
                        ${pnl:,.0f} ({pnl_pct:+.1f}%)
                    </div>
                </div>
                <div style="display:flex; gap:20px; margin-top:6px; color:#888; font-size:12px;">
                    <span>Entry: ${entry:,.2f}</span>
                    <span>Shares: {shares}</span>
                    <span>Stop: ${stop:,.2f}</span>
                    <span>TP: ${tp:,.2f}</span>
                    <span>Best: ${best:,.2f}</span>
                </div>
            </div>
            """)

        html = f"""
        <div style="margin-bottom:8px;">
            <div style="font-size:14px; font-weight:bold; color:white; margin-bottom:6px;">
                OPEN POSITIONS ({len(trades)})
            </div>
            {''.join(cards)}
        </div>
        """
        return pn.pane.HTML(html, sizing_mode='stretch_width')

    return pn.bind(_render, positions_version=state.param.positions_version)


def _trade_history(state):
    """Recent trade history from DB. Bound to trades_version."""
    def _render(trades_version):
        if not hasattr(state, 'trade_db') or state.trade_db is None:
            return pn.pane.HTML('')

        trades = state.trade_db.get_closed_trades(source='ib', limit=50)
        if not trades:
            return pn.pane.HTML(
                '<div style="color:#888; padding:8px;">No trade history</div>')

        rows = []
        for t in trades:
            pnl = t.get('pnl', 0)
            pnl_color = '#00e676' if pnl >= 0 else '#ff5252'
            direction = t.get('direction', 'long')
            dir_arrow = '\u25b2' if direction == 'long' else '\u25bc'

            entry_time = t.get('entry_time', '')
            if entry_time:
                try:
                    dt = datetime.fromisoformat(entry_time)
                    entry_time = dt.strftime('%m/%d %H:%M')
                except (ValueError, TypeError):
                    pass

            rows.append(f"""
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:4px 8px; color:#aaa; font-size:11px;">{entry_time}</td>
                <td style="padding:4px 8px;">{t.get('algo_id', '?')}</td>
                <td style="padding:4px 8px;">{dir_arrow} {direction}</td>
                <td style="padding:4px 8px;">${t.get('entry_price', 0):,.2f}</td>
                <td style="padding:4px 8px;">${t.get('exit_price', 0):,.2f}</td>
                <td style="padding:4px 8px; color:{pnl_color};">${pnl:,.0f}</td>
                <td style="padding:4px 8px; color:#aaa;">{t.get('exit_reason', '')}</td>
            </tr>""")

        html = f"""
        <details style="margin-bottom:8px;">
            <summary style="font-size:14px; font-weight:bold; color:white; cursor:pointer;
                           padding:8px; background:#1a1a2e; border-radius:8px; border:1px solid #333;">
                TRADE HISTORY ({len(trades)} recent)
            </summary>
            <div style="background:#1a1a2e; border:1px solid #333; border-radius:0 0 8px 8px;
                        padding:8px; margin-top:-1px;">
                <table style="width:100%; border-collapse:collapse; font-family:monospace; font-size:12px; color:white;">
                    <tr style="border-bottom:2px solid #555; color:#888;">
                        <th style="text-align:left; padding:4px 8px;">Time</th>
                        <th style="text-align:left; padding:4px 8px;">Algo</th>
                        <th style="text-align:left; padding:4px 8px;">Dir</th>
                        <th style="text-align:left; padding:4px 8px;">Entry</th>
                        <th style="text-align:left; padding:4px 8px;">Exit</th>
                        <th style="text-align:left; padding:4px 8px;">P&L</th>
                        <th style="text-align:left; padding:4px 8px;">Reason</th>
                    </tr>
                    {''.join(rows)}
                </table>
            </div>
        </details>
        """
        return pn.pane.HTML(html, sizing_mode='stretch_width')

    return pn.bind(_render, trades_version=state.param.trades_version)


def _degraded_banner(state):
    """Warning banner for ib_degraded or migration_failed states."""
    def _render(ib_connected):
        parts = []
        if hasattr(state, 'migration_failed') and state.migration_failed:
            parts.append("""
            <div style="background:#ff5252; color:white; padding:12px; border-radius:8px;
                        margin-bottom:8px; font-weight:bold;">
                MIGRATION FAILED — Scanner loops blocked. Fix the issue and restart.
            </div>""")
        if hasattr(state, 'ib_degraded') and state.ib_degraded:
            parts.append("""
            <div style="background:#ff9800; color:black; padding:12px; border-radius:8px;
                        margin-bottom:8px; font-weight:bold;">
                IB/DB MISMATCH — Automated entries paused. Review positions and re-reconcile.
            </div>""")
        if not ib_connected:
            parts.append("""
            <div style="background:#ff5252; color:white; padding:8px; border-radius:8px;
                        margin-bottom:8px;">
                IB DISCONNECTED — No live price source
            </div>""")
        if not parts:
            return pn.pane.HTML('')
        return pn.pane.HTML('\n'.join(parts), sizing_mode='stretch_width')

    return pn.bind(_render, ib_connected=state.param.ib_connected)


def _kill_switch_panel(state):
    """Kill All button with confirmation."""
    kill_btn = pn.widgets.Button(
        name='KILL ALL', button_type='danger', width=120)
    status = pn.pane.HTML('', width=200)

    def _on_kill(event):
        if hasattr(state, 'ib_scanner_manager') and state.ib_scanner_manager:
            if state.kill_switch:
                state.ib_scanner_manager.unkill()
                state.kill_switch = False
                kill_btn.name = 'KILL ALL'
                kill_btn.button_type = 'danger'
                status.object = '<span style="color:#00e676;">Algos re-enabled</span>'
            else:
                state.ib_scanner_manager.kill_all()
                state.kill_switch = True
                kill_btn.name = 'UNKILL'
                kill_btn.button_type = 'warning'
                status.object = '<span style="color:#ff5252;">All algos KILLED</span>'

    kill_btn.on_click(_on_kill)
    return pn.Row(kill_btn, status)


def ib_live_tab(state):
    """Build the IB Live tab. Returns a Panel Column."""
    components = [
        _degraded_banner(state),
        _price_banner(state),
        _kill_switch_panel(state),
        _algo_pnl_summary(state),
        _open_positions(state),
    ]

    # Order entry panel (if available)
    try:
        from v15.panel_dashboard.order_entry import order_entry_panel
        components.append(order_entry_panel(state))
    except ImportError:
        pass

    components.append(_trade_history(state))

    return pn.Column(*components, sizing_mode='stretch_width')
