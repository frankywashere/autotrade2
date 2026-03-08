"""
yfinance Simulation Tab — Hypothetical trade tracking using yfinance prices.

All trades logged to DB with source='yf'. No IB orders placed.
Runs CS-DW + OE-Sig5 via a second LiveEngine fed by YfinanceDataProvider.
"""

import logging
from datetime import datetime

import panel as pn

logger = logging.getLogger(__name__)


def _yf_price_banner(state):
    """yfinance price banner."""
    def _render(positions_version):
        yf_price = 0
        if hasattr(state, 'price_manager') and state.price_manager:
            yf_price = state.price_manager.get_price('TSLA', 'yf')
        if yf_price <= 0:
            yf_price = state.tsla_price  # fallback to IB price for display

        return pn.pane.HTML(f"""
        <div style="display:flex; align-items:center; justify-content:space-between;
                    padding:12px 20px; background:#1a1a2e; border-radius:8px;
                    border:1px solid #333; margin-bottom:8px;">
            <div>
                <span style="font-size:20px; font-weight:bold; color:white;">
                    TSLA ${yf_price:,.2f}
                </span>
                <span style="font-size:14px; color:#888; margin-left:8px;">(yfinance)</span>
            </div>
        </div>
        """, sizing_mode='stretch_width')

    return pn.bind(_render, positions_version=state.param.positions_version)


def _yf_algo_summary(state):
    """Per-algo P&L summary for yfinance source."""
    def _render(positions_version):
        if not hasattr(state, 'trade_db') or state.trade_db is None:
            return pn.pane.HTML(
                '<div style="color:#888; padding:8px;">No trade database</div>')

        db = state.trade_db
        rows_html = []

        algos = ['yf-dw', 'yf-oe']
        for algo_id in algos:
            open_trades = db.get_open_trades(source='yf', algo_id=algo_id)
            closed = db.get_closed_trades(source='yf', algo_id=algo_id)

            total_pnl = sum(t.get('pnl', 0) for t in closed)
            n_trades = len(closed)
            n_open = len(open_trades)
            wins = sum(1 for t in closed if t.get('pnl', 0) > 0)
            wr = (wins / n_trades * 100) if n_trades > 0 else 0

            if n_trades == 0 and n_open == 0:
                continue  # Skip algos with no activity

            pnl_color = '#00e676' if total_pnl >= 0 else '#ff5252'

            rows_html.append(f"""
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:6px 12px; color:white;">{algo_id}</td>
                <td style="padding:6px 12px; color:{pnl_color};">${total_pnl:,.0f}</td>
                <td style="padding:6px 12px; color:#aaa;">{n_trades}</td>
                <td style="padding:6px 12px; color:#aaa;">{n_open}</td>
                <td style="padding:6px 12px; color:#aaa;">{wr:.0f}%</td>
            </tr>""")

        if not rows_html:
            return pn.pane.HTML(
                '<div style="color:#888; padding:8px; background:#1a1a2e; '
                'border-radius:8px; border:1px solid #333;">No yfinance trades yet</div>')

        html = f"""
        <div style="background:#1a1a2e; border-radius:8px; border:1px solid #333;
                    padding:12px; margin-bottom:8px;">
            <div style="font-size:14px; font-weight:bold; color:white; margin-bottom:8px;">
                SIMULATION P&L (yfinance)
            </div>
            <table style="width:100%; border-collapse:collapse; font-family:monospace; font-size:13px;">
                <tr style="border-bottom:2px solid #555; color:#888;">
                    <th style="text-align:left; padding:4px 12px;">Algo</th>
                    <th style="text-align:left; padding:4px 12px;">Total P&L</th>
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


def _yf_open_positions(state):
    """Open yf positions."""
    def _render(positions_version):
        if not hasattr(state, 'trade_db') or state.trade_db is None:
            return pn.pane.HTML('')

        trades = state.trade_db.get_open_trades(source='yf')
        if not trades:
            return pn.pane.HTML(
                '<div style="color:#888; padding:8px; background:#1a1a2e; '
                'border-radius:8px; border:1px solid #333; margin-bottom:8px;">'
                'No open yfinance positions</div>')

        yf_price = state.tsla_price
        if hasattr(state, 'price_manager') and state.price_manager:
            p = state.price_manager.get_price('TSLA', 'yf')
            if p > 0:
                yf_price = p

        cards = []
        for t in trades:
            entry = t.get('entry_price', 0)
            shares = t.get('open_shares', t.get('shares', 0))
            direction = t.get('direction', 'long')
            algo = t.get('algo_id', '?')

            price = yf_price if yf_price > 0 else entry
            if direction == 'long':
                pnl = (price - entry) * shares
                pnl_pct = ((price - entry) / entry * 100) if entry > 0 else 0
            else:
                pnl = (entry - price) * shares
                pnl_pct = ((entry - price) / entry * 100) if entry > 0 else 0

            pnl_color = '#00e676' if pnl >= 0 else '#ff5252'
            dir_arrow = '\u25b2' if direction == 'long' else '\u25bc'

            cards.append(f"""
            <div style="background:#1a1a2e; border:1px solid #333; border-radius:8px;
                        padding:10px; margin-bottom:4px;">
                <div style="display:flex; justify-content:space-between;">
                    <span>{dir_arrow} {direction.upper()} <span style="color:#888;">{algo}</span></span>
                    <span style="color:{pnl_color}; font-weight:bold;">${pnl:,.0f} ({pnl_pct:+.1f}%)</span>
                </div>
                <div style="color:#888; font-size:11px; margin-top:4px;">
                    Entry: ${entry:,.2f} | Shares: {shares} | Stop: ${t.get('stop_price', 0):,.2f}
                </div>
            </div>""")

        return pn.pane.HTML(
            f'<div style="margin-bottom:8px;">'
            f'<div style="font-size:14px; font-weight:bold; color:white; margin-bottom:6px;">'
            f'OPEN POSITIONS ({len(trades)})</div>'
            f'{"".join(cards)}</div>',
            sizing_mode='stretch_width')

    return pn.bind(_render, positions_version=state.param.positions_version)


def _yf_trade_history(state):
    """yf trade history."""
    def _render(trades_version):
        if not hasattr(state, 'trade_db') or state.trade_db is None:
            return pn.pane.HTML('')

        trades = state.trade_db.get_closed_trades(source='yf', limit=50)
        if not trades:
            return pn.pane.HTML(
                '<div style="color:#888; padding:8px;">No yfinance trade history</div>')

        rows = []
        for t in trades:
            pnl = t.get('pnl', 0)
            pnl_color = '#00e676' if pnl >= 0 else '#ff5252'
            dir_arrow = '\u25b2' if t.get('direction', 'long') == 'long' else '\u25bc'

            entry_time = t.get('entry_time', '')
            try:
                dt = datetime.fromisoformat(entry_time)
                entry_time = dt.strftime('%m/%d %H:%M')
            except (ValueError, TypeError):
                pass

            rows.append(f"""
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:4px 8px; color:#aaa; font-size:11px;">{entry_time}</td>
                <td style="padding:4px 8px;">{t.get('algo_id', '?')}</td>
                <td style="padding:4px 8px;">{dir_arrow}</td>
                <td style="padding:4px 8px;">${t.get('entry_price', 0):,.2f}</td>
                <td style="padding:4px 8px;">${t.get('exit_price', 0):,.2f}</td>
                <td style="padding:4px 8px; color:{pnl_color};">${pnl:,.0f}</td>
            </tr>""")

        return pn.pane.HTML(f"""
        <details style="margin-bottom:8px;">
            <summary style="font-size:14px; font-weight:bold; color:white; cursor:pointer;
                           padding:8px; background:#1a1a2e; border-radius:8px; border:1px solid #333;">
                TRADE HISTORY ({len(trades)} recent)
            </summary>
            <div style="background:#1a1a2e; border:1px solid #333; border-radius:0 0 8px 8px;
                        padding:8px; margin-top:-1px;">
                <table style="width:100%; border-collapse:collapse; font-family:monospace; font-size:12px; color:white;">
                    {''.join(rows)}
                </table>
            </div>
        </details>
        """, sizing_mode='stretch_width')

    return pn.bind(_render, trades_version=state.param.trades_version)


def yf_sim_tab(state):
    """Build the yfinance Simulation tab."""
    return pn.Column(
        _yf_price_banner(state),
        _yf_algo_summary(state),
        _yf_open_positions(state),
        _yf_trade_history(state),
        sizing_mode='stretch_width',
    )
