"""Model Comparisons tab — compares live performance across all scanner models."""

import logging

import panel as pn
import numpy as np

logger = logging.getLogger(__name__)


def model_comparisons_tab(state, prefix='') -> pn.Column:
    """Build the Model Comparisons tab, bound to state.model_data_version.

    prefix='': show IB scanners (keys not starting with 'yf-')
    prefix='yf-': show yfinance A/B scanners only
    """
    title = "yfinance A/B Comparisons" if prefix == 'yf-' else "Model Comparisons"
    subtitle = ("Compares yfinance-backed scanners against IB scanners. "
                "Same configs, different data source." if prefix == 'yf-'
                else "Compares live performance across all scanner models. "
                     "Refreshes every hour.")
    parts = [pn.pane.Markdown(f"## {title}\n{subtitle}")]

    # Live status panel for yfinance A/B tab
    if prefix == 'yf-':
        parts.append(pn.bind(_yf_live_status, state.param.yf_status_version,
                             state=state))

    parts.append(pn.bind(_model_content, state.param.model_data_version,
                         model_data=state.model_data, prefix=prefix))

    return pn.Column(*parts, sizing_mode='stretch_width')


def _yf_live_status(version, state=None):
    """Live status banner for yfinance A/B tab — shows current signal, price, scanner states."""
    yf_status = state.yf_status if state else None
    if not yf_status:
        return pn.pane.HTML(
            '<div style="background:#1a1a2e;border:1px solid #333;border-radius:8px;'
            'padding:12px;margin-bottom:12px;color:#888;">'
            'Waiting for first yfinance analysis cycle (runs every 150s)...</div>',
            sizing_mode='stretch_width',
        )

    price = yf_status.get('price', 0)
    sig_action = yf_status.get('signal', 'HOLD')
    confidence = yf_status.get('confidence', 0)
    primary_tf = yf_status.get('primary_tf', '?')
    reason = yf_status.get('reason', '')
    update_time = yf_status.get('time', '?')
    data_bars = yf_status.get('data_bars', 0)
    dw_sig = yf_status.get('dw_signal')

    # Signal color
    sig_colors = {'BUY': '#00e676', 'SELL': '#ff5252', 'HOLD': '#888'}
    sig_color = sig_colors.get(sig_action, '#888')

    # DW signal line
    dw_html = ''
    if dw_sig:
        dw_color = sig_colors.get(dw_sig['action'], '#888')
        dw_html = (f'<span style="margin-left:24px;">DW: '
                   f'<b style="color:{dw_color}">{dw_sig["action"]}</b> '
                   f'{dw_sig["confidence"]:.0%} ({dw_sig["primary_tf"]})</span>')

    # Per-scanner status rows
    scanner_cells = []
    for s in yf_status.get('scanners', []):
        tag = s['tag']
        n_pos = s['positions']
        n_closed = s['closed']
        dpnl = s['daily_pnl']
        dpnl_color = '#00e676' if dpnl >= 0 else '#ff5252'
        last = s.get('last_signal')
        if last:
            ls_color = sig_colors.get(last['action'], '#888')
            ls_text = (f'<span style="color:{ls_color}">{last["action"]}</span> '
                       f'{last["confidence"]:.0%} ({last.get("signal_source", "?")})')
        else:
            ls_text = '<span style="color:#555">no signals yet</span>'
        scanner_cells.append(
            f'<tr style="border-bottom:1px solid #2a2a3e;">'
            f'<td style="padding:3px 8px;font-weight:500;">{tag}</td>'
            f'<td style="padding:3px 8px;">{ls_text}</td>'
            f'<td style="padding:3px 8px;">{n_pos} open</td>'
            f'<td style="padding:3px 8px;">{n_closed} closed</td>'
            f'<td style="padding:3px 8px;color:{dpnl_color}">${dpnl:+,.0f}</td>'
            f'</tr>'
        )

    scanner_table = ''
    if scanner_cells:
        scanner_table = (
            '<table style="width:100%;font-size:12px;color:#aaa;border-collapse:collapse;'
            'margin-top:8px;">'
            '<thead><tr style="color:#666;border-bottom:1px solid #333;">'
            '<th style="text-align:left;padding:2px 8px;">Scanner</th>'
            '<th style="text-align:left;padding:2px 8px;">Last Signal</th>'
            '<th style="text-align:left;padding:2px 8px;">Positions</th>'
            '<th style="text-align:left;padding:2px 8px;">Trades</th>'
            '<th style="text-align:left;padding:2px 8px;">Daily P&L</th>'
            '</tr></thead>'
            f'<tbody>{"".join(scanner_cells)}</tbody></table>'
        )

    html = f"""
    <div style="background:#1a1a2e;border:1px solid #333;border-radius:8px;
                padding:12px;margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
            <div>
                <span style="color:#666;font-size:11px;">YF PRICE</span><br>
                <span style="color:#fff;font-size:18px;font-weight:600;">
                    ${price:,.2f}</span>
            </div>
            <div>
                <span style="color:#666;font-size:11px;">CS-5TF SIGNAL</span><br>
                <span style="color:{sig_color};font-size:18px;font-weight:600;">
                    {sig_action}</span>
                <span style="color:#aaa;font-size:13px;">
                    {confidence:.0%} ({primary_tf})</span>
            </div>
            <div>
                <span style="color:#666;font-size:11px;">LAST ANALYSIS</span><br>
                <span style="color:#ccc;font-size:14px;">{update_time}</span>
                <span style="color:#666;font-size:11px;margin-left:4px;">
                    ({data_bars} bars)</span>
            </div>
            {f'<div><span style="color:#666;font-size:11px;">CS-DW</span><br>{dw_html}</div>' if dw_html else ''}
        </div>
        {f'<div style="color:#666;font-size:11px;margin-top:6px;">Reason: {reason}</div>' if reason else ''}
        {scanner_table}
    </div>
    """
    return pn.pane.HTML(html, sizing_mode='stretch_width')


def _model_content(version, model_data=None, prefix=''):
    if not model_data:
        return pn.pane.HTML(
            '<div style="color:#888;padding:16px;">No model data loaded yet. '
            'Scanner state files will appear after the first trade.</div>',
            sizing_mode='stretch_width',
        )

    # Filter to model keys by prefix
    if prefix:
        model_keys = [k for k in model_data if not k.startswith('_') and k.startswith(prefix)]
    else:
        model_keys = [k for k in model_data if not k.startswith('_') and not k.startswith('yf-')]
    if not model_keys:
        return pn.pane.HTML(
            '<div style="color:#888;padding:16px;">No model data found. '
            'Start a live scanner session to populate it.</div>',
            sizing_mode='stretch_width',
        )

    last_updated = model_data.get('_last_updated', 'unknown')

    # Compute stats per model
    stats = {k: _compute_stats(model_data[k]) for k in model_keys}

    # Summary table
    summary_rows = []
    for tag, s in stats.items():
        pnl_color = "#00e676" if s["total_pnl"] >= 0 else "#ff5252"
        has_trades = s["n_trades"] > 0
        wr_td = f'<td>{s["win_rate"]:.1%}</td>' if has_trades else '<td>&mdash;</td>'
        avg_td = f'<td>${s["avg_pnl"]:+,.0f}</td>' if has_trades else '<td>&mdash;</td>'
        best_td = f'<td style="color:#00e676">${s["best_trade"]:+,.0f}</td>' if has_trades else '<td>&mdash;</td>'
        worst_td = f'<td style="color:#ff5252">${s["worst_trade"]:+,.0f}</td>' if has_trades else '<td>&mdash;</td>'
        hold_td = f'<td>{s["avg_hold_min"]:.0f}m</td>' if has_trades else '<td>&mdash;</td>'
        summary_rows.append(
            f'<tr>'
            f'<td style="font-weight:600;">{tag}</td>'
            f'<td>${s["equity"]:,.0f}</td>'
            f'<td style="color:{pnl_color}">${s["total_pnl"]:+,.0f}</td>'
            f'<td>{s["n_trades"]}</td>'
            f'{wr_td}{avg_td}{best_td}{worst_td}{hold_td}'
            f'<td>{s["open_positions"]}</td>'
            f'</tr>'
        )

    summary_html = f"""
    <div style="font-size:12px;color:#888;margin-bottom:8px;">Last updated: {last_updated}</div>
    <div style="overflow-x:auto;">
        <table style="width:100%;font-size:13px;color:#ccc;border-collapse:collapse;">
            <thead><tr style="border-bottom:2px solid #444;color:#888;">
                <th>Model</th><th>Equity</th><th>Total P&L</th><th>Trades</th>
                <th>Win Rate</th><th>Avg P&L</th><th>Best</th><th>Worst</th>
                <th>Avg Hold</th><th>Open</th>
            </tr></thead>
            <tbody>{''.join(summary_rows)}</tbody>
        </table>
    </div>
    """

    # Equity curves
    equity_pane = _equity_curves(stats)

    # Recent trades
    recent_html = _recent_trades_html(stats, model_keys)

    # Open positions
    open_html = _open_positions_html(model_data, model_keys)

    return pn.Column(
        pn.pane.HTML(summary_html, sizing_mode='stretch_width'),
        equity_pane,
        pn.pane.HTML(recent_html, sizing_mode='stretch_width'),
        pn.pane.HTML(open_html, sizing_mode='stretch_width'),
        sizing_mode='stretch_width',
    )


def _compute_stats(mdata: dict) -> dict:
    trades = mdata.get('closed_trades', [])
    equity = mdata.get('equity', 100_000.0)
    positions = mdata.get('positions', {})

    if not trades:
        return {
            'equity': equity, 'total_pnl': 0.0, 'n_trades': 0,
            'win_rate': 0.0, 'avg_pnl': 0.0, 'best_trade': 0.0,
            'worst_trade': 0.0, 'avg_hold_min': 0.0,
            'open_positions': len(positions), 'trades': [],
        }

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    holds = [t.get('hold_minutes', 0) for t in trades]

    return {
        'equity': equity,
        'total_pnl': sum(pnls),
        'n_trades': len(trades),
        'win_rate': len(wins) / len(pnls),
        'avg_pnl': sum(pnls) / len(pnls),
        'best_trade': max(pnls),
        'worst_trade': min(pnls),
        'avg_hold_min': sum(holds) / len(holds) if holds else 0.0,
        'open_positions': len(positions),
        'trades': trades,
    }


def _equity_curves(stats):
    any_trades = any(s['n_trades'] > 0 for s in stats.values())
    if not any_trades:
        return pn.pane.HTML('')

    try:
        import plotly.graph_objects as go
    except ImportError:
        return pn.pane.HTML('<div style="color:#888;">Plotly not available for equity curves</div>')

    fig = go.Figure()
    colors = ['#00c853', '#2196f3', '#ff9800', '#e91e63', '#9c27b0']

    for i, (tag, s) in enumerate(stats.items()):
        trades = sorted(s['trades'], key=lambda t: t.get('exit_time', ''))
        if not trades:
            continue
        initial = 100_000.0
        times, equity_vals = [], []
        running = initial
        for t in trades:
            running += t['pnl']
            times.append(t.get('exit_time', ''))
            equity_vals.append(running)
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=times, y=equity_vals, name=tag,
            mode='lines+markers', line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate=f'<b>{tag}</b><br>%{{x}}<br>Equity: $%{{y:,.0f}}<extra></extra>',
        ))

    fig.update_layout(
        title='Equity Curves',
        template='plotly_dark',
        xaxis_title='Exit Time',
        yaxis_title='Equity ($)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=60, r=20, t=40, b=40),
        height=400,
    )

    return pn.pane.Plotly(fig, sizing_mode='stretch_width')


def _recent_trades_html(stats, model_keys) -> str:
    columns_html = []
    for tag in model_keys:
        s = stats[tag]
        trades = sorted(s['trades'], key=lambda t: t.get('exit_time', ''), reverse=True)[:10]

        if not trades:
            column_content = '<div style="color:#888;font-size:12px;">No trades yet</div>'
        else:
            trade_items = []
            for t in trades:
                pnl = t['pnl']
                color = '#00c853' if pnl >= 0 else '#ff5252'
                trade_items.append(
                    f'<div style="font-size:12px;border-left:3px solid {color};'
                    f'padding:3px 8px;margin:2px 0;">'
                    f'<b style="color:{color}">${pnl:+,.0f}</b> '
                    f'<span style="color:#888">{t.get("exit_reason", "?")} &middot; '
                    f'{t.get("hold_minutes", 0):.0f}m</span></div>'
                )
            column_content = ''.join(trade_items)

        columns_html.append(
            f'<div style="flex:1;min-width:150px;">'
            f'<div style="font-weight:600;color:#ccc;margin-bottom:4px;">{tag}</div>'
            f'{column_content}'
            f'</div>'
        )

    return f"""
    <div style="margin-top:16px;">
        <div style="font-size:14px;font-weight:600;color:#ccc;margin-bottom:8px;">Recent Trades</div>
        <div style="display:flex;gap:16px;flex-wrap:wrap;">
            {''.join(columns_html)}
        </div>
    </div>
    """


def _open_positions_html(model_data, model_keys) -> str:
    open_models = [
        (tag, model_data[tag].get('positions', {}))
        for tag in model_keys
        if model_data[tag].get('positions')
    ]

    if not open_models:
        return ''

    sections = []
    for tag, positions in open_models:
        pos_items = []
        for pos_id, pos in positions.items():
            direction_icon = '&#128994;' if pos.get('direction') == 'long' else '&#128308;'
            pos_items.append(
                f'<div style="font-size:12px;color:#aaa;padding:2px 0;">'
                f'{direction_icon} {pos_id} | Entry ${pos.get("entry_price", 0):.2f} | '
                f'TP ${pos.get("tp_price", 0):.2f} | Stop ${pos.get("stop_price", 0):.2f} | '
                f'Notional ${pos.get("notional", 0):,.0f}</div>'
            )
        sections.append(
            f'<div style="margin:4px 0;">'
            f'<div style="font-weight:600;color:#ccc;">{tag} &mdash; {len(positions)} open</div>'
            f'{"".join(pos_items)}'
            f'</div>'
        )

    return f"""
    <div style="margin-top:16px;">
        <div style="font-size:14px;font-weight:600;color:#ccc;margin-bottom:8px;">Open Positions</div>
        {''.join(sections)}
    </div>
    """
