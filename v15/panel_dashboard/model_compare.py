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
    return pn.Column(
        pn.pane.Markdown(f"## {title}\n{subtitle}"),
        pn.bind(_model_content, state.param.model_data_version,
                model_data=state.model_data, prefix=prefix),
        sizing_mode='stretch_width',
    )


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
