"""
Comparison Tab — IB vs yfinance side-by-side trade comparison.

Data sourced entirely from TradeDB:
  - get_algo_summary(source='ib') vs get_algo_summary(source='yf')
  - get_daily_pnl per source for equity curves
  - Trade-level diff by matching (algo_id, entry_time +/- 5min)
"""

import logging

import panel as pn

logger = logging.getLogger(__name__)


def _comparison_table(state):
    """Per-algo IB vs yf comparison table."""
    def _render(trades_version):
        if not hasattr(state, 'trade_db') or state.trade_db is None:
            return pn.pane.HTML(
                '<div style="color:#888; padding:8px;">No trade database</div>')

        db = state.trade_db
        algos = ['c16', 'c16-dw', 'c16-ml', 'c16-intra', 'c16-oe']
        rows = []

        for algo_id in algos:
            ib_closed = db.get_closed_trades(source='ib', algo_id=algo_id)
            yf_closed = db.get_closed_trades(source='yf', algo_id=algo_id)

            ib_pnl = sum(t.get('pnl', 0) for t in ib_closed)
            yf_pnl = sum(t.get('pnl', 0) for t in yf_closed)
            delta = ib_pnl - yf_pnl

            ib_count = len(ib_closed)
            yf_count = len(yf_closed)

            ib_wins = sum(1 for t in ib_closed if t.get('pnl', 0) > 0)
            yf_wins = sum(1 for t in yf_closed if t.get('pnl', 0) > 0)
            ib_wr = (ib_wins / ib_count * 100) if ib_count > 0 else 0
            yf_wr = (yf_wins / yf_count * 100) if yf_count > 0 else 0

            if ib_count == 0 and yf_count == 0:
                continue

            delta_color = '#00e676' if delta >= 0 else '#ff5252'
            ib_color = '#00e676' if ib_pnl >= 0 else '#ff5252'
            yf_color = '#00e676' if yf_pnl >= 0 else '#ff5252'

            rows.append(f"""
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:6px 12px; color:white;">{algo_id}</td>
                <td style="padding:6px 12px; color:{ib_color};">${ib_pnl:,.0f}</td>
                <td style="padding:6px 12px; color:#aaa;">{ib_count}</td>
                <td style="padding:6px 12px; color:#aaa;">{ib_wr:.0f}%</td>
                <td style="padding:6px 12px; color:{yf_color};">${yf_pnl:,.0f}</td>
                <td style="padding:6px 12px; color:#aaa;">{yf_count}</td>
                <td style="padding:6px 12px; color:#aaa;">{yf_wr:.0f}%</td>
                <td style="padding:6px 12px; color:{delta_color}; font-weight:bold;">${delta:,.0f}</td>
            </tr>""")

        if not rows:
            return pn.pane.HTML(
                '<div style="color:#888; padding:8px; background:#1a1a2e; '
                'border-radius:8px; border:1px solid #333;">No trades to compare</div>')

        html = f"""
        <div style="background:#1a1a2e; border-radius:8px; border:1px solid #333;
                    padding:12px; margin-bottom:8px;">
            <div style="font-size:14px; font-weight:bold; color:white; margin-bottom:8px;">
                IB vs yfinance COMPARISON
            </div>
            <table style="width:100%; border-collapse:collapse; font-family:monospace; font-size:13px;">
                <tr style="border-bottom:2px solid #555; color:#888;">
                    <th style="text-align:left; padding:4px 12px;">Algo</th>
                    <th style="text-align:left; padding:4px 12px;">IB P&L</th>
                    <th style="text-align:left; padding:4px 12px;">IB #</th>
                    <th style="text-align:left; padding:4px 12px;">IB WR</th>
                    <th style="text-align:left; padding:4px 12px;">yf P&L</th>
                    <th style="text-align:left; padding:4px 12px;">yf #</th>
                    <th style="text-align:left; padding:4px 12px;">yf WR</th>
                    <th style="text-align:left; padding:4px 12px;">Delta</th>
                </tr>
                {''.join(rows)}
            </table>
        </div>
        """
        return pn.pane.HTML(html, sizing_mode='stretch_width')

    return pn.bind(_render, trades_version=state.param.trades_version)


def _trade_diff(state):
    """Trade-level diff: trades that diverged between IB and yf."""
    def _render(trades_version):
        if not hasattr(state, 'trade_db') or state.trade_db is None:
            return pn.pane.HTML('')

        db = state.trade_db
        algos = ['c16', 'c16-dw', 'c16-ml', 'c16-intra', 'c16-oe']
        diffs = []

        for algo_id in algos:
            ib_trades = db.get_closed_trades(source='ib', algo_id=algo_id)
            yf_trades = db.get_closed_trades(source='yf', algo_id=algo_id)

            # Match by entry_time (within 5 min)
            yf_matched = set()
            for ib_t in ib_trades:
                ib_entry = ib_t.get('entry_time', '')
                matched = False
                for j, yf_t in enumerate(yf_trades):
                    if j in yf_matched:
                        continue
                    yf_entry = yf_t.get('entry_time', '')
                    if _times_close(ib_entry, yf_entry, 300):
                        yf_matched.add(j)
                        matched = True
                        # Check for P&L divergence
                        ib_pnl = ib_t.get('pnl', 0)
                        yf_pnl = yf_t.get('pnl', 0)
                        if abs(ib_pnl - yf_pnl) > 50:  # >$50 diff
                            diffs.append({
                                'algo': algo_id,
                                'entry': ib_entry,
                                'ib_pnl': ib_pnl,
                                'yf_pnl': yf_pnl,
                                'type': 'pnl_diff',
                            })
                        break
                if not matched:
                    diffs.append({
                        'algo': algo_id,
                        'entry': ib_entry,
                        'ib_pnl': ib_t.get('pnl', 0),
                        'yf_pnl': None,
                        'type': 'ib_only',
                    })

            for j, yf_t in enumerate(yf_trades):
                if j not in yf_matched:
                    diffs.append({
                        'algo': algo_id,
                        'entry': yf_t.get('entry_time', ''),
                        'ib_pnl': None,
                        'yf_pnl': yf_t.get('pnl', 0),
                        'type': 'yf_only',
                    })

        if not diffs:
            return pn.pane.HTML(
                '<div style="color:#888; padding:8px; background:#1a1a2e; '
                'border-radius:8px; border:1px solid #333; margin-bottom:8px;">'
                'No trade divergences detected</div>')

        rows = []
        for d in diffs[:30]:  # Limit to 30
            type_label = {
                'pnl_diff': 'P&L Diff',
                'ib_only': 'IB Only',
                'yf_only': 'yf Only',
            }.get(d['type'], '?')
            type_color = {
                'pnl_diff': '#ffab00',
                'ib_only': '#2196f3',
                'yf_only': '#9c27b0',
            }.get(d['type'], '#888')

            ib_str = f"${d['ib_pnl']:,.0f}" if d['ib_pnl'] is not None else '-'
            yf_str = f"${d['yf_pnl']:,.0f}" if d['yf_pnl'] is not None else '-'

            rows.append(f"""
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:4px 8px; color:{type_color};">{type_label}</td>
                <td style="padding:4px 8px; color:white;">{d['algo']}</td>
                <td style="padding:4px 8px; color:#aaa; font-size:11px;">{d['entry'][:16]}</td>
                <td style="padding:4px 8px;">{ib_str}</td>
                <td style="padding:4px 8px;">{yf_str}</td>
            </tr>""")

        return pn.pane.HTML(f"""
        <details style="margin-bottom:8px;">
            <summary style="font-size:14px; font-weight:bold; color:white; cursor:pointer;
                           padding:8px; background:#1a1a2e; border-radius:8px; border:1px solid #333;">
                TRADE DIVERGENCES ({len(diffs)})
            </summary>
            <div style="background:#1a1a2e; border:1px solid #333; border-radius:0 0 8px 8px;
                        padding:8px; margin-top:-1px;">
                <table style="width:100%; border-collapse:collapse; font-family:monospace; font-size:12px; color:white;">
                    <tr style="border-bottom:2px solid #555; color:#888;">
                        <th style="text-align:left; padding:4px 8px;">Type</th>
                        <th style="text-align:left; padding:4px 8px;">Algo</th>
                        <th style="text-align:left; padding:4px 8px;">Entry</th>
                        <th style="text-align:left; padding:4px 8px;">IB P&L</th>
                        <th style="text-align:left; padding:4px 8px;">yf P&L</th>
                    </tr>
                    {''.join(rows)}
                </table>
            </div>
        </details>
        """, sizing_mode='stretch_width')

    return pn.bind(_render, trades_version=state.param.trades_version)


def _times_close(t1: str, t2: str, max_seconds: int) -> bool:
    """Check if two ISO timestamps are within max_seconds of each other."""
    if not t1 or not t2:
        return False
    try:
        from datetime import datetime
        dt1 = datetime.fromisoformat(t1)
        dt2 = datetime.fromisoformat(t2)
        # Handle timezone-aware vs naive
        if dt1.tzinfo and not dt2.tzinfo:
            dt2 = dt2.replace(tzinfo=dt1.tzinfo)
        elif dt2.tzinfo and not dt1.tzinfo:
            dt1 = dt1.replace(tzinfo=dt2.tzinfo)
        return abs((dt1 - dt2).total_seconds()) <= max_seconds
    except (ValueError, TypeError):
        return False


def comparison_tab(state):
    """Build the Comparison tab."""
    return pn.Column(
        _comparison_table(state),
        _trade_diff(state),
        sizing_mode='stretch_width',
    )
