"""
IB Live Tab — Real-time IB trading view with P&L, positions, order entry.

Layout (top to bottom):
  - Price banner (tick-driven)
  - Per-algo P&L summary with on/off toggles
  - Open positions with live P&L and trailing stop visualization
  - Manual order entry panel
  - Exit alerts, market insights, 5-min channel chart, TF positions
  - Trade history (from DB)
  - Audio alerts (JS-based, fires on trade open/close)
"""

import logging
from datetime import datetime

import pandas as pd
import panel as pn
import param

logger = logging.getLogger(__name__)

# Zone thresholds (match v15/core/channel_surfer.py)
ZONE_OVERSOLD = 0.15
ZONE_LOWER = 0.30
ZONE_UPPER = 0.70
ZONE_OVERBOUGHT = 0.85

TF_ORDER = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly']


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
    """Per-algo P&L summary with on/off toggles. Bound to positions_version + algo_control_version."""
    def _render(positions_version, algo_control_version):
        if not hasattr(state, 'trade_db') or state.trade_db is None:
            return pn.pane.HTML(
                '<div style="color:#888; padding:8px;">No trade database</div>')

        db = state.trade_db
        rows_html = []

        # Dynamic algo list from LiveEngine + manual
        engine = getattr(state, 'live_engine', None)
        if engine:
            algos = [a.algo_id for a in engine._algos] + ['manual']
        else:
            algos = ['c16', 'c16-dw', 'c16-ml', 'c16-intra', 'c16-oe', 'manual']
        for algo_id in algos:
            open_trades = db.get_open_trades(source='ib', algo_id=algo_id)
            closed = db.get_closed_trades(source='ib', algo_id=algo_id)

            total_pnl = sum(t.get('pnl', 0) for t in closed)
            # Today's realized P&L (trades closed today ET)
            today_et = pd.Timestamp.now(tz='US/Eastern').date()
            day_pnl = 0
            day_trades = 0
            for t in closed:
                exit_ts = t.get('exit_time') or t.get('closed_at', '')
                if exit_ts:
                    try:
                        ts = pd.Timestamp(exit_ts)
                        if ts.tzinfo is None:
                            ts = ts.tz_localize('US/Eastern')
                        else:
                            ts = ts.tz_convert('US/Eastern')
                        if ts.date() == today_et:
                            day_pnl += t.get('pnl', 0)
                            day_trades += 1
                    except Exception:
                        pass
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
            day_color = '#00e676' if day_pnl >= 0 else '#ff5252'
            unr_color = '#00e676' if unrealized >= 0 else '#ff5252'

            # Enabled toggle state
            enabled = True
            if hasattr(state, 'live_engine') and state.live_engine:
                enabled = state.live_engine._algo_enabled.get(algo_id, True)

            dot = '\u25cf' if enabled else '\u25cb'
            dot_color = '#00e676' if enabled else '#666'

            rows_html.append(f"""
            <tr style="border-bottom:1px solid #333;">
                <td style="padding:6px 12px; color:{dot_color};">{dot} {algo_id}</td>
                <td style="padding:6px 12px; color:{pnl_color};">${total_pnl:,.0f}</td>
                <td style="padding:6px 12px; color:{day_color};">${day_pnl:,.0f}</td>
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
                    <th style="text-align:left; padding:4px 12px;">Today</th>
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

    return pn.bind(_render,
                   positions_version=state.param.positions_version,
                   algo_control_version=state.param.algo_control_version)


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
    """Warning banner for ib_degraded state."""
    def _render(ib_connected):
        parts = []
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
    """Kill All button with confirmation. Uses unkill() to restore pre-kill state."""
    kill_btn = pn.widgets.Button(
        name='KILL ALL', button_type='danger', width=120)
    status = pn.pane.HTML('', width=200)

    def _on_kill(event):
        engine = getattr(state, 'live_engine', None)
        if engine:
            if state.kill_switch:
                engine.unkill()
                state.kill_switch = False
                state.algo_control_version += 1
                kill_btn.name = 'KILL ALL'
                kill_btn.button_type = 'danger'
                status.object = '<span style="color:#00e676;">Algos restored</span>'
            else:
                engine.kill_all()
                state.kill_switch = True
                state.algo_control_version += 1
                kill_btn.name = 'UNKILL'
                kill_btn.button_type = 'warning'
                status.object = '<span style="color:#ff5252;">All algos KILLED</span>'

    kill_btn.on_click(_on_kill)
    return pn.Row(kill_btn, status)


# ---------------------------------------------------------------------------
# Exit Alerts
# ---------------------------------------------------------------------------

def _exit_alerts_pane(exit_alert_html):
    """Exit alerts. Only re-renders when an exit actually happens."""
    if not exit_alert_html:
        return pn.pane.HTML('')
    return pn.pane.HTML(exit_alert_html, sizing_mode='stretch_width')


# ---------------------------------------------------------------------------
# Market Insights
# ---------------------------------------------------------------------------

def _market_insights(analysis):
    if analysis is None:
        return pn.pane.HTML('')

    insights = []
    tf_hours = {
        '5min': 1/12, '15min': 0.25, '30min': 0.5, '1h': 1, '2h': 2, '3h': 3,
        '4h': 4, 'daily': 24, 'weekly': 168, 'monthly': 720,
    }

    for tf, state in analysis.tf_states.items():
        if not state.valid:
            continue
        hrs = tf_hours.get(tf, 1)

        if getattr(state, 'momentum_is_turning', False):
            direction = 'sell-off' if getattr(state, 'momentum_direction', 0) < 0 else 'rally'
            insights.append(f'&#9888;&#65039; {tf} momentum turning — {direction} may be near exhaustion')

        pos = getattr(state, 'position_pct', 0.5)
        if pos is not None:
            pos_f = float(pos)
            if pos_f <= 0.05:
                hl = getattr(state, 'ou_half_life', None)
                rs = getattr(state, 'ou_reversion_score', 0)
                hl_str = f', bounce expected ~{hl * hrs:.0f}h' if hl is not None and rs > 0.2 else ''
                insights.append(f'&#128205; {tf} price AT channel bottom (position={pos_f:.1%}){hl_str}')
            elif pos_f >= 0.95:
                hl = getattr(state, 'ou_half_life', None)
                rs = getattr(state, 'ou_reversion_score', 0)
                hl_str = f', pullback expected ~{hl * hrs:.0f}h' if hl is not None and rs > 0.2 else ''
                insights.append(f'&#128205; {tf} price AT channel top (position={pos_f:.1%}){hl_str}')

        bp_dn = float(getattr(state, 'break_prob_down', 0))
        bp_up = float(getattr(state, 'break_prob_up', 0))
        if bp_dn > 0.55:
            insights.append(f'&#128308; {tf} high breakdown probability: {bp_dn:.0%}')
        elif bp_up > 0.55:
            insights.append(f'&#128994; {tf} high breakout probability: {bp_up:.0%}')

        ch = float(getattr(state, 'channel_health', 1.0))
        if ch < 0.35:
            insights.append(f'&#128993; {tf} channel health weak ({ch:.2f}) — signal less reliable')

        te = float(getattr(state, 'total_energy', 0))
        be = float(getattr(state, 'binding_energy', 1))
        if be > 0 and te / be > 2.5:
            insights.append(f'&#9889; {tf} energy ratio {te/be:.1f}x — channel under stress')

    # Confluence note
    cf = getattr(analysis, 'confluence_matrix', {})
    if cf:
        vals = list(cf.values())
        if len(vals) >= 3 and all(abs(v - vals[0]) < 0.01 for v in vals):
            direction = 'bearish' if getattr(analysis.signal, 'action', '') == 'SELL' else 'bullish'
            insights.append(f'&#9989; All {len(cf)} timeframes in full consensus ({direction}) — high conviction signal')

    if not insights:
        return pn.pane.HTML('')

    items_html = ''.join(f'<div style="padding:3px 0;color:#ccc;font-size:13px;">{msg}</div>' for msg in insights)
    html = f"""
    <details style="background:#1a1a2e;border:1px solid #333;border-radius:8px;padding:8px 12px;margin:6px 0;">
        <summary style="cursor:pointer;font-weight:600;color:#aaa;">Market Insights</summary>
        <div style="margin-top:6px;">{items_html}</div>
    </details>
    """
    return pn.pane.HTML(html, sizing_mode='stretch_width')


# ---------------------------------------------------------------------------
# 5-min Channel Chart
# ---------------------------------------------------------------------------

def _channel_chart(analysis, current_tsla):
    if analysis is None or current_tsla is None or len(current_tsla) == 0:
        return pn.pane.HTML('')

    try:
        import plotly.graph_objects as go
        import pytz
        import pandas as pd
        from v15.core.channel import detect_channels_multi_window, select_best_channel
    except ImportError:
        return pn.pane.HTML('<div style="color:#888;">Plotly or channel module not available</div>')

    # Filter to today's trading session
    try:
        et_tz = pytz.timezone('America/New_York')
        if current_tsla.index.tz is not None:
            dates_et = current_tsla.index.tz_convert(et_tz).date
        else:
            dates_et = current_tsla.index.date
        today_et = pd.Timestamp.now(tz=et_tz).date()
        df_today = current_tsla[dates_et == today_et]
        if len(df_today) < 5 and len(current_tsla) > 0:
            most_recent = dates_et[-1]
            df_today = current_tsla[dates_et == most_recent]
        df_chart = df_today.copy() if len(df_today) > 0 else current_tsla.tail(100).copy()
    except Exception:
        df_chart = current_tsla.tail(100).copy()

    # Detect 5-min channel
    try:
        multi_ch = detect_channels_multi_window(df_chart, windows=[10, 15, 20, 30, 40])
        best_ch, best_w = select_best_channel(multi_ch)
    except Exception:
        best_ch = None

    x_values = list(range(len(df_chart)))

    # ET tick labels
    disp_idx = df_chart.index
    try:
        if disp_idx.tz is not None:
            disp_idx = disp_idx.tz_convert('America/New_York')
    except Exception:
        pass

    n = len(df_chart)
    tick_step = max(1, n // 10)
    tick_positions = list(range(0, n, tick_step))
    if tick_positions and tick_positions[-1] != n - 1:
        tick_positions.append(n - 1)
    tick_labels = [disp_idx[p].strftime('%H:%M') if p < len(disp_idx) else '' for p in tick_positions]

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=x_values,
        open=df_chart['open'], high=df_chart['high'],
        low=df_chart['low'], close=df_chart['close'],
        name='TSLA 5min',
        increasing_line_color='#00c853', decreasing_line_color='#ff1744',
    ))

    # Channel overlay
    if best_ch and best_ch.valid:
        ch_len = len(best_ch.center_line)
        ch_start = max(0, len(df_chart) - ch_len)
        ch_x = x_values[ch_start:ch_start + ch_len]

        fig.add_trace(go.Scatter(
            x=ch_x, y=best_ch.upper_line[:len(ch_x)],
            mode='lines', name='Upper',
            line=dict(color='rgba(255,100,100,0.6)', width=1, dash='dash'),
        ))
        fig.add_trace(go.Scatter(
            x=ch_x, y=best_ch.lower_line[:len(ch_x)],
            mode='lines', name='Lower',
            line=dict(color='rgba(100,255,100,0.6)', width=1, dash='dash'),
            fill='tonexty', fillcolor='rgba(100,100,255,0.05)',
        ))
        fig.add_trace(go.Scatter(
            x=ch_x, y=best_ch.center_line[:len(ch_x)],
            mode='lines', name='Center',
            line=dict(color='rgba(200,200,200,0.4)', width=1, dash='dot'),
        ))

    # Signal marker
    sig = analysis.signal
    if sig.action in ('BUY', 'SELL') and len(df_chart) > 0:
        last_x = x_values[-1]
        last_price = float(df_chart['close'].iloc[-1])
        color = '#00ff55' if sig.action == 'BUY' else '#ff4444'
        symbol = 'triangle-up' if sig.action == 'BUY' else 'triangle-down'
        fig.add_trace(go.Scatter(
            x=[last_x], y=[last_price],
            mode='markers+text',
            marker=dict(size=18, color=color, symbol=symbol),
            text=[sig.action],
            textposition='top center' if sig.action == 'BUY' else 'bottom center',
            textfont=dict(size=14, color=color),
            name=sig.action,
        ))

    # Title annotation
    state_5m = analysis.tf_states.get('5min')
    title_extra = ""
    if state_5m and state_5m.valid:
        title_extra = (f" | Pos: {state_5m.position_pct:.0%} | Health: {state_5m.channel_health:.0%}"
                       f" | Break: {state_5m.break_prob:.0%}")

    fig.update_layout(
        title=f"TSLA 5min Channel{title_extra}",
        template='plotly_dark',
        height=450,
        xaxis_rangeslider_visible=False,
        xaxis=dict(tickvals=tick_positions, ticktext=tick_labels),
        showlegend=False,
        margin=dict(l=50, r=20, t=40, b=30),
    )

    return pn.pane.Plotly(fig, sizing_mode='stretch_width')


# ---------------------------------------------------------------------------
# Per-TF Channel Positions
# ---------------------------------------------------------------------------

def _render_position_gauge(position_pct: float, width_px: int = 300) -> str:
    pos = max(0, min(1, position_pct))
    pct = pos * 100

    if pos <= ZONE_OVERSOLD:
        color, zone = '#00c853', 'STRONG BUY'
    elif pos <= ZONE_LOWER:
        color, zone = '#69f0ae', 'BUY ZONE'
    elif pos >= ZONE_OVERBOUGHT:
        color, zone = '#ff1744', 'STRONG SELL'
    elif pos >= ZONE_UPPER:
        color, zone = '#ff8a80', 'SELL ZONE'
    else:
        color, zone = '#ffab40', 'NEUTRAL'

    return f"""
    <div style="position:relative;height:32px;background:linear-gradient(90deg,
        #00c853 0%,#00c853 15%,#69f0ae 15%,#69f0ae 30%,#555 30%,#555 70%,
        #ff8a80 70%,#ff8a80 85%,#ff1744 85%,#ff1744 100%);
        border-radius:6px;width:{width_px}px;margin:2px 0;">
        <div style="position:absolute;left:{pct}%;top:-2px;transform:translateX(-50%);
            width:4px;height:36px;background:white;border-radius:2px;
            box-shadow:0 0 4px rgba(255,255,255,0.8);"></div>
        <div style="position:absolute;left:{pct}%;top:34px;transform:translateX(-50%);
            font-size:11px;color:{color};font-weight:bold;white-space:nowrap;">
            {pct:.0f}% {zone}</div>
    </div>
    """


def _tf_positions(analysis):
    if analysis is None or not analysis.tf_states:
        return pn.pane.HTML('')

    sorted_tfs = sorted(
        analysis.tf_states.items(),
        key=lambda x: TF_ORDER.index(x[0]) if x[0] in TF_ORDER else 99,
    )

    rows_html = []
    for tf, state in sorted_tfs:
        if not state.valid:
            continue

        dir_emoji = {'bull': '+', 'bear': '-', 'sideways': '~'}.get(state.channel_direction, '?')
        health_color = '#00c853' if state.channel_health > 0.6 else '#ffab40' if state.channel_health > 0.3 else '#ff1744'
        break_color = '#ff1744' if state.break_prob > 0.5 else '#ffab40' if state.break_prob > 0.3 else '#888'

        gauge = _render_position_gauge(state.position_pct, width_px=280)

        rows_html.append(f"""
        <div style="display:flex;align-items:flex-start;gap:12px;margin:8px 0;padding:4px 0;
                    border-bottom:1px solid #222;">
            <div style="min-width:70px;font-weight:600;color:#ccc;">{tf} ({dir_emoji})</div>
            <div>{gauge}<div style="height:16px;"></div></div>
            <div style="font-size:12px;color:#aaa;min-width:250px;">
                <span style="color:{health_color};font-weight:bold;">Health: {state.channel_health:.0%}</span> |
                <span style="color:{break_color};">Break: {state.break_prob:.0%}</span> |
                OU: {state.ou_half_life:.0f}bars |
                R2: {state.r_squared:.2f}
            </div>
        </div>
        """)

    html = f"""
    <div style="margin:10px 0;">
        <div style="font-size:14px;font-weight:600;color:#ccc;margin-bottom:6px;">
            Channel Positions by Timeframe</div>
        {''.join(rows_html)}
    </div>
    """
    return pn.pane.HTML(html, sizing_mode='stretch_width')


# ---------------------------------------------------------------------------
# Audio Alerts
# ---------------------------------------------------------------------------

_PREV_TRADES_VERSION = [0]


def _audio_alert(trades_version, exit_alert_html):
    if trades_version <= _PREV_TRADES_VERSION[0]:
        return pn.pane.HTML('')
    _PREV_TRADES_VERSION[0] = trades_version

    # Exit alert present -> exit sound, otherwise entry sound
    is_exit = bool(exit_alert_html)
    if is_exit:
        is_profit = 'take_profit' in exit_alert_html or 'trailing_stop' in exit_alert_html
        if is_profit:
            # Celebratory ascending chime: C5 -> E5 -> G5
            js = """
            <script>
            (function() {
                try {
                    const ctx = new (window.AudioContext || window.webkitAudioContext)();
                    [523, 659, 784].forEach(function(f, i) {
                        const osc = ctx.createOscillator();
                        const gain = ctx.createGain();
                        osc.connect(gain); gain.connect(ctx.destination);
                        osc.type = 'sine';
                        osc.frequency.value = f;
                        gain.gain.setValueAtTime(0.25, ctx.currentTime + i*0.15);
                        gain.gain.linearRampToValueAtTime(0, ctx.currentTime + i*0.15 + 0.3);
                        osc.start(ctx.currentTime + i*0.15);
                        osc.stop(ctx.currentTime + i*0.15 + 0.3);
                    });
                } catch(e) {}
            })();
            </script>
            """
        else:
            # Stop loss / other exit: descending tone
            js = """
            <script>
            (function() {
                try {
                    const ctx = new (window.AudioContext || window.webkitAudioContext)();
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.connect(gain); gain.connect(ctx.destination);
                    osc.type = 'square';
                    osc.frequency.setValueAtTime(600, ctx.currentTime);
                    osc.frequency.linearRampToValueAtTime(300, ctx.currentTime + 0.4);
                    gain.gain.setValueAtTime(0.15, ctx.currentTime);
                    gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.4);
                    osc.start(ctx.currentTime);
                    osc.stop(ctx.currentTime + 0.4);
                } catch(e) {}
            })();
            </script>
            """
    else:
        # Entry: ascending tone
        js = """
        <script>
        (function() {
            try {
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const osc = ctx.createOscillator();
                const gain = ctx.createGain();
                osc.connect(gain); gain.connect(ctx.destination);
                osc.type = 'sine';
                osc.frequency.setValueAtTime(400, ctx.currentTime);
                osc.frequency.linearRampToValueAtTime(800, ctx.currentTime + 0.3);
                gain.gain.setValueAtTime(0.3, ctx.currentTime);
                gain.gain.linearRampToValueAtTime(0, ctx.currentTime + 0.5);
                osc.start(ctx.currentTime);
                osc.stop(ctx.currentTime + 0.5);
            } catch(e) {}
        })();
        </script>
        """
    return pn.pane.HTML(js)


# ---------------------------------------------------------------------------
# Tab Assembly
# ---------------------------------------------------------------------------

def ib_live_tab(state):
    """Build the IB Live tab. Returns a Panel Column."""
    components = [
        _degraded_banner(state),
        _price_banner(state),
        _algo_pnl_summary(state),
        _open_positions(state),
    ]

    # Order entry panel (if available)
    try:
        from v15.panel_dashboard.order_entry import order_entry_panel
        components.append(order_entry_panel(state))
    except ImportError:
        pass

    # Exit alerts
    components.append(pn.bind(_exit_alerts_pane, state.param.exit_alert_html))

    # Market insights
    components.append(pn.bind(_market_insights, state.param.analysis))

    # 5-min channel chart
    components.append(pn.bind(_channel_chart, state.param.analysis, state.param.current_tsla))

    # Per-TF channel positions
    components.append(pn.bind(_tf_positions, state.param.analysis))

    # Trade history
    components.append(_trade_history(state))

    # Audio alerts (fires on trade open/close)
    components.append(pn.bind(_audio_alert, state.param.trades_version,
                              state.param.exit_alert_html))

    return pn.Column(*components, sizing_mode='stretch_width')
