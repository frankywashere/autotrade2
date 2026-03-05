"""Channel Surfer tab — reactive Panel port of the Streamlit CS tab."""

import logging

import panel as pn
import param
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Zone thresholds (match v15/core/channel_surfer.py)
ZONE_OVERSOLD = 0.15
ZONE_LOWER = 0.30
ZONE_UPPER = 0.70
ZONE_OVERBOUGHT = 0.85

TF_ORDER = ['5min', '15min', '30min', '1h', '2h', '3h', '4h', 'daily', 'weekly', 'monthly']

# Consistent display names for signal_source values
SOURCE_DISPLAY = {
    'CS-5TF': 'CS-5TF',
    'CS-DW': 'CS-DW',
    'surfer_ml': 'Surfer ML',
    'intraday': 'Intraday',
    'oe_signals5': 'OE-Sig5',
}


def channel_surfer_tab(state) -> pn.Column:
    """Build the Channel Surfer tab. Each section is bound to the specific param it depends on."""

    return pn.Column(
        # Price banner — re-renders on price change
        pn.bind(_price_banner, state.param.tsla_price, state.param.price_source,
                state.param.price_delta),
        # Active trades banner — only re-renders on actual trade open/close
        pn.bind(_active_trades_banner, state.param.trades_version,
                scanner=state.scanner, scanner_dw=state.scanner_dw,
                scanner_ml=state.scanner_ml, scanner_intra=state.scanner_intra,
                scanner_oe=state.scanner_oe,
                extra_scanners=[state.scanner_14a, state.scanner_14a_dw,
                                state.scanner_14a_ml, state.scanner_14a_intra]),
        # Market insights
        pn.bind(_market_insights, state.param.analysis),
        # Scanner metrics + live P&L — re-renders on price change
        pn.bind(_scanner_metrics, state.param.positions_version, state.param.tsla_price,
                scanner=state.scanner, scanner_dw=state.scanner_dw,
                scanner_ml=state.scanner_ml, scanner_intra=state.scanner_intra,
                scanner_oe=state.scanner_oe,
                extra_scanners=[state.scanner_14a, state.scanner_14a_dw,
                                state.scanner_14a_ml, state.scanner_14a_intra]),
        # Exit alerts — only re-renders when exit_alert_html changes
        pn.bind(_exit_alerts_pane, state.param.exit_alert_html),
        # Open positions with live P&L — re-renders on price change
        pn.bind(_open_positions_pane, state.param.positions_version, state.param.tsla_price,
                scanner=state.scanner, scanner_dw=state.scanner_dw,
                scanner_ml=state.scanner_ml, scanner_intra=state.scanner_intra,
                scanner_oe=state.scanner_oe,
                extra_scanners=[state.scanner_14a, state.scanner_14a_dw,
                                state.scanner_14a_ml, state.scanner_14a_intra]),
        # Trade history — only re-renders on actual trade open/close (preserves <details> state)
        pn.bind(_trade_history_pane, state.param.trades_version,
                scanner=state.scanner, scanner_dw=state.scanner_dw,
                scanner_ml=state.scanner_ml, scanner_intra=state.scanner_intra,
                scanner_oe=state.scanner_oe,
                extra_scanners=[state.scanner_14a, state.scanner_14a_dw,
                                state.scanner_14a_ml, state.scanner_14a_intra]),
        # Regime indicator
        pn.bind(_regime_indicator, state.param.analysis),
        # Break direction predictor
        pn.bind(_break_predictor, state.param.analysis,
                native_tf_data=state.native_tf_data),
        # 5-min channel chart
        pn.bind(_channel_chart, state.param.analysis, state.param.current_tsla),
        # Signal components
        pn.bind(_signal_components, state.param.analysis),
        # Per-TF channel positions
        pn.bind(_tf_positions, state.param.analysis),
        # Collapsible detail cards
        pn.bind(_detail_cards, state.param.analysis),
        # Audio alerts — fires only on actual trade open/close
        pn.bind(_audio_alert, state.param.trades_version,
                state.param.exit_alert_html),
        sizing_mode='stretch_width',
    )


# ---------------------------------------------------------------------------
# Price Banner
# ---------------------------------------------------------------------------

def _price_banner(tsla_price, price_source, price_delta):
    if tsla_price <= 0:
        return pn.pane.HTML(
            '<div style="text-align:center;padding:10px;color:#888;">Waiting for price data...</div>',
            sizing_mode='stretch_width',
        )

    delta_color = '#00e676' if price_delta >= 0 else '#ff5252'
    delta_pct = price_delta / (tsla_price - price_delta) * 100 if (tsla_price - price_delta) != 0 else 0

    live_dot = ''
    if price_source == 'IB LIVE':
        live_dot = '<span style="color:#00e676;font-size:10px;">&#9679;</span> '

    if price_source == 'IB LIVE':
        source_color = '#00e676'
    elif price_source == 'NONE':
        source_color = '#ff5252'
    else:
        source_color = '#888'

    html = f"""
    <div style="display:flex;align-items:center;gap:16px;padding:8px 16px;
                background:#111;border-radius:8px;margin:4px 0;">
        <div>
            {live_dot}<span style="color:{source_color};font-size:11px;font-weight:600;">
            {price_source}</span>
        </div>
        <div style="font-size:28px;font-weight:700;color:#fff;">
            TSLA ${tsla_price:.2f}
        </div>
        <div style="font-size:16px;color:{delta_color};font-weight:600;">
            {price_delta:+.2f} ({delta_pct:+.2f}%)
        </div>
    </div>
    """
    return pn.pane.HTML(html, sizing_mode='stretch_width')


# ---------------------------------------------------------------------------
# Active Trades Banner — only renders when positions are open
# ---------------------------------------------------------------------------

def _active_trades_banner(trades_version, scanner=None, scanner_dw=None, scanner_ml=None, scanner_intra=None, scanner_oe=None, extra_scanners=None):
    """Prominent banner showing active trades with entry/stop/TP and signal source.

    Bound to trades_version only — re-renders on actual trade open/close,
    not on every price tick or position P&L update.
    """
    all_positions = []
    scanners = [scanner, scanner_dw, scanner_ml, scanner_intra, scanner_oe] + (extra_scanners or [])
    for scnr in scanners:
        if scnr and scnr.positions:
            all_positions.extend(scnr.positions.values())
    if not all_positions:
        return pn.pane.HTML('')

    cards = []
    for pos in all_positions:
        source = SOURCE_DISPLAY.get(getattr(pos, 'signal_source', ''), 'CS')
        sig_type = getattr(pos, 'signal_type', 'bounce')
        type_label = 'BREAKOUT' if sig_type == 'break' else 'BOUNCE'
        type_icon = '&#x26A1;' if sig_type == 'break' else '&#x21C4;'
        action = 'BUY' if pos.direction == 'long' else 'SELL'

        # Colors by direction
        if action == 'BUY':
            bg = "linear-gradient(135deg,#004d1a,#00802b)"
            border, glow = "#00c853", "#00ff55"
            info_color, reason_color = '#ccffcc', '#88cc99'
        else:
            bg = "linear-gradient(135deg,#4d0000,#800000)"
            border, glow = "#ff1744", "#ff4444"
            info_color, reason_color = '#ffcccc', '#cc8888'

        # Source badge color
        if source == 'CS-DW':
            src_bg = '#e6a800'
        elif source == 'Surfer ML':
            src_bg = '#7b1fa2'
        elif source == 'Intraday':
            src_bg = '#0288d1'
        elif source == 'OE-Sig5':
            src_bg = '#2e7d32'
        else:
            src_bg = '#e65100'

        # R:R from entry
        stop_dist = abs(pos.entry_price - pos.stop_price)
        tp_dist = abs(pos.tp_price - pos.entry_price)
        rr = tp_dist / stop_dist if stop_dist > 0 else 0

        cards.append(f"""
        <div style="background:{bg};border:2px solid {border};
        border-radius:12px;padding:20px;text-align:center;margin:10px 0;">
        <div style="display:flex;justify-content:center;align-items:center;gap:10px;margin-bottom:4px;">
            <span style="font-size:12px;font-weight:700;color:#ffcc00;letter-spacing:2px;">
            {type_icon} {type_label}</span>
            <span style="background:{src_bg};color:#fff;padding:2px 10px;border-radius:6px;
            font-size:13px;font-weight:700;">{source}</span>
        </div>
        <div style="font-size:48px;font-weight:900;color:{glow};text-shadow:0 0 20px {glow};">
        {action}</div>
        <div style="font-size:18px;color:{info_color};margin-top:8px;">
        Confidence: {pos.confidence:.0%} | {pos.primary_tf} | {pos.shares} shares
        </div>
        <div style="font-size:16px;color:{info_color};margin-top:8px;font-family:monospace;">
        Entry: ${pos.entry_price:.2f} &nbsp; Stop: ${pos.stop_price:.2f} &nbsp;
        TP: ${pos.tp_price:.2f} &nbsp; R:R {rr:.1f}:1</div>
        <div style="font-size:13px;color:{reason_color};margin-top:4px;">{pos.reason}</div>
        </div>""")

    return pn.pane.HTML('\n'.join(cards), sizing_mode='stretch_width')


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
# Scanner Panel
# ---------------------------------------------------------------------------

def _scanner_metrics(positions_version, tsla_price, scanner=None, scanner_dw=None, scanner_ml=None, scanner_intra=None, scanner_oe=None, extra_scanners=None):
    """Scanner equity/P&L metrics. Bound to tsla_price for live unrealized updates."""
    if scanner is None:
        return pn.pane.HTML(
            '<div style="color:#888;padding:8px;">Scanner not available</div>',
            sizing_mode='stretch_width',
        )

    all_scanners = [s for s in [scanner, scanner_dw, scanner_ml, scanner_intra, scanner_oe] + (extra_scanners or []) if s is not None]
    total_equity = sum(s.equity for s in all_scanners)
    total_capital = sum(s.config.initial_capital for s in all_scanners)
    unrealized = sum(s.get_unrealized_pnl(tsla_price) for s in all_scanners) if tsla_price > 0 else 0.0

    model_rows = ''
    for s in all_scanners:
        name = getattr(s, 'display_name', getattr(s, 'model_tag', '?'))
        eq_delta = s.equity - s.config.initial_capital
        eq_color = '#00e676' if eq_delta >= 0 else '#ff5252'
        model_rows += (
            f'<span style="color:#888;font-size:11px;margin-left:12px;">'
            f'{name}: ${s.equity:,.0f} (<span style="color:{eq_color}">{eq_delta:+,.0f}</span>)</span>'
        )

    html = f"""
    <div style="border:1px solid #333;border-radius:8px;padding:12px;margin:6px 0;">
        <div style="font-size:14px;font-weight:700;color:#ccc;margin-bottom:8px;">Live Scanner</div>
        <div style="display:flex;gap:24px;padding:8px 12px;background:#111;border-radius:8px;margin:4px 0;">
            <div>
                <div style="font-size:11px;color:#888;">Starting Capital</div>
                <div style="font-size:18px;font-weight:600;color:#ccc;">${total_capital:,.0f}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#888;">Equity</div>
                <div style="font-size:18px;font-weight:600;color:#ccc;">${total_equity:,.0f}</div>
                <div style="font-size:11px;color:{'#00e676' if total_equity >= total_capital else '#ff5252'};">
                {total_equity - total_capital:+,.0f}</div>
                <div>{model_rows}</div>
            </div>
            <div>
                <div style="font-size:11px;color:#888;">Unrealized P&L</div>
                <div style="font-size:18px;font-weight:600;color:{'#00e676' if unrealized >= 0 else '#ff5252'};">
                ${unrealized:+,.0f}</div>
            </div>
        </div>
    </div>
    """
    return pn.pane.HTML(html, sizing_mode='stretch_width')


def _exit_alerts_pane(exit_alert_html):
    """Exit alerts. Only re-renders when an exit actually happens."""
    if not exit_alert_html:
        return pn.pane.HTML('')
    return pn.pane.HTML(exit_alert_html, sizing_mode='stretch_width')


def _open_positions_pane(positions_version, tsla_price, scanner=None, scanner_dw=None, scanner_ml=None, scanner_intra=None, scanner_oe=None, extra_scanners=None):
    """Open positions with live P&L. Bound to tsla_price for live updates."""
    all_positions = {}
    for scnr in [scanner, scanner_dw, scanner_ml, scanner_intra, scanner_oe] + (extra_scanners or []):
        if scnr and scnr.positions:
            all_positions.update(scnr.positions)

    if not all_positions:
        return pn.pane.HTML(
            '<div style="color:#888;font-size:12px;padding:4px 0;">No open positions.</div>',
            sizing_mode='stretch_width',
        )

    position_html = ''
    cs5_pos = {k: v for k, v in all_positions.items()
               if getattr(v, 'signal_source', '') == 'CS-5TF'}
    dw_pos = {k: v for k, v in all_positions.items()
              if getattr(v, 'signal_source', '') == 'CS-DW'}
    ml_pos = {k: v for k, v in all_positions.items()
              if getattr(v, 'signal_source', '') == 'surfer_ml'}
    id_pos = {k: v for k, v in all_positions.items()
              if getattr(v, 'signal_source', '') == 'intraday'}
    oe_pos = {k: v for k, v in all_positions.items()
              if getattr(v, 'signal_source', '') == 'oe_signals5'}

    if cs5_pos:
        position_html += '<div style="font-weight:600;color:#ccc;margin:6px 0;">CS-5TF Positions</div>'
        for pos in cs5_pos.values():
            position_html += _position_card_html(pos, tsla_price)
    if dw_pos:
        position_html += '<div style="font-weight:600;color:#ccc;margin:6px 0;">CS-DW Positions</div>'
        for pos in dw_pos.values():
            position_html += _position_card_html(pos, tsla_price)
    if ml_pos:
        position_html += '<div style="font-weight:600;color:#ccc;margin:6px 0;">Surfer ML Positions</div>'
        for pos in ml_pos.values():
            position_html += _position_card_html(pos, tsla_price)
    if id_pos:
        position_html += '<div style="font-weight:600;color:#ccc;margin:6px 0;">Intraday Positions</div>'
        for pos in id_pos.values():
            position_html += _position_card_html(pos, tsla_price)
    if oe_pos:
        position_html += '<div style="font-weight:600;color:#ccc;margin:6px 0;">OE-Sig5 Positions</div>'
        for pos in oe_pos.values():
            position_html += _position_card_html(pos, tsla_price)

    return pn.pane.HTML(position_html, sizing_mode='stretch_width')


def _trade_history_pane(trades_version, scanner=None, scanner_dw=None, scanner_ml=None, scanner_intra=None, scanner_oe=None, extra_scanners=None):
    """Trade history. Bound to trades_version ONLY — not tsla_price or positions_version.

    Only re-renders when a trade actually opens or closes, so the <details>
    element preserves its open/closed state between price ticks.
    """
    all_closed = []
    for scnr in [scanner, scanner_dw, scanner_ml, scanner_intra, scanner_oe] + (extra_scanners or []):
        if scnr and scnr.closed_trades:
            all_closed.extend(scnr.closed_trades)

    if not all_closed:
        return pn.pane.HTML('')

    total = len(all_closed)
    total_pnl = sum(t.pnl for t in all_closed)
    wins = sum(1 for t in all_closed if t.pnl > 0)
    wr = wins / total if total > 0 else 0

    html = f"""
    <details style="margin-top:8px;">
        <summary style="cursor:pointer;font-size:13px;color:#aaa;">
            Trade History: {total} trades | WR {wr:.0%} | Total P&L ${total_pnl:+,.0f}
        </summary>
        <div style="max-height:300px;overflow-y:auto;margin-top:4px;">
            {_trade_history_html(all_closed)}
        </div>
    </details>
    """
    return pn.pane.HTML(html, sizing_mode='stretch_width')


def _position_card_html(pos, current_price: float) -> str:
    raw_source = getattr(pos, 'signal_source', '')
    sys_tag = SOURCE_DISPLAY.get(raw_source, raw_source or 'CS')
    if pos.direction == 'long':
        upnl = (current_price - pos.entry_price) * pos.shares if current_price > 0 else 0
        dist_stop = (current_price - pos.stop_price) / current_price if current_price > 0 else 0
        dist_tp = (pos.tp_price - current_price) / current_price if current_price > 0 else 0
    else:
        upnl = (pos.entry_price - current_price) * pos.shares if current_price > 0 else 0
        dist_stop = (pos.stop_price - current_price) / current_price if current_price > 0 else 0
        dist_tp = (current_price - pos.tp_price) / current_price if current_price > 0 else 0

    upnl_color = "#00e676" if upnl >= 0 else "#ff5252"
    stop_color = '#ff5252' if dist_stop < 0.003 else ('#ff9800' if dist_stop < 0.01 else '#888')
    if raw_source == 'CS-DW':
        tag_color = '#ffd54f'
    elif raw_source == 'surfer_ml':
        tag_color = '#ce93d8'
    elif raw_source == 'intraday':
        tag_color = '#4fc3f7'
    elif raw_source == 'oe_signals5':
        tag_color = '#66bb6a'
    else:
        tag_color = '#ff9800'

    return (
        f'<div style="background:#1a2233;padding:8px;border-radius:6px;margin:3px 0;">'
        f'<span style="background:{tag_color};color:#fff;padding:1px 6px;border-radius:4px;'
        f'font-size:11px;font-weight:600;">{sys_tag}</span> '
        f'[{pos.pos_id}] <b>{pos.direction.upper()}</b> {pos.shares}sh '
        f'@ ${pos.entry_price:.2f} | '
        f'<span style="color:{stop_color}">SL: ${pos.stop_price:.2f} ({dist_stop:.1%} away)</span> | '
        f'TP: ${pos.tp_price:.2f} ({dist_tp:.1%} away) | '
        f'Unrealized: <b style="color:{upnl_color}">${upnl:+,.0f}</b>'
        f'</div>'
    )


def _trade_history_html(closed_trades) -> str:
    rows = []
    for t in reversed(closed_trades[-50:]):
        pnl_color = '#00e676' if t.pnl >= 0 else '#ff5252'
        raw_source = getattr(t, 'signal_source', '')
        source = SOURCE_DISPLAY.get(raw_source, raw_source)
        source_tag = (f'<span style="color:#64b5f6;font-size:10px;background:#1a237e;'
                      f'padding:1px 4px;border-radius:3px;margin-right:4px;">'
                      f'{source}</span>') if source else ''
        rows.append(
            f'<div style="font-size:12px;border-left:3px solid {pnl_color};'
            f'padding:3px 8px;margin:2px 0;">'
            f'{source_tag}'
            f'<b style="color:{pnl_color}">${t.pnl:+,.0f}</b> '
            f'<span style="color:#888">{t.direction.upper()} | {t.exit_reason} | '
            f'{t.hold_minutes:.0f}m | Entry ${t.entry_price:.2f} Exit ${t.exit_price:.2f}</span>'
            f'</div>'
        )
    return ''.join(rows)


# ---------------------------------------------------------------------------
# Regime Indicator
# ---------------------------------------------------------------------------

def _regime_indicator(analysis):
    if analysis is None:
        return pn.pane.HTML('')

    regime = getattr(analysis, 'regime', None)
    if regime is None:
        return pn.pane.HTML('')

    regime_colors = {'ranging': '#4fc3f7', 'trending': '#ff9800', 'transitioning': '#ce93d8'}
    r_color = regime_colors.get(regime.regime, '#888')
    trend_arrow = '&#x2191;' if regime.trend_direction > 0 else (
        '&#x2193;' if regime.trend_direction < 0 else '&#x2194;')

    html = f"""
    <div style="text-align:center;margin:5px 0;font-size:13px;padding:6px;
                background:#111;border-radius:6px;">
        <span style="color:{r_color};font-weight:700;">Market: {regime.regime.upper()}</span>
        <span style="color:#888;margin:0 10px;">|</span>
        <span style="color:#aaa;">Health: {regime.avg_health:.0%}</span>
        <span style="color:#888;margin:0 10px;">|</span>
        <span style="color:#aaa;">Trend: {trend_arrow}</span>
    </div>
    """
    return pn.pane.HTML(html, sizing_mode='stretch_width')


# ---------------------------------------------------------------------------
# Break Direction Predictor
# ---------------------------------------------------------------------------

def _break_predictor(analysis, native_tf_data=None):
    _unavail = (
        '<div style="background:#1a1a2e;border:1px solid #333;border-radius:6px;'
        'padding:8px 12px;margin:4px 0;display:flex;align-items:center;gap:12px;">'
        '<span style="color:#aaa;font-size:11px;font-weight:600;white-space:nowrap;">'
        'BREAK PREDICTOR</span>'
        '<span style="color:#555;font-size:14px;font-weight:700;">UNAVAILABLE</span>'
        '<span style="color:#555;font-size:11px;">{reason}</span></div>'
    )

    try:
        from v15.core.break_predictor import extract_break_features, predict_break
    except ImportError as e:
        return pn.pane.HTML(_unavail.format(reason=f"import failed: {e}"), sizing_mode='stretch_width')

    if analysis is None:
        return pn.pane.HTML(_unavail.format(reason="no channel analysis"), sizing_mode='stretch_width')

    try:
        # SPY/VIX daily data comes from native_tf_data; no separate DataFrames needed
        features = extract_break_features(analysis, native_tf_data, None, None)
        if features is None:
            return pn.pane.HTML(_unavail.format(reason="insufficient data"), sizing_mode='stretch_width')
        result = predict_break(features)
    except Exception as e:
        return pn.pane.HTML(_unavail.format(reason=f"{type(e).__name__}: {e}"), sizing_mode='stretch_width')

    direction = result['direction']
    confidence = result['confidence']
    will_break = result['will_break']
    position = features['position']

    if position > 0.80:
        pos_label = "near upper boundary"
    elif position < 0.20:
        pos_label = "near lower boundary"
    else:
        pos_label = f"mid-channel ({position:.0%})"

    if will_break:
        dir_color = '#4caf50' if direction == 'UP' else '#ef5350'
        dir_arrow = '&#8593;' if direction == 'UP' else '&#8595;'
        dir_text = f"{dir_arrow} {direction}"
        status_text = "Break predicted"
    else:
        dir_color = '#888888'
        dir_text = "&#8594; HOLD"
        status_text = "Channel holds"

    # Alignment check
    align_html = ''
    if analysis.tf_states:
        for s in analysis.tf_states.values():
            if s.valid:
                if s.position_pct < 0.25 and direction == 'UP':
                    align_html = '<span style="color:#4caf50;font-size:11px;margin-left:8px;">&#10003; aligned</span>'
                elif s.position_pct < 0.25 and direction == 'DOWN':
                    align_html = '<span style="color:#ff9800;font-size:11px;margin-left:8px;">&#9888; counter-signal</span>'
                elif s.position_pct > 0.75 and direction == 'UP':
                    align_html = '<span style="color:#4caf50;font-size:11px;margin-left:8px;">&#10003; aligned</span>'
                elif s.position_pct > 0.75 and direction == 'DOWN':
                    align_html = '<span style="color:#ff9800;font-size:11px;margin-left:8px;">&#9888; counter-signal</span>'
                break

    html = f"""<div style="background:#1a1a2e;border:1px solid #333;border-radius:6px;
                    padding:8px 12px;margin:4px 0;display:flex;align-items:center;gap:12px;">
        <span style="color:#aaa;font-size:11px;font-weight:600;white-space:nowrap;">BREAK PREDICTOR</span>
        <span style="color:{dir_color};font-size:18px;font-weight:700;">{dir_text}</span>
        <span style="color:#888;font-size:11px;">{status_text}</span>
        <span style="color:#666;font-size:11px;">&#183;</span>
        <span style="color:#aaa;font-size:11px;">{pos_label}</span>
        <span style="color:#666;font-size:11px;">&#183;</span>
        <span style="color:#888;font-size:11px;">conf {confidence:.0%}</span>
        {align_html}
    </div>"""
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
# Signal Components
# ---------------------------------------------------------------------------

def _signal_components(analysis):
    if analysis is None:
        return pn.pane.HTML('')

    sig = analysis.signal
    sig_type_label = getattr(sig, 'signal_type', 'bounce').upper()

    components = [
        ("Type", sig_type_label, True),
        ("Position", sig.position_score, False),
        ("Energy", sig.energy_score, False),
        ("Entropy", sig.entropy_score, False),
        ("Confluence", sig.confluence_score, False),
        ("Timing", sig.timing_score, False),
        ("Health", sig.channel_health, False),
    ]

    cells = []
    for name, val, is_str in components:
        if is_str:
            color = "#4fc3f7" if val == "BOUNCE" else "#ff9800"
            cells.append(
                f'<div style="text-align:center;">'
                f'<div style="font-size:11px;color:#888;">{name}</div>'
                f'<div style="font-size:20px;font-weight:700;color:{color};">{val}</div>'
                f'</div>'
            )
        else:
            val_f = float(val)
            if val_f >= 0.7:
                color = '#00c853'
            elif val_f >= 0.4:
                color = '#ff9800'
            else:
                color = '#ff1744'
            cells.append(
                f'<div style="text-align:center;">'
                f'<div style="font-size:11px;color:#888;">{name}</div>'
                f'<div style="font-size:22px;font-weight:700;color:{color};">{val_f:.0%}</div>'
                f'</div>'
            )

    html = f"""
    <div style="margin:10px 0;">
        <div style="font-size:14px;font-weight:600;color:#ccc;margin-bottom:6px;">
            Signal Components ({sig_type_label})</div>
        <div style="display:flex;justify-content:space-around;background:#111;
                    border-radius:8px;padding:12px;">
            {''.join(cells)}
        </div>
    </div>
    """
    return pn.pane.HTML(html, sizing_mode='stretch_width')


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
# Collapsible Detail Cards
# ---------------------------------------------------------------------------

def _detail_cards(analysis):
    if analysis is None or not analysis.tf_states:
        return pn.pane.HTML('')

    sorted_tfs = sorted(
        analysis.tf_states.items(),
        key=lambda x: TF_ORDER.index(x[0]) if x[0] in TF_ORDER else 99,
    )

    # Detailed TF Table
    tf_table_rows = []
    for tf, state in sorted_tfs:
        if not state.valid:
            continue
        tf_table_rows.append(
            f'<tr><td>{tf}</td><td>{state.position_pct:.0%}</td><td>{state.channel_direction}</td>'
            f'<td>{state.channel_health:.0%}</td><td>{state.ou_theta:.3f}</td>'
            f'<td>{state.ou_half_life:.0f}</td><td>{state.ou_reversion_score:.0%}</td>'
            f'<td>{state.break_prob:.0%}</td><td>{state.potential_energy:.2f}</td>'
            f'<td>{state.kinetic_energy:.2f}</td><td>{state.binding_energy:.2f}</td>'
            f'<td>{state.entropy:.2f}</td><td>{state.r_squared:.2f}</td>'
            f'<td>{state.bounce_count}</td><td>{state.width_pct:.2f}</td></tr>'
        )
    tf_table_html = f"""
    <details style="margin:6px 0;background:#111;border-radius:8px;padding:8px 12px;">
        <summary style="cursor:pointer;font-weight:600;color:#aaa;">Detailed Per-TF Analysis</summary>
        <div style="overflow-x:auto;margin-top:6px;">
            <table style="width:100%;font-size:12px;color:#ccc;border-collapse:collapse;">
                <thead><tr style="border-bottom:1px solid #444;color:#888;">
                    <th>TF</th><th>Pos</th><th>Dir</th><th>Health</th><th>OU&theta;</th>
                    <th>OU t&frac12;</th><th>Revert</th><th>Break%</th><th>PE</th>
                    <th>KE</th><th>Bind</th><th>Entropy</th><th>R2</th>
                    <th>Bounces</th><th>Width%</th>
                </tr></thead>
                <tbody>{''.join(tf_table_rows)}</tbody>
            </table>
        </div>
    </details>
    """

    # Confluence Matrix
    cf = analysis.confluence_matrix
    conf_rows = []
    if cf:
        for tf_name, score in cf.items():
            if score > 0:
                direction = 'Bullish' if score > 0.6 else ('Bearish' if score < 0.4 else 'Neutral')
                conf_rows.append(f'<tr><td>{tf_name}</td><td>{score:.0%}</td><td>{direction}</td></tr>')

    conf_html = f"""
    <details style="margin:6px 0;background:#111;border-radius:8px;padding:8px 12px;">
        <summary style="cursor:pointer;font-weight:600;color:#aaa;">Multi-TF Confluence</summary>
        <div style="margin-top:6px;">
            <table style="font-size:12px;color:#ccc;border-collapse:collapse;">
                <thead><tr style="border-bottom:1px solid #444;color:#888;">
                    <th>Timeframe</th><th>Alignment</th><th>Direction</th>
                </tr></thead>
                <tbody>{''.join(conf_rows) if conf_rows else '<tr><td colspan="3" style="color:#888;">No confluence data</td></tr>'}</tbody>
            </table>
        </div>
    </details>
    """

    # Energy diagram (Plotly)
    energy_pane = _energy_diagram(sorted_tfs)

    return pn.Column(
        pn.pane.HTML(tf_table_html, sizing_mode='stretch_width'),
        pn.pane.HTML(conf_html, sizing_mode='stretch_width'),
        energy_pane,
        sizing_mode='stretch_width',
    )


def _energy_diagram(sorted_tfs):
    valid_states = [(tf, s) for tf, s in sorted_tfs if s.valid]
    if not valid_states:
        return pn.pane.HTML('')

    try:
        import plotly.graph_objects as go
    except ImportError:
        return pn.pane.HTML('')

    tfs = [tf for tf, _ in valid_states]
    pe_vals = [s.potential_energy for _, s in valid_states]
    ke_vals = [s.kinetic_energy for _, s in valid_states]
    bind_vals = [s.binding_energy for _, s in valid_states]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Potential Energy', x=tfs, y=pe_vals, marker_color='#ff6b35'))
    fig.add_trace(go.Bar(name='Kinetic Energy', x=tfs, y=ke_vals, marker_color='#00b4d8'))
    fig.add_trace(go.Scatter(
        name='Binding Energy', x=tfs, y=bind_vals,
        mode='lines+markers', marker=dict(size=10, color='#e63946'),
        line=dict(width=3, dash='dash'),
    ))
    fig.update_layout(
        title='Channel Energy vs Binding Energy',
        yaxis_title='Energy (0-1)',
        barmode='stack',
        template='plotly_dark',
        height=350,
    )

    html = """
    <details style="margin:6px 0;background:#111;border-radius:8px;padding:8px 12px;">
        <summary style="cursor:pointer;font-weight:600;color:#aaa;">Energy State Diagram</summary>
        <div id="energy-chart-container" style="margin-top:6px;"></div>
    </details>
    """
    # Return as Plotly pane inside a Card
    return pn.Card(
        pn.pane.Plotly(fig, sizing_mode='stretch_width'),
        pn.pane.HTML(
            '<div style="font-size:11px;color:#888;padding:4px;">When total energy (PE+KE) exceeds binding energy, channel breakout is likely.</div>',
        ),
        title='Energy State Diagram',
        collapsed=True,
        sizing_mode='stretch_width',
        styles={'background': '#111'},
    )


# ---------------------------------------------------------------------------
# Audio Alerts — fires only on actual trade open/close
# ---------------------------------------------------------------------------

_PREV_TRADES_VERSION = [0]


def _audio_alert(trades_version, exit_alert_html):
    if trades_version <= _PREV_TRADES_VERSION[0]:
        return pn.pane.HTML('')
    _PREV_TRADES_VERSION[0] = trades_version

    # Exit alert present → exit sound, otherwise entry sound
    is_exit = bool(exit_alert_html)
    if is_exit:
        # Exit: check for profit vs loss in the HTML
        is_profit = 'take_profit' in exit_alert_html or 'trailing_stop' in exit_alert_html
        if is_profit:
            # Celebratory ascending chime: C5 → E5 → G5
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
