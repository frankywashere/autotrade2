"""Manual order entry panel for IB paper trading."""

import logging
import panel as pn

logger = logging.getLogger(__name__)

ORDER_TYPE_MAP = {'Market': 'MKT', 'Limit': 'LMT', 'Stop': 'STP'}


def order_entry_panel(state) -> pn.Column:
    """Build the manual order entry panel with form, price slider, and blotter."""

    # ── Account Summary ─────────────────────────────────────────────

    account_pane = pn.pane.HTML('', sizing_mode='stretch_width')

    def _render_account():
        if not state.ib_client or not state.ib_client.is_connected():
            return ('<div style="color:#ff5252;font-size:12px;font-weight:bold">'
                    'IB not connected — account data unavailable</div>')
        acct = state.ib_client.get_account_summary()
        if not acct:
            return ('<div style="color:#ffab00;font-size:12px;font-weight:bold">'
                    'Account summary empty — reqAccountSummary may have failed</div>')

        def _fmt(tag):
            val = acct.get(tag, '')
            if not val:
                return '--'
            try:
                return f'${float(val):,.0f}'
            except (ValueError, TypeError):
                return str(val)

        def _pnl_fmt(tag):
            val = acct.get(tag, '')
            if not val:
                return '--', '#aaa'
            try:
                v = float(val)
                color = '#00e676' if v >= 0 else '#ff5252'
                sign = '+' if v > 0 else ''
                return f'{sign}${v:,.0f}', color
            except (ValueError, TypeError):
                return str(val), '#aaa'

        nlv = _fmt('NetLiquidation')
        cash = _fmt('TotalCashValue')
        bp = _fmt('BuyingPower')
        gross = _fmt('GrossPositionValue')
        upnl, upnl_c = _pnl_fmt('UnrealizedPnL')
        rpnl, rpnl_c = _pnl_fmt('RealizedPnL')

        return (
            '<div style="display:flex;gap:20px;flex-wrap:wrap;font-size:13px;'
            'padding:6px 8px;background:#16213e;border-radius:6px">'
            f'<span><span style="color:#888">Net Liq</span> <b>{nlv}</b></span>'
            f'<span><span style="color:#888">Cash</span> <b>{cash}</b></span>'
            f'<span><span style="color:#888">Buying Power</span> <b>{bp}</b></span>'
            f'<span><span style="color:#888">Positions</span> <b>{gross}</b></span>'
            f'<span><span style="color:#888">Unreal P&L</span> '
            f'<b style="color:{upnl_c}">{upnl}</b></span>'
            f'<span><span style="color:#888">Real P&L</span> '
            f'<b style="color:{rpnl_c}">{rpnl}</b></span>'
            '</div>'
        )

    # ── Open Positions ───────────────────────────────────────────────

    positions_pane = pn.pane.HTML('', sizing_mode='stretch_width')

    def _render_positions():
        if not state.ib_client or not state.ib_client.is_connected():
            return ''
        positions = state.ib_client.get_positions()
        if not positions:
            return ('<div style="color:#888;font-size:12px;padding:4px 0">'
                    'No open positions</div>')

        td = 'padding:2px 8px'
        th = 'padding:2px 8px;text-align:left;color:#888'
        rows = ''
        for sym, p in sorted(positions.items()):
            qty = p['position']
            direction = 'LONG' if qty > 0 else 'SHORT'
            dir_color = '#00e676' if qty > 0 else '#ff5252'
            upnl = p['unrealizedPNL']
            pnl_color = '#00e676' if upnl >= 0 else '#ff5252'
            pnl_sign = '+' if upnl > 0 else ''
            rows += (
                f'<tr>'
                f'<td style="{td};font-weight:bold">{sym}</td>'
                f'<td style="{td};color:{dir_color}">{direction}</td>'
                f'<td style="{td}">{abs(qty):.0f}</td>'
                f'<td style="{td}">${p["avgCost"]:.2f}</td>'
                f'<td style="{td}">${p["marketPrice"]:.2f}</td>'
                f'<td style="{td}">${p["marketValue"]:,.0f}</td>'
                f'<td style="{td};color:{pnl_color};font-weight:bold">'
                f'{pnl_sign}${upnl:,.0f}</td>'
                f'</tr>'
            )

        return (
            f'<table style="width:100%;font-size:12px;border-collapse:collapse;'
            f'margin:4px 0">'
            f'<tr style="border-bottom:1px solid #333">'
            f'<th style="{th}">Symbol</th>'
            f'<th style="{th}">Side</th>'
            f'<th style="{th}">Qty</th>'
            f'<th style="{th}">Avg Cost</th>'
            f'<th style="{th}">Mkt Price</th>'
            f'<th style="{th}">Mkt Value</th>'
            f'<th style="{th}">Unreal P&L</th>'
            f'</tr>{rows}</table>'
        )

    # ── Order Form ───────────────────────────────────────────────────

    buy_btn = pn.widgets.Button(
        name='BUY', button_type='success', width=60, height=36, margin=(0, 2, 0, 0))
    sell_btn = pn.widgets.Button(
        name='SELL', button_type='default', width=60, height=36, margin=(0, 12, 0, 0))
    _direction = ['BUY']

    qty_input = pn.widgets.IntInput(
        value=100, start=1, step=10, width=80, height=36, margin=(0, 5, 0, 0))
    qty_label = pn.pane.HTML(
        '<span style="color:#888;font-size:11px">Shares</span>',
        width=40, margin=(0, 2, 0, 0), align='center')

    order_type_select = pn.widgets.Select(
        options=['Market', 'Limit', 'Stop'], value='Market',
        width=80, height=36, margin=(0, 5, 0, 0))
    type_label = pn.pane.HTML(
        '<span style="color:#888;font-size:11px">Type</span>',
        width=30, margin=(0, 2, 0, 0), align='center')

    session_select = pn.widgets.Select(
        options=['RTH', 'Extended Hours', 'Overnight'], value='RTH',
        width=120, height=36, margin=(0, 5, 0, 0))
    session_label = pn.pane.HTML(
        '<span style="color:#888;font-size:11px">Session</span>',
        width=45, margin=(0, 2, 0, 0), align='center')

    tif_select = pn.widgets.RadioButtonGroup(
        options=['DAY', 'GTC'], value='DAY',
        button_style='outline', button_type='default', margin=(0, 12, 0, 0))

    submit_btn = pn.widgets.Button(
        name='SUBMIT BUY', button_type='success', width=120, height=36,
        margin=(0, 5, 0, 0))

    status_msg = pn.pane.HTML('', width=200, margin=(0, 0, 0, 5), align='center')

    def _click_buy(event):
        _direction[0] = 'BUY'
        buy_btn.button_type = 'success'
        sell_btn.button_type = 'default'
        submit_btn.name = 'SUBMIT BUY'
        submit_btn.button_type = 'success'

    def _click_sell(event):
        _direction[0] = 'SELL'
        buy_btn.button_type = 'default'
        sell_btn.button_type = 'danger'
        submit_btn.name = 'SUBMIT SELL'
        submit_btn.button_type = 'danger'

    buy_btn.on_click(_click_buy)
    sell_btn.on_click(_click_sell)

    # ── Price Section (Limit/Stop only) ──────────────────────────────

    price_info = pn.pane.HTML('', sizing_mode='stretch_width',
                              margin=(0, 0, 0, 0))

    price_slider = pn.widgets.FloatSlider(
        name='', start=0.0, end=1.0, step=0.01, value=0.0,
        sizing_mode='stretch_width', show_value=False,
        margin=(0, 0, 0, 0))

    price_input = pn.widgets.FloatInput(
        value=0.0, step=0.01, width=90, height=36, format='0.00',
        margin=(0, 5, 0, 0))
    price_label = pn.pane.HTML(
        '<span style="color:#888;font-size:11px">Price $</span>',
        width=42, margin=(0, 2, 0, 0), align='center')

    lock_toggle = pn.widgets.Toggle(
        name='Unlocked', button_type='default', width=75, height=36,
        value=False, margin=(0, 0, 0, 0))

    slider_container = pn.Column(
        price_info,
        price_slider,
        pn.Row(price_label, price_input, lock_toggle, align='center',
               margin=(2, 0, 0, 0)),
        visible=False, margin=(8, 0, 0, 0),
        styles={'background': '#16213e', 'padding': '8px',
                'border-radius': '6px'},
        sizing_mode='stretch_width',
    )

    _locked = [False]
    _programmatic = [False]

    def _on_slider_change(event):
        if _programmatic[0]:
            return
        _locked[0] = True
        lock_toggle.value = True
        lock_toggle.name = 'Locked'
        lock_toggle.button_type = 'warning'
        price_input.value = round(event.new, 2)

    def _on_price_input_change(event):
        if _programmatic[0]:
            return
        _locked[0] = True
        lock_toggle.value = True
        lock_toggle.name = 'Locked'
        lock_toggle.button_type = 'warning'
        _programmatic[0] = True
        price_slider.value = event.new
        _programmatic[0] = False

    def _on_lock_toggle(event):
        _locked[0] = event.new
        if event.new:
            lock_toggle.name = 'Locked'
            lock_toggle.button_type = 'warning'
        else:
            lock_toggle.name = 'Unlocked'
            lock_toggle.button_type = 'default'

    price_slider.param.watch(_on_slider_change, 'value')
    price_input.param.watch(_on_price_input_change, 'value')
    lock_toggle.param.watch(_on_lock_toggle, 'value')

    def _on_order_type_change(event):
        slider_container.visible = event.new in ('Limit', 'Stop')

    order_type_select.param.watch(_on_order_type_change, 'value')

    # ── Submit Logic ─────────────────────────────────────────────────

    def _on_submit(event):
        if not state.ib_client or not state.ib_client.is_connected():
            status_msg.object = ('<span style="color:#ff5252;font-weight:bold">'
                                 'IB not connected</span>')
            return

        action = _direction[0]
        qty = qty_input.value
        otype = ORDER_TYPE_MAP[order_type_select.value]
        session = session_select.value
        tif = tif_select.value

        outside_rth = session != 'RTH'
        exchange = None
        if session == 'Overnight':
            tif = 'GTC'
            exchange = 'OVERNIGHT'
            if otype == 'MKT':
                otype = 'LMT'
                # Use mid price for overnight market orders (no true MKT on ATS)
                data = state.ib_client.get_price_data('TSLA')
                bid = data.get('bid', 0.0)
                ask = data.get('ask', 0.0)
                if bid > 0 and ask > 0:
                    price = round((bid + ask) / 2, 2)
                else:
                    status_msg.object = ('<span style="color:#ff5252;font-weight:bold">'
                                         'No bid/ask for overnight market order</span>')
                    return

        price_val = price
        if otype in ('LMT', 'STP') and price_val == 0.0:
            price_val = price_input.value
            if price_val <= 0:
                status_msg.object = ('<span style="color:#ff5252;font-weight:bold">'
                                     'Price must be > 0</span>')
                return

        result = state.ib_client.place_order(
            'TSLA', action, qty, order_type=otype, price=price_val,
            tif=tif, outside_rth=outside_rth, exchange=exchange)

        if 'error' in result:
            status_msg.object = (f'<span style="color:#ff5252;font-weight:bold">'
                                 f'Error: {result["error"]}</span>')
        else:
            status_msg.object = (
                f'<span style="color:#00e676;font-weight:bold">'
                f'Order #{result["order_id"]} {result["status"]}</span>')
            state.order_version += 1

        # Clear status message after 3 seconds
        def _clear_status():
            status_msg.object = ''
        pn.state.add_periodic_callback(_clear_status, period=3000, count=1)

    submit_btn.on_click(_on_submit)

    # ── Order Blotter ────────────────────────────────────────────────

    def _render_blotter(order_version):
        if not state.ib_client:
            return pn.pane.HTML(
                '<div style="color:#888;font-size:12px">IB not connected</div>')
        log = state.ib_client.get_order_log()
        if not log:
            return pn.pane.HTML(
                '<div style="color:#888;font-size:12px">No orders yet</div>')

        rows_html = ''
        pending_ids = []
        td = 'padding:2px 6px'
        for entry in reversed(log):
            status = entry['status']
            if status == 'Filled':
                color = '#00e676'
            elif status in ('Submitted', 'PreSubmitted', 'PendingSubmit'):
                color = '#ffab00'
                pending_ids.append(entry['order_id'])
            elif status == 'Cancelled':
                color = '#ff5252'
            else:
                color = '#aaa'

            price_str = f"${entry['price']:.2f}" if entry['price'] > 0 else '--'
            fill_str = f"${entry['fill_price']:.2f}" if entry['fill_price'] > 0 else '--'
            action_color = '#00e676' if entry['action'] == 'BUY' else '#ff5252'
            oid = entry['order_id']

            rows_html += (
                f'<tr>'
                f'<td style="{td};color:#888">#{oid}</td>'
                f'<td style="{td}">{entry["time"]}</td>'
                f'<td style="{td};color:{action_color};font-weight:bold">'
                f'{entry["action"]}</td>'
                f'<td style="{td}">{entry["qty"]}</td>'
                f'<td style="{td}">{entry["order_type"]}</td>'
                f'<td style="{td}">{price_str}</td>'
                f'<td style="{td};color:{color};font-weight:bold">{status}</td>'
                f'<td style="{td}">{fill_str}</td>'
                f'<td style="{td}">{entry["fill_time"]}</td>'
                f'</tr>'
            )

        th = 'padding:2px 6px;text-align:left'
        table_html = (
            f'<table style="width:100%;font-size:12px;border-collapse:collapse">'
            f'<tr style="border-bottom:1px solid #444">'
            f'<th style="{th}">ID</th>'
            f'<th style="{th}">Time</th>'
            f'<th style="{th}">Action</th>'
            f'<th style="{th}">Qty</th>'
            f'<th style="{th}">Type</th>'
            f'<th style="{th}">Price</th>'
            f'<th style="{th}">Status</th>'
            f'<th style="{th}">Fill</th>'
            f'<th style="{th}">Fill Time</th>'
            f'</tr>{rows_html}</table>'
        )

        items = [pn.pane.HTML(table_html, sizing_mode='stretch_width')]

        if pending_ids:
            cancel_btns = []
            for oid in pending_ids:
                btn = pn.widgets.Button(
                    name=f"Cancel #{oid}", button_type='danger',
                    width=100, height=26)

                def _make_cancel(order_id):
                    def _cancel(event):
                        state.ib_client.cancel_order(order_id)
                        state.order_version += 1
                    return _cancel

                btn.on_click(_make_cancel(oid))
                cancel_btns.append(btn)
            items.append(pn.Row(*cancel_btns, margin=(4, 0, 0, 0)))

        return pn.Column(*items)

    # ── Periodic Callback (250ms) ────────────────────────────────────

    _last_log_snapshot = [None]
    _acct_counter = [0]

    def _periodic_update():
        _acct_counter[0] += 1
        if _acct_counter[0] % 8 == 0:
            account_pane.object = _render_account()
            positions_pane.object = _render_positions()

        if state.ib_client:
            data = state.ib_client.get_price_data('TSLA')
            bid = data.get('bid', 0.0)
            ask = data.get('ask', 0.0)
            if bid > 0 and ask > 0:
                mid = round((bid + ask) / 2, 2)
                spread = ask - bid
                pad = max(0.50, spread * 2)

                price_info.object = (
                    f'<div style="display:flex;justify-content:space-between;'
                    f'font-size:13px;padding:0 4px">'
                    f'<span><b style="color:#00e676">Bid ${bid:.2f}</b></span>'
                    f'<span>Mid <b>${mid:.2f}</b>'
                    f'&nbsp;&nbsp;<span style="color:#888">Spd ${spread:.2f}</span></span>'
                    f'<span><b style="color:#ff5252">Ask ${ask:.2f}</b></span>'
                    f'</div>')

                if not _locked[0]:
                    _programmatic[0] = True
                    price_slider.start = round(bid - pad, 2)
                    price_slider.end = round(ask + pad, 2)
                    price_slider.value = mid
                    price_input.value = mid
                    _programmatic[0] = False
                # Locked: don't touch slider at all — knob and range stay frozen
                # Only bid/ask/mid labels above update (already done)

            log = state.ib_client.get_order_log()
            log_snapshot = [(e['order_id'], e['status']) for e in log]
            if log_snapshot != _last_log_snapshot[0]:
                _last_log_snapshot[0] = log_snapshot
                state.order_version += 1

    # ── Assemble ─────────────────────────────────────────────────────

    controls_row = pn.Row(
        buy_btn, sell_btn,
        qty_label, qty_input,
        type_label, order_type_select,
        session_label, session_select,
        tif_select,
        submit_btn, status_msg,
        align='center', margin=(8, 0, 4, 0),
    )

    form = pn.Column(
        pn.pane.HTML(
            '<b style="font-size:15px">Manual Order Entry (TSLA)</b>',
            margin=(0, 0, 4, 0)),
        account_pane,
        positions_pane,
        controls_row,
        slider_container,
        styles={'background': '#1a1a2e', 'padding': '10px 12px',
                'border-radius': '8px', 'border': '1px solid #333'},
        sizing_mode='stretch_width',
    )

    blotter = pn.Column(
        pn.pane.HTML('<b style="font-size:13px">Order Blotter</b>',
                     margin=(0, 0, 4, 0)),
        pn.bind(_render_blotter, state.param.order_version),
        styles={'background': '#1a1a2e', 'padding': '10px 12px',
                'border-radius': '8px', 'border': '1px solid #333'},
        sizing_mode='stretch_width',
    )

    result = pn.Column(form, blotter, sizing_mode='stretch_width',
                       margin=(0, 0, 5, 0))
    pn.state.add_periodic_callback(_periodic_update, period=250)
    return result
