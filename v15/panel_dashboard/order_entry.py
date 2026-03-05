"""Manual order entry panel for IB paper trading."""

import logging
import panel as pn

logger = logging.getLogger(__name__)

# Order type mapping for IB
ORDER_TYPE_MAP = {'Market': 'MKT', 'Limit': 'LMT', 'Stop': 'STP'}


def order_entry_panel(state) -> pn.Column:
    """Build the manual order entry panel with form, price slider, and blotter."""

    # ── Account Summary ─────────────────────────────────────────────

    account_pane = pn.pane.HTML('', sizing_mode='stretch_width')

    def _render_account():
        if not state.ib_client or not state.ib_client.is_connected():
            return ''
        acct = state.ib_client.get_account_summary()
        if not acct:
            return '<div style="color:#888;font-size:12px">Loading account...</div>'

        def _fmt(tag, prefix='$'):
            val = acct.get(tag, '')
            if not val:
                return '--'
            try:
                return f'{prefix}{float(val):,.0f}'
            except (ValueError, TypeError):
                return str(val)

        nlv = _fmt('NetLiquidation')
        cash = _fmt('TotalCashValue')
        bp = _fmt('BuyingPower')
        gross = _fmt('GrossPositionValue')
        upnl_raw = acct.get('UnrealizedPnL', '')
        rpnl_raw = acct.get('RealizedPnL', '')

        def _pnl_fmt(raw):
            if not raw:
                return '--', '#aaa'
            try:
                v = float(raw)
                color = '#00e676' if v >= 0 else '#ff5252'
                return f'${v:,.0f}', color
            except (ValueError, TypeError):
                return str(raw), '#aaa'

        upnl, upnl_c = _pnl_fmt(upnl_raw)
        rpnl, rpnl_c = _pnl_fmt(rpnl_raw)

        return (
            '<div style="display:flex;gap:24px;font-size:13px;padding:4px 0">'
            f'<span><b>Net Liq:</b> {nlv}</span>'
            f'<span><b>Cash:</b> {cash}</span>'
            f'<span><b>Buying Power:</b> {bp}</span>'
            f'<span><b>Positions:</b> {gross}</span>'
            f'<span><b>Unreal P&L:</b> <span style="color:{upnl_c}">{upnl}</span></span>'
            f'<span><b>Real P&L:</b> <span style="color:{rpnl_c}">{rpnl}</span></span>'
            '</div>'
        )

    # ── A. Order Form ────────────────────────────────────────────────

    direction_toggle = pn.widgets.RadioButtonGroup(
        name='Direction', options=['BUY', 'SELL'], value='BUY',
        button_style='outline', button_type='success')

    qty_input = pn.widgets.IntInput(
        name='Shares', value=100, start=1, step=10, width=120)

    order_type_select = pn.widgets.Select(
        name='Order Type', options=['Market', 'Limit', 'Stop'],
        value='Market', width=120)

    session_select = pn.widgets.Select(
        name='Session', options=['RTH', 'Extended Hours', 'Overnight'],
        value='RTH', width=140)

    tif_select = pn.widgets.RadioButtonGroup(
        name='TIF', options=['DAY', 'GTC'], value='DAY',
        button_style='outline', button_type='default')

    submit_btn = pn.widgets.Button(
        name='SUBMIT ORDER', button_type='success', width=160)

    status_msg = pn.pane.HTML('', height=30)

    # ── B. Price Slider (Limit/Stop only) ────────────────────────────

    bid_label = pn.pane.HTML('<b>Bid:</b> --', width=100)
    ask_label = pn.pane.HTML('<b>Ask:</b> --', width=100)
    mid_label = pn.pane.HTML('Mid: --', styles={'text-align': 'center'})

    price_slider = pn.widgets.FloatSlider(
        name='', start=0.0, end=1.0, step=0.01, value=0.0, width=300,
        format='0.00')

    price_input = pn.widgets.FloatInput(
        name='Price', value=0.0, step=0.01, width=120, format='0.00')

    lock_toggle = pn.widgets.Toggle(
        name='Unlocked', button_type='default', width=90, value=False)

    slider_container = pn.Column(
        pn.Row(bid_label, price_slider, ask_label),
        pn.Row(pn.Spacer(width=100), mid_label, pn.Spacer(width=100)),
        pn.Row(price_input, lock_toggle),
        visible=False,
    )

    # Slider lock state
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

    # Direction color styling
    def _on_direction_change(event):
        if event.new == 'BUY':
            direction_toggle.button_type = 'success'
            submit_btn.button_type = 'success'
            submit_btn.name = 'SUBMIT BUY'
        else:
            direction_toggle.button_type = 'danger'
            submit_btn.button_type = 'danger'
            submit_btn.name = 'SUBMIT SELL'

    direction_toggle.param.watch(_on_direction_change, 'value')
    submit_btn.name = 'SUBMIT BUY'  # initial

    # ── Submit Logic ─────────────────────────────────────────────────

    def _on_submit(event):
        if not state.ib_client or not state.ib_client.is_connected():
            status_msg.object = ('<span style="color:#ff5252;font-weight:bold">'
                                 'IB not connected</span>')
            return

        action = direction_toggle.value
        qty = qty_input.value
        otype = ORDER_TYPE_MAP[order_type_select.value]
        session = session_select.value
        tif = tif_select.value

        outside_rth = session != 'RTH'
        if session == 'Overnight':
            tif = 'GTC'

        price = 0.0
        if otype in ('LMT', 'STP'):
            price = price_input.value
            if price <= 0:
                status_msg.object = ('<span style="color:#ff5252;font-weight:bold">'
                                     'Price must be > 0 for Limit/Stop</span>')
                return

        result = state.ib_client.place_order(
            'TSLA', action, qty, order_type=otype, price=price,
            tif=tif, outside_rth=outside_rth)

        if 'error' in result:
            status_msg.object = (f'<span style="color:#ff5252;font-weight:bold">'
                                 f'Error: {result["error"]}</span>')
        else:
            status_msg.object = (
                f'<span style="color:#00e676;font-weight:bold">'
                f'Order #{result["order_id"]} {result["status"]}</span>')
            state.order_version += 1

    submit_btn.on_click(_on_submit)

    # ── C. Order Blotter ─────────────────────────────────────────────

    blotter_pane = pn.pane.HTML('', sizing_mode='stretch_width')
    cancel_row = pn.Row()

    def _render_blotter(order_version):
        if not state.ib_client:
            return '<div style="color:#888;font-size:12px">IB not connected</div>'
        log = state.ib_client.get_order_log()
        if not log:
            return '<div style="color:#888;font-size:12px">No orders yet</div>'

        rows_html = ''
        for entry in reversed(log):
            status = entry['status']
            if status == 'Filled':
                color = '#00e676'
            elif status in ('Submitted', 'PreSubmitted'):
                color = '#ffab00'
            elif status == 'Cancelled':
                color = '#ff5252'
            else:
                color = '#aaa'

            price_str = f"${entry['price']:.2f}" if entry['price'] > 0 else '--'
            fill_str = f"${entry['fill_price']:.2f}" if entry['fill_price'] > 0 else '--'
            action_color = '#00e676' if entry['action'] == 'BUY' else '#ff5252'

            rows_html += (
                f'<tr>'
                f'<td style="padding:2px 6px">{entry["time"]}</td>'
                f'<td style="padding:2px 6px;color:{action_color};font-weight:bold">{entry["action"]}</td>'
                f'<td style="padding:2px 6px">{entry["qty"]}</td>'
                f'<td style="padding:2px 6px">{entry["order_type"]}</td>'
                f'<td style="padding:2px 6px">{price_str}</td>'
                f'<td style="padding:2px 6px;color:{color};font-weight:bold">{status}</td>'
                f'<td style="padding:2px 6px">{fill_str}</td>'
                f'<td style="padding:2px 6px">{entry["fill_time"]}</td>'
                f'</tr>'
            )

        return (
            '<table style="width:100%;font-size:12px;border-collapse:collapse">'
            '<tr style="border-bottom:1px solid #444">'
            '<th style="padding:2px 6px;text-align:left">Time</th>'
            '<th style="padding:2px 6px;text-align:left">Action</th>'
            '<th style="padding:2px 6px;text-align:left">Qty</th>'
            '<th style="padding:2px 6px;text-align:left">Type</th>'
            '<th style="padding:2px 6px;text-align:left">Price</th>'
            '<th style="padding:2px 6px;text-align:left">Status</th>'
            '<th style="padding:2px 6px;text-align:left">Fill</th>'
            '<th style="padding:2px 6px;text-align:left">Fill Time</th>'
            '</tr>'
            f'{rows_html}'
            '</table>'
        )

    def _render_cancel_buttons(order_version):
        if not state.ib_client:
            return pn.Row()
        log = state.ib_client.get_order_log()
        buttons = []
        for entry in reversed(log):
            if entry['status'] in ('Submitted', 'PreSubmitted'):
                oid = entry['order_id']
                btn = pn.widgets.Button(
                    name=f"Cancel #{oid}", button_type='danger',
                    width=100, height=28)

                def _make_cancel(order_id):
                    def _cancel(event):
                        state.ib_client.cancel_order(order_id)
                        state.order_version += 1
                    return _cancel

                btn.on_click(_make_cancel(oid))
                buttons.append(btn)
        return pn.Row(*buttons) if buttons else pn.Row()

    # ── Periodic Callback (250ms) — update slider + poll order status ─

    _last_log_snapshot = [None]

    _acct_counter = [0]

    def _periodic_update():
        # Update account summary every ~2s (every 8th tick at 250ms)
        _acct_counter[0] += 1
        if _acct_counter[0] % 8 == 0:
            account_pane.object = _render_account()

        # Update bid/ask/mid slider
        if state.ib_client:
            data = state.ib_client.get_price_data('TSLA')
            bid = data.get('bid', 0.0)
            ask = data.get('ask', 0.0)
            if bid > 0 and ask > 0:
                mid = round((bid + ask) / 2, 2)
                spread = ask - bid
                pad = max(0.50, spread * 2)

                bid_label.object = f'<b>Bid:</b> ${bid:.2f}'
                ask_label.object = f'<b>Ask:</b> ${ask:.2f}'
                mid_label.object = f'Mid: ${mid:.2f}'

                if not _locked[0]:
                    _programmatic[0] = True
                    price_slider.start = round(bid - pad, 2)
                    price_slider.end = round(ask + pad, 2)
                    price_slider.value = mid
                    price_input.value = mid
                    _programmatic[0] = False
                else:
                    # Still update the range so slider doesn't clip
                    current_val = price_slider.value
                    new_start = round(min(bid - pad, current_val - 0.01), 2)
                    new_end = round(max(ask + pad, current_val + 0.01), 2)
                    _programmatic[0] = True
                    price_slider.start = new_start
                    price_slider.end = new_end
                    _programmatic[0] = False

            # Poll order log for status changes
            log = state.ib_client.get_order_log()
            log_snapshot = [(e['order_id'], e['status']) for e in log]
            if log_snapshot != _last_log_snapshot[0]:
                _last_log_snapshot[0] = log_snapshot
                state.order_version += 1

    # ── Assemble ─────────────────────────────────────────────────────

    form = pn.Column(
        pn.pane.HTML('<h4 style="margin:0 0 4px 0">Manual Order Entry (TSLA)</h4>'),
        account_pane,
        pn.Row(direction_toggle, qty_input, order_type_select,
               session_select, tif_select),
        slider_container,
        pn.Row(submit_btn, status_msg),
        styles={'background': '#1a1a2e', 'padding': '10px',
                'border-radius': '8px', 'border': '1px solid #333'},
    )

    blotter = pn.Column(
        pn.pane.HTML('<h4 style="margin:8px 0 4px 0">Order Blotter</h4>'),
        pn.bind(_render_blotter, state.param.order_version),
        pn.bind(_render_cancel_buttons, state.param.order_version),
        styles={'background': '#1a1a2e', 'padding': '10px',
                'border-radius': '8px', 'border': '1px solid #333'},
    )

    panel = pn.Column(form, blotter, sizing_mode='stretch_width')

    # Register periodic callback
    pn.state.add_periodic_callback(_periodic_update, period=250)

    return panel
