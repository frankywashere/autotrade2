"""Dashboard state — central param.Parameterized class for reactive updates."""

import os
import logging
import traceback
from datetime import datetime

import param
import pandas as pd

logger = logging.getLogger(__name__)


class DashboardState(param.Parameterized):
    """Central reactive state for the Panel dashboard.

    When params change, only bound components re-render.
    """

    # Price (updated every 500ms by periodic callback)
    tsla_price = param.Number(0.0)
    spy_price = param.Number(0.0)
    price_delta = param.Number(0.0)
    price_source = param.String('bar close')  # 'LIVE', 'REST', 'bar close'

    # Market data (loaded on startup, refreshed every 5 min)
    current_tsla = param.Parameter(None)      # 5-min OHLCV DataFrame
    native_tf_data = param.Dict({})           # {symbol: {tf: DataFrame}}
    analysis = param.Parameter(None)          # ChannelAnalysis object
    last_analysis = param.String('')          # Timestamp string

    # Scanners
    scanner = param.Parameter(None)           # SurferLiveScanner (CS-5TF + intraday)
    scanner_dw = param.Parameter(None)        # SurferLiveScanner (CS-DW)
    positions_version = param.Integer(0)      # Bump to trigger position card re-render
    exit_alert_html = param.String('')        # Latest exit alert HTML

    # Config (sidebar widgets bind to these)
    scanner_capital = param.Number(100_000, bounds=(10_000, 1_000_000))
    kill_switch = param.Boolean(False)

    # Model comparisons (1-hour cache)
    model_data = param.Dict({})
    model_data_version = param.Integer(0)

    # Internal
    _prev_price = param.Number(0.0, precedence=-1)
    _ws_client = param.Parameter(None, precedence=-1)

    def load_market_data(self):
        """Startup: load native TF data, fetch 5-min bars, initialize scanner."""
        logger.info("Loading market data...")

        # Load native TF data
        try:
            from v15.data.native_tf import load_native_tf_data
            from pathlib import Path
            # Use /tmp for cache in Docker (home dir may not be writable on HF Spaces)
            cache_dir = Path('/tmp/.x14_native_tf_cache') if os.environ.get('SPACE_ID') else None
            self.native_tf_data = load_native_tf_data(
                symbols=['TSLA', 'SPY', '^VIX'],
                timeframes=['daily', 'weekly', 'monthly', '1h', '2h', '3h', '4h'],
                verbose=True,
                cache_dir=cache_dir,
            )
            logger.info("Native TF data loaded: %s", list(self.native_tf_data.keys()))
        except Exception as e:
            logger.error("Failed to load native TF data: %s", e)
            self.native_tf_data = {}

        # Fetch 5-min bars
        try:
            from v15.live_data import fetch_live_data
            tsla_df, spy_df, vix_df = fetch_live_data(period='5d', interval='5m')
            self.current_tsla = tsla_df
            if tsla_df is not None and len(tsla_df) > 0:
                self.tsla_price = float(tsla_df['close'].iloc[-1])
                # Use second-to-last bar for initial delta
                if len(tsla_df) > 1:
                    self._prev_price = float(tsla_df['close'].iloc[-2])
                    self.price_delta = self.tsla_price - self._prev_price
                else:
                    self._prev_price = self.tsla_price
            if spy_df is not None and len(spy_df) > 0:
                self.spy_price = float(spy_df['close'].iloc[-1])
            logger.info("5-min data loaded: %d bars", len(tsla_df) if tsla_df is not None else 0)
        except Exception as e:
            logger.error("Failed to fetch 5-min data: %s", e)

        # Initialize WebSocket client
        try:
            from v15.data.finnhub_ws import get_ws_client
            self._ws_client = get_ws_client()
            if self._ws_client:
                logger.info("WebSocket client initialized")
        except Exception:
            pass

        # Initialize scanner
        self._init_scanner()

        # Run initial analysis
        self.run_analysis()

    def update_prices(self):
        """500ms periodic callback: read WebSocket prices, check exits."""
        price = 0.0
        source = 'bar close'

        # Try WebSocket first
        if self._ws_client:
            try:
                tick = self._ws_client.get_price('TSLA')
                if tick and tick.price > 0:
                    price = tick.price
                    source = 'LIVE' if tick.is_fresh else f'WS ({tick.age_seconds:.0f}s ago)'
            except Exception:
                pass

            try:
                spy_tick = self._ws_client.get_price('SPY')
                if spy_tick and spy_tick.price > 0:
                    self.spy_price = spy_tick.price
            except Exception:
                pass

        # Fall back to REST
        if price == 0.0:
            try:
                from v15.live_data import YFinanceLiveData
                rt = YFinanceLiveData().get_realtime_prices()
                if rt.get('TSLA'):
                    price = float(rt['TSLA'])
                    source = 'REST'
                if rt.get('SPY'):
                    self.spy_price = float(rt['SPY'])
            except Exception:
                pass

        # Fall back to bar close
        if price == 0.0 and self.current_tsla is not None and len(self.current_tsla) > 0:
            price = float(self.current_tsla['close'].iloc[-1])
            source = 'bar close'

        if price > 0:
            if self._prev_price > 0:
                self.price_delta = price - self._prev_price
            self._prev_price = self.tsla_price if self.tsla_price > 0 else price
            self.tsla_price = price
            self.price_source = source

        # Check exits if positions are open (both scanners)
        html_parts = []
        any_exits = False
        for scnr in [self.scanner, self.scanner_dw]:
            if scnr and scnr.positions and price > 0:
                try:
                    exit_alerts = scnr.check_exits(price, price, price)
                    if exit_alerts:
                        for ea in exit_alerts:
                            html_parts.append(_exit_alert_html(ea))
                        any_exits = True
                except Exception as e:
                    logger.warning("Exit check failed: %s", e)
        if any_exits:
            self.exit_alert_html = '\n'.join(html_parts)
            self.positions_version += 1  # Re-render banner to clear exited positions
        # Bump version for live P&L updates
        has_positions = ((self.scanner and self.scanner.positions)
                         or (self.scanner_dw and self.scanner_dw.positions))
        if has_positions and price > 0:
            self.positions_version += 1

    def run_analysis(self):
        """5-min periodic callback + manual button: run Channel Surfer analysis."""
        if not self.native_tf_data:
            logger.warning("No native TF data available for analysis")
            return

        # Refresh 5-min bars
        try:
            from v15.live_data import fetch_live_data
            tsla_df, spy_df, vix_df = fetch_live_data(period='5d', interval='5m')
            if tsla_df is not None and len(tsla_df) > 0:
                self.current_tsla = tsla_df
        except Exception as e:
            logger.warning("Failed to refresh 5-min data: %s", e)

        # Run multi-TF analysis
        try:
            from v15.core.channel_surfer import prepare_multi_tf_analysis
            analysis = prepare_multi_tf_analysis(
                native_data=self.native_tf_data,
                live_5min_tsla=self.current_tsla,
                target_tfs=['5min', '1h', '4h', 'daily', 'weekly'],
            )
            self.analysis = analysis
            self.last_analysis = datetime.now().strftime('%H:%M:%S')
            logger.info("Analysis complete: %s %s (%.0f%%)",
                        analysis.signal.action,
                        analysis.signal.primary_tf,
                        analysis.signal.confidence * 100)

            # Evaluate signals if scanner is ready
            if self.scanner and self.tsla_price > 0:
                # --- CS-5TF: Original 5-timeframe signal ---
                sig = analysis.signal
                if sig.action != 'HOLD':
                    entry_alert = self.scanner.evaluate_signal(
                        analysis, self.tsla_price, signal_source='CS-5TF')
                    if entry_alert and entry_alert.alert_type == 'ENTRY':
                        self.positions_version += 1

                # --- CS-DW: Daily+Weekly only signal (separate scanner/model) ---
                if self.scanner_dw and self.tsla_price > 0:
                    try:
                        dw_analysis = prepare_multi_tf_analysis(
                            native_data=self.native_tf_data,
                            live_5min_tsla=self.current_tsla,
                            target_tfs=['daily', 'weekly'],
                        )
                        dw_sig = dw_analysis.signal
                        if dw_sig.action != 'HOLD':
                            dw_alert = self.scanner_dw.evaluate_signal(
                                dw_analysis, self.tsla_price, signal_source='CS-DW')
                            if dw_alert and dw_alert.alert_type == 'ENTRY':
                                self.positions_version += 1
                        logger.info("DW analysis: %s %s (%.0f%%)",
                                    dw_sig.action, dw_sig.primary_tf,
                                    dw_sig.confidence * 100)
                    except Exception as e:
                        logger.warning("DW analysis failed: %s", e)

                # Evaluate intraday signal
                self._evaluate_intraday(analysis)

        except Exception as e:
            logger.error("Analysis failed: %s\n%s", e, traceback.format_exc())

    def _init_scanner(self):
        """Create SurferLiveScanners with Gist credentials from env vars."""
        try:
            from v15.trading.surfer_live_scanner import SurferLiveScanner, ScannerConfig
            config = ScannerConfig(initial_capital=self.scanner_capital)
            gist_id = os.environ.get('GIST_ID', '')
            github_token = os.environ.get('GITHUB_TOKEN', '')
            self.scanner = SurferLiveScanner(
                config, gist_id=gist_id, github_token=github_token,
                model_tag='c13a',
            )
            self.scanner_dw = SurferLiveScanner(
                config, gist_id=gist_id, github_token=github_token,
                model_tag='c13a-dw',
            )
            logger.info("Scanners initialized (c13a + c13a-dw, capital=$%,.0f)",
                         self.scanner_capital)
        except Exception as e:
            logger.error("Scanner init failed: %s", e)
            self.scanner = None
            self.scanner_dw = None

    def _evaluate_intraday(self, analysis):
        """Extract 5-min features from analysis and evaluate intraday signal."""
        if analysis is None or not analysis.tf_states:
            return
        if self.current_tsla is None or len(self.current_tsla) == 0:
            return
        if self.scanner is None or self.tsla_price <= 0:
            return

        tf_states = analysis.tf_states
        state_5m = tf_states.get('5min')
        state_1h = tf_states.get('1h')
        state_4h = tf_states.get('4h')
        state_daily = tf_states.get('daily')

        if not state_5m or not state_5m.valid:
            return

        cp5 = state_5m.position_pct
        h1_cp = state_1h.position_pct if state_1h and state_1h.valid else float('nan')
        h4_cp = state_4h.position_pct if state_4h and state_4h.valid else float('nan')
        daily_cp = state_daily.position_pct if state_daily and state_daily.valid else float('nan')

        # Compute VWAP distance
        vwap_dist = float('nan')
        try:
            close_arr = self.current_tsla['close'].values
            high_arr = self.current_tsla['high'].values
            low_arr = self.current_tsla['low'].values
            vol_arr = self.current_tsla['volume'].values
            tp = (high_arr + low_arr + close_arr) / 3.0
            dates = self.current_tsla.index.date
            today = dates[-1]
            today_mask = dates == today
            if today_mask.sum() > 0:
                today_tp = tp[today_mask]
                today_vol = vol_arr[today_mask]
                cum_tv = (today_tp * today_vol).cumsum()
                cum_v = today_vol.cumsum()
                valid = cum_v > 0
                if valid.any():
                    vwap_val = cum_tv[valid][-1] / cum_v[valid][-1]
                    vwap_dist = (close_arr[-1] - vwap_val) / vwap_val * 100.0
        except Exception:
            pass

        # Channel slopes from TF states
        daily_slope = float('nan')
        h1_slope = float('nan')
        h4_slope = float('nan')
        try:
            if state_daily and hasattr(state_daily, 'channel_direction'):
                daily_slope = 1.0 if state_daily.channel_direction == 'bull' else (
                    -1.0 if state_daily.channel_direction == 'bear' else 0.0)
            if state_1h and hasattr(state_1h, 'channel_direction'):
                h1_slope = 1.0 if state_1h.channel_direction == 'bull' else (
                    -1.0 if state_1h.channel_direction == 'bear' else 0.0)
            if state_4h and hasattr(state_4h, 'channel_direction'):
                h4_slope = 1.0 if state_4h.channel_direction == 'bull' else (
                    -1.0 if state_4h.channel_direction == 'bear' else 0.0)
        except Exception:
            pass

        try:
            alert = self.scanner.evaluate_intraday_signal(
                current_price=self.tsla_price,
                cp5=cp5, vwap_dist=vwap_dist,
                daily_cp=daily_cp, h1_cp=h1_cp, h4_cp=h4_cp,
                daily_slope=daily_slope, h1_slope=h1_slope, h4_slope=h4_slope,
            )
            if alert and alert.alert_type == 'ENTRY':
                self.positions_version += 1
        except Exception as e:
            logger.warning("Intraday eval failed: %s", e)

    def load_model_data(self):
        """Fetch multi-model state from GitHub Gist API."""
        import json
        import urllib.request

        gist_id = os.environ.get('GIST_ID', '')
        github_token = os.environ.get('GITHUB_TOKEN', '')

        if not gist_id or not github_token:
            return

        try:
            headers = {
                'Authorization': f'token {github_token}',
                'Accept': 'application/vnd.github.v3+json',
            }
            url = f'https://api.github.com/gists/{gist_id}'
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as resp:
                gist_data = json.loads(resp.read().decode())
            content = gist_data.get('files', {}).get(
                'surfer_scanner_state.json', {}
            ).get('content', '')
            if content:
                self.model_data = json.loads(content)
                self.model_data_version += 1
                logger.info("Model data loaded: %d models",
                            len([k for k in self.model_data if not k.startswith('_')]))
        except Exception as e:
            logger.error("Failed to load model data: %s", e)


def _exit_alert_html(ea) -> str:
    """Build HTML for a single exit alert."""
    if ea.exit_reason == 'take_profit':
        bg, border, icon, label = '#0a3320', '#00e676', '&#127919;', 'TAKE PROFIT HIT'
    elif ea.exit_reason == 'stop_loss':
        bg, border, icon, label = '#3a0a0a', '#ff1744', '&#128721;', 'STOP LOSS HIT'
    elif ea.exit_reason == 'trailing_stop':
        bg, border, icon, label = '#2a1a0a', '#ff9100', '&#128200;', 'TRAILING STOP HIT'
    else:
        bg, border, icon, label = '#1a1a2e', '#888888', '&#9201;', ea.exit_reason.upper()
    pnl_color = '#00e676' if ea.pnl >= 0 else '#ff5252'
    source_tag = ''
    if getattr(ea, 'signal_source', ''):
        source_tag = (f' <span style="color:#64b5f6;font-size:12px;'
                      f'background:#1a237e;padding:2px 6px;border-radius:4px">'
                      f'{ea.signal_source}</span>')
    return (
        f'<div style="background:{bg};padding:12px 16px;border-radius:8px;margin:6px 0;'
        f'border:2px solid {border};">'
        f'<span style="font-size:18px">{icon}</span> '
        f'<b style="color:{border};font-size:15px"> {label}</b> '
        f'<span style="color:#aaa">[{ea.pos_id}]</span>{source_tag} @ '
        f'<b>${ea.price:.2f}</b> &mdash; '
        f'P&L: <b style="color:{pnl_color}">${ea.pnl:+,.0f}</b> ({ea.pnl_pct:+.2%})'
        f'</div>'
    )
