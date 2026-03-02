"""Dashboard state — central param.Parameterized class for reactive updates.

c12a: Single CS-5TF scanner only. No CS-DW, no intraday signals.
"""

import os
import logging
import threading
import traceback
from datetime import datetime

import param
import panel as pn
import pandas as pd

logger = logging.getLogger(__name__)


class DashboardState(param.Parameterized):
    """Central reactive state for the Panel dashboard.

    When params change, only bound components re-render.
    """

    # Price (updated every 5s by periodic callback via REST)
    tsla_price = param.Number(0.0)
    spy_price = param.Number(0.0)
    price_delta = param.Number(0.0)
    price_source = param.String('bar close')  # 'REST', 'bar close'

    # Market data (loaded on startup, refreshed every 2.5 min)
    current_tsla = param.Parameter(None)      # 5-min OHLCV DataFrame
    native_tf_data = param.Dict({})           # {symbol: {tf: DataFrame}}
    analysis = param.Parameter(None)          # ChannelAnalysis object
    last_analysis = param.String('')          # Timestamp string

    # Scanner (single CS-5TF only)
    scanner = param.Parameter(None)           # SurferLiveScanner
    positions_version = param.Integer(0)      # Bump for live P&L re-render (price-sensitive)
    trades_version = param.Integer(0)        # Bump only on actual trade open/close
    exit_alert_html = param.String('')        # Latest exit alert HTML

    # Config (sidebar widgets bind to these)
    scanner_capital = param.Number(100_000, bounds=(10_000, 1_000_000))
    kill_switch = param.Boolean(False)

    # Model comparisons (1-hour cache)
    model_data = param.Dict({})
    model_data_version = param.Integer(0)

    # Internal
    _prev_price = param.Number(0.0, precedence=-1)

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

        # Initialize scanner
        self._init_scanner()
        self._analysis_running = False

        # Run initial analysis (synchronous at startup — no UI to block yet)
        try:
            results = self._run_analysis_core()
            if results:
                self._apply_analysis_results(*results)
        except Exception as e:
            logger.error("Initial analysis failed: %s\n%s", e, traceback.format_exc())

    def update_prices(self):
        """5s periodic callback: poll REST for prices, check exits."""
        price = 0.0
        source = 'bar close'

        # REST quote via yfinance
        try:
            from v15.live_data import YFinanceLiveData
            rt = YFinanceLiveData().get_realtime_prices()
            if rt.get('TSLA'):
                price = float(rt['TSLA'])
                source = 'REST'
            if rt.get('SPY'):
                spy = float(rt['SPY'])
                if spy != self.spy_price:
                    self.spy_price = spy
        except Exception:
            pass

        # Fall back to bar close
        if price == 0.0 and self.current_tsla is not None and len(self.current_tsla) > 0:
            price = float(self.current_tsla['close'].iloc[-1])
            source = 'bar close'

        if price > 0:
            price_changed = price != self.tsla_price
            if self._prev_price > 0:
                new_delta = price - self._prev_price
                if new_delta != self.price_delta:
                    self.price_delta = new_delta
            self._prev_price = self.tsla_price if self.tsla_price > 0 else price
            if price_changed:
                self.tsla_price = price
            if source != self.price_source:
                self.price_source = source

        # Check exits on single scanner
        any_exits = False
        if self.scanner and self.scanner.positions and price > 0:
            try:
                exit_alerts = self.scanner.check_exits(price, price, price)
                if exit_alerts:
                    html_parts = [_exit_alert_html(ea) for ea in exit_alerts]
                    self.exit_alert_html = '\n'.join(html_parts)
                    any_exits = True
            except Exception as e:
                logger.warning("Exit check failed: %s", e)

        if any_exits:
            self.positions_version += 1
            self.trades_version += 1
        # Bump version for live P&L updates only when price actually changed
        elif price_changed if price > 0 else False:
            if self.scanner and self.scanner.positions:
                self.positions_version += 1

    def run_analysis(self):
        """2.5-min periodic callback + manual button: launch analysis in background thread."""
        if not self.native_tf_data:
            logger.warning("No native TF data available for analysis")
            return
        if self._analysis_running:
            logger.debug("Analysis already running, skipping")
            return

        self._analysis_running = True

        def _bg():
            try:
                results = self._run_analysis_core()
                if results:
                    pn.state.execute(lambda: self._apply_analysis_results(*results))
            except Exception as e:
                logger.error("Analysis failed: %s\n%s", e, traceback.format_exc())
            finally:
                self._analysis_running = False

        threading.Thread(target=_bg, daemon=True).start()

    def _run_analysis_core(self):
        """Heavy lifting: fetch bars + compute analysis. Runs in background thread.

        Returns (tsla_df, analysis) or None.
        """
        from v15.core.channel_surfer import prepare_multi_tf_analysis

        # Refresh 5-min bars
        tsla_df = None
        try:
            from v15.live_data import fetch_live_data
            tsla_df, spy_df, vix_df = fetch_live_data(period='5d', interval='5m')
        except Exception as e:
            logger.warning("Failed to refresh 5-min data: %s", e)

        effective_tsla = tsla_df if tsla_df is not None and len(tsla_df) > 0 else self.current_tsla

        analysis = prepare_multi_tf_analysis(
            native_data=self.native_tf_data,
            live_5min_tsla=effective_tsla,
            target_tfs=['5min', '1h', '4h', 'daily', 'weekly'],
        )

        return tsla_df, analysis

    def _apply_analysis_results(self, tsla_df, analysis):
        """Apply computed results on the main thread: update params + evaluate signals."""
        self._analysis_running = False

        if tsla_df is not None and len(tsla_df) > 0:
            self.current_tsla = tsla_df

        self.analysis = analysis
        self.last_analysis = datetime.now().strftime('%H:%M:%S')
        logger.info("Analysis complete: %s %s (%.0f%%)",
                     analysis.signal.action,
                     analysis.signal.primary_tf,
                     analysis.signal.confidence * 100)

        # Evaluate signal with single scanner
        if self.scanner and self.tsla_price > 0:
            sig = analysis.signal
            if sig.action != 'HOLD':
                entry_alert = self.scanner.evaluate_signal(
                    analysis, self.tsla_price)
                if entry_alert and entry_alert.alert_type == 'ENTRY':
                    self.positions_version += 1
                    self.trades_version += 1

    def _init_scanner(self):
        """Create a single SurferLiveScanner with Gist credentials from env vars."""
        try:
            from v15.trading.surfer_live_scanner import SurferLiveScanner, ScannerConfig
            config = ScannerConfig(initial_capital=self.scanner_capital)
            gist_id = os.environ.get('GIST_ID', '')
            github_token = os.environ.get('GITHUB_TOKEN', '')
            self.scanner = SurferLiveScanner(
                config, gist_id=gist_id, github_token=github_token,
            )
            logger.info("Scanner initialized (c12a, capital=$%,.0f)",
                         self.scanner_capital)
        except Exception as e:
            logger.error("Scanner init failed: %s", e)
            self.scanner = None

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
    return (
        f'<div style="background:{bg};padding:12px 16px;border-radius:8px;margin:6px 0;'
        f'border:2px solid {border};">'
        f'<span style="font-size:18px">{icon}</span> '
        f'<b style="color:{border};font-size:15px"> {label}</b> '
        f'<span style="color:#aaa">[{ea.pos_id}]</span> @ '
        f'<b>${ea.price:.2f}</b> &mdash; '
        f'P&L: <b style="color:{pnl_color}">${ea.pnl:+,.0f}</b> ({ea.pnl_pct:+.2%})'
        f'</div>'
    )
