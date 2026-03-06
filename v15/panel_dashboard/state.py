"""Dashboard state — central param.Parameterized class for reactive updates."""

import os
import logging
import threading
import time
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

    # Price (updated every 500ms by periodic callback)
    tsla_price = param.Number(0.0)
    spy_price = param.Number(0.0)
    price_delta = param.Number(0.0)
    vix_price = param.Number(0.0)
    price_source = param.String('NONE')  # 'IB LIVE' or 'NONE'
    data_source = param.String('yfinance')  # 'IB' or 'yfinance'

    # Market data (loaded on startup, refreshed every 5 min)
    current_tsla = param.Parameter(None)      # 5-min OHLCV DataFrame
    native_tf_data = param.Dict({})           # {symbol: {tf: DataFrame}}
    analysis = param.Parameter(None)          # ChannelAnalysis object
    last_analysis = param.String('')          # Timestamp string

    # Scanners — c16 generation
    scanner = param.Parameter(None)           # SurferLiveScanner (CS-5TF)
    scanner_dw = param.Parameter(None)        # SurferLiveScanner (CS-DW)
    scanner_ml = param.Parameter(None)        # SurferLiveScanner (Surfer ML)
    scanner_intra = param.Parameter(None)     # SurferLiveScanner (Intraday ML)
    scanner_oe = param.Parameter(None)        # SurferLiveScanner (OE Signals_5)
    # Scanners — c14a generation (alongside c16)
    scanner_14a = param.Parameter(None)       # SurferLiveScanner (CS-5TF [14a])
    scanner_14a_dw = param.Parameter(None)    # SurferLiveScanner (CS-DW [14a])
    scanner_14a_ml = param.Parameter(None)    # SurferLiveScanner (Surfer ML [14a])
    scanner_14a_intra = param.Parameter(None) # SurferLiveScanner (Intraday [14a])
    # Scanners — yfinance A/B (parallel yf-backed copies of c16 + c14a)
    scanner_yf = param.Parameter(None)
    scanner_yf_dw = param.Parameter(None)
    scanner_yf_ml = param.Parameter(None)
    scanner_yf_intra = param.Parameter(None)
    scanner_yf_oe = param.Parameter(None)
    scanner_yf_14a = param.Parameter(None)
    scanner_yf_14a_dw = param.Parameter(None)
    scanner_yf_14a_ml = param.Parameter(None)
    scanner_yf_14a_intra = param.Parameter(None)
    positions_version = param.Integer(0)      # Bump to trigger position card re-render (price-sensitive)
    trades_version = param.Integer(0)        # Bump only on actual trade open/close (not price ticks)
    exit_alert_html = param.String('')        # Latest exit alert HTML

    # Config (sidebar widgets bind to these)
    scanner_capital = param.Number(100_000, bounds=(10_000, 1_000_000))
    kill_switch = param.Boolean(False)

    # Model comparisons (1-hour cache)
    model_data = param.Dict({})
    model_data_version = param.Integer(0)
    order_version = param.Integer(0)       # Bump on order submit/fill/cancel

    # IB connection
    ib_connected = param.Boolean(False)

    # Internal
    _prev_price = param.Number(0.0, precedence=-1)
    _ml_model = param.Parameter(None, precedence=-1)
    _ml_feature_names = param.Parameter(None, precedence=-1)
    _ml_history_buffer = param.Parameter(None, precedence=-1)
    _yf_ml_history_buffer = param.Parameter(None, precedence=-1)  # Separate buffer for yf A/B
    _intraday_ml_model = param.Parameter(None, precedence=-1)
    _intraday_ml_features = param.Parameter(None, precedence=-1)
    _intraday_trade_state = param.Parameter(None, precedence=-1)
    _yf_intraday_trade_state = param.Parameter(None, precedence=-1)  # Separate for yf A/B
    _el_model = param.Parameter(None, precedence=-1)   # ExtremeLoserDetector
    _er_model = param.Parameter(None, precedence=-1)   # ExtendedRunPredictor
    _ws_client = param.Parameter(None, precedence=-1)  # UNUSED — kept for param compat
    _price_updated_at = param.Number(0.0, precedence=-1)  # time.time() of last successful price fetch
    _last_intraday_exit_check = param.Number(0.0, precedence=-1)  # throttle intraday exits to 5s

    @property
    def _all_scanners(self):
        """All scanner instances (c16 + c14a + yf-*) for iteration."""
        return [s for s in [
            self.scanner, self.scanner_dw, self.scanner_ml,
            self.scanner_intra, self.scanner_oe,
            self.scanner_14a, self.scanner_14a_dw,
            self.scanner_14a_ml, self.scanner_14a_intra,
            self.scanner_yf, self.scanner_yf_dw, self.scanner_yf_ml,
            self.scanner_yf_intra, self.scanner_yf_oe,
            self.scanner_yf_14a, self.scanner_yf_14a_dw,
            self.scanner_yf_14a_ml, self.scanner_yf_14a_intra,
        ] if s is not None]

    @property
    def _main_scanners(self):
        """Non-yf scanners (c16 + c14a) — used for main tab UI updates."""
        return [s for s in [
            self.scanner, self.scanner_dw, self.scanner_ml,
            self.scanner_intra, self.scanner_oe,
            self.scanner_14a, self.scanner_14a_dw,
            self.scanner_14a_ml, self.scanner_14a_intra,
        ] if s is not None]

    @property
    def _yf_scanners(self):
        """yf-* scanners only — exits don't bump main tab versions."""
        return [s for s in [
            self.scanner_yf, self.scanner_yf_dw, self.scanner_yf_ml,
            self.scanner_yf_intra, self.scanner_yf_oe,
            self.scanner_yf_14a, self.scanner_yf_14a_dw,
            self.scanner_yf_14a_ml, self.scanner_yf_14a_intra,
        ] if s is not None]

    def load_market_data(self):
        """Startup: connect IB, load TF data (IB first, yfinance fallback), init scanner."""
        logger.info("Loading market data...")

        # Price feed: IB first
        self._ws_client = None
        self._price_err_count = 0
        self.ib_client = None
        self._bar_aggregator = None
        self._historical_5min_tsla = None
        self._init_ib()

        # Try IB-based data loading first, fall back to yfinance
        ib_data_ok = False
        if self.ib_client and self.ib_client.is_connected():
            try:
                self.native_tf_data = self._load_ib_historical()
                if self.native_tf_data and 'TSLA' in self.native_tf_data:
                    ib_data_ok = True
                    self.data_source = 'IB'
                    logger.info("Data source: IB (all TFs)")
            except Exception as e:
                logger.warning("IB historical load failed, falling back to yfinance: %s", e)

        if not ib_data_ok:
            self.data_source = 'yfinance'
            logger.info("Data source: yfinance fallback (degraded — no tick-level updates)")
            try:
                from v15.data.native_tf import load_native_tf_data
                from pathlib import Path
                cache_dir = Path('/tmp/.x14_native_tf_cache') if os.environ.get('SPACE_ID') else None
                self.native_tf_data = load_native_tf_data(
                    symbols=['TSLA', 'SPY', '^VIX'],
                    timeframes=['daily', 'weekly', 'monthly', '1h', '2h', '3h', '4h'],
                    verbose=True,
                    cache_dir=cache_dir,
                )
                logger.info("Native TF data loaded (yfinance): %s", list(self.native_tf_data.keys()))
            except Exception as e:
                logger.error("Failed to load native TF data: %s", e)
                self.native_tf_data = {}

        # Fetch 5-min bars (IB or yfinance)
        if ib_data_ok:
            try:
                tsla_5m = self.ib_client.fetch_historical('TSLA', '5 D', '5 mins', use_rth=False)
                if tsla_5m is not None and len(tsla_5m) > 0:
                    tsla_5m['date'] = pd.to_datetime(tsla_5m['date'])
                    tsla_5m = tsla_5m.set_index('date')
                    self.current_tsla = tsla_5m
                    self._historical_5min_tsla = tsla_5m.copy()
                    logger.info("5-min data loaded (IB): %d bars", len(tsla_5m))
                # Set up live 5-min bar aggregator for TSLA
                self._bar_aggregator = self.ib_client.create_bar_aggregator('TSLA', 5)
            except Exception as e:
                logger.warning("IB 5-min fetch failed: %s", e)

        if self.current_tsla is None:
            try:
                from v15.live_data import fetch_live_data
                tsla_df, spy_df, vix_df = fetch_live_data(period='5d', interval='5m')
                self.current_tsla = tsla_df
                if spy_df is not None and len(spy_df) > 0:
                    self.spy_price = float(spy_df['close'].iloc[-1])
                logger.info("5-min data loaded (yfinance): %d bars",
                            len(tsla_df) if tsla_df is not None else 0)
            except Exception as e:
                logger.error("Failed to fetch 5-min data: %s", e)

        # Set initial prices from loaded data
        if self.current_tsla is not None and len(self.current_tsla) > 0:
            self.tsla_price = float(self.current_tsla['close'].iloc[-1])
            if len(self.current_tsla) > 1:
                self._prev_price = float(self.current_tsla['close'].iloc[-2])
                self.price_delta = self.tsla_price - self._prev_price
            else:
                self._prev_price = self.tsla_price

        # Load yfinance data for A/B comparison (always, even when IB is primary)
        self._yf_native_tf_data = {}
        self._yf_current_tsla = None
        self._load_yf_data()

        # Initialize scanner
        self._init_scanner()
        self._analysis_running = False
        self._yf_analysis_running = False

        # Run initial analysis (synchronous at startup — no UI to block yet)
        try:
            results = self._run_analysis_core()
            if results:
                self._apply_analysis_results(*results)
        except Exception as e:
            logger.error("Initial analysis failed: %s\n%s", e, traceback.format_exc())

    def update_prices(self):
        """2s periodic callback: fetch prices via IB (no fallback)."""
        price = 0.0
        source = 'NONE'

        # Try IB first
        if self.ib_client and self.ib_client.is_connected():
            ib_price = self.ib_client.get_last_price('TSLA')
            if ib_price > 0:
                price = ib_price
                source = 'IB LIVE'
                if not self.ib_connected:
                    self.ib_connected = True
            ib_spy = self.ib_client.get_last_price('SPY')
            if ib_spy > 0 and ib_spy != self.spy_price:
                self.spy_price = ib_spy
            ib_vix = self.ib_client.get_last_price('VIX')
            if ib_vix > 0:
                self.vix_price = ib_vix
        elif self.ib_client and not self.ib_client.is_connected():
            if self.ib_connected:
                self.ib_connected = False
                logger.error("IB DISCONNECTED — no price source available!")

        # NO FALLBACKS — IB is the only price source.
        if price == 0.0:
            self._price_err_count += 1
            if self._price_err_count <= 5 or self._price_err_count % 10 == 0:
                logger.error("NO LIVE PRICE — IB returned 0 (count=%d). "
                             "Check IB Gateway connection. No fallback.", self._price_err_count)
            source = 'NONE'
            # Socket connected but no prices flowing — enable Reconnect button
            if self._price_err_count >= 10 and self.ib_connected:
                self.ib_connected = False
                logger.warning("IB socket up but no prices for %d checks — marking disconnected",
                               self._price_err_count)

        # Always update source label (so UI shows NONE when no price)
        if source != self.price_source:
            self.price_source = source

        if price > 0:
            self._price_updated_at = time.time()
            self._price_err_count = 0
            price_changed = price != self.tsla_price
            if self._prev_price > 0:
                new_delta = price - self._prev_price
                if new_delta != self.price_delta:
                    self.price_delta = new_delta
            self._prev_price = self.tsla_price if self.tsla_price > 0 else price
            if price_changed:
                self.tsla_price = price

        # Check exits — main scanners bump UI versions, yf scanners don't
        # Intraday scanners throttled to every 5s (tick noise causes instant stopouts)
        html_parts = []
        main_exit_alerts = []
        yf_exit_alerts = []
        now_ts = time.time()
        intraday_due = (now_ts - self._last_intraday_exit_check) >= 5.0
        intraday_scanners = {self.scanner_intra, self.scanner_14a_intra,
                             self.scanner_yf_intra, self.scanner_yf_14a_intra}
        if intraday_due:
            self._last_intraday_exit_check = now_ts
        for scnr in self._main_scanners:
            if scnr.positions and price > 0:
                if scnr in intraday_scanners and not intraday_due:
                    continue
                try:
                    exit_alerts = scnr.check_exits(price, price, price)
                    if exit_alerts:
                        for ea in exit_alerts:
                            html_parts.append(_exit_alert_html(ea))
                            main_exit_alerts.append(ea)
                except Exception as e:
                    logger.warning("Exit check failed: %s", e)
        for scnr in self._yf_scanners:
            if scnr.positions and price > 0:
                if scnr in intraday_scanners and not intraday_due:
                    continue
                try:
                    exit_alerts = scnr.check_exits(price, price, price)
                    if exit_alerts:
                        yf_exit_alerts.extend(exit_alerts)
                except Exception as e:
                    logger.warning("yf exit check failed: %s", e)

        if main_exit_alerts:
            self.exit_alert_html = '\n'.join(html_parts)
            self.positions_version += 1
            self.trades_version += 1
            self.load_model_data()
            # Update IB intraday ML trade state from intraday exits
            if self._intraday_trade_state:
                for ea in main_exit_alerts:
                    if getattr(ea, 'signal_source', '') == 'intraday':
                        ts = self._intraday_trade_state
                        if getattr(ea, 'pnl', 0) > 0:
                            ts['consec_wins'] += 1
                            ts['consec_losses'] = 0
                        else:
                            ts['consec_losses'] += 1
                            ts['consec_wins'] = 0

        if yf_exit_alerts:
            self.load_model_data()  # Update model comparisons tab only
            # Update yf-specific intraday trade state
            if self._yf_intraday_trade_state:
                for ea in yf_exit_alerts:
                    if getattr(ea, 'signal_source', '') == 'intraday':
                        ts = self._yf_intraday_trade_state
                        if getattr(ea, 'pnl', 0) > 0:
                            ts['consec_wins'] += 1
                            ts['consec_losses'] = 0
                        else:
                            ts['consec_losses'] += 1
                            ts['consec_wins'] = 0
        # Bump version for live P&L updates only when price actually changed
        elif not all_exit_alerts and (price_changed if price > 0 else False):
            has_positions = any(s.positions for s in self._all_scanners)
            if has_positions:
                self.positions_version += 1

    def run_analysis(self):
        """5-min periodic callback + manual button: launch analysis in background thread."""
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
                    # Schedule UI/param updates on the main Bokeh thread
                    pn.state.execute(lambda: self._apply_analysis_results(*results))
            except Exception as e:
                logger.error("Analysis failed: %s\n%s", e, traceback.format_exc())
            finally:
                self._analysis_running = False

        threading.Thread(target=_bg, daemon=True).start()

    def _run_analysis_core(self):
        """Heavy lifting: fetch bars + compute analysis. Runs in background thread.

        Returns (tsla_df, analysis, dw_analysis) or None.
        """
        from v15.core.channel_surfer import prepare_multi_tf_analysis

        # Refresh 5-min bars: IB aggregator first, then yfinance fallback
        tsla_df = None
        if self._bar_aggregator and self._historical_5min_tsla is not None:
            live_bars = self._bar_aggregator.get_bars_df(include_current=False)
            if len(live_bars) > 0:
                tsla_df = pd.concat([self._historical_5min_tsla, live_bars])
                tsla_df = tsla_df[~tsla_df.index.duplicated(keep='last')]
                logger.info("5-min bars: %d historical + %d live = %d total",
                           len(self._historical_5min_tsla), len(live_bars), len(tsla_df))
            else:
                tsla_df = self._historical_5min_tsla
        if tsla_df is None:
            try:
                from v15.live_data import fetch_live_data
                tsla_df, spy_df, vix_df = fetch_live_data(period='5d', interval='5m')
            except Exception as e:
                logger.warning("Failed to refresh 5-min data: %s", e)

        effective_tsla = tsla_df if tsla_df is not None and len(tsla_df) > 0 else self.current_tsla

        # CS-5TF analysis
        analysis = prepare_multi_tf_analysis(
            native_data=self.native_tf_data,
            live_5min_tsla=effective_tsla,
            target_tfs=['5min', '1h', '4h', 'daily', 'weekly'],
        )

        # CS-DW analysis
        dw_analysis = None
        if self.scanner_dw:
            dw_analysis = prepare_multi_tf_analysis(
                native_data=self.native_tf_data,
                live_5min_tsla=effective_tsla,
                target_tfs=['daily', 'weekly'],
            )

        return tsla_df, analysis, dw_analysis

    def _apply_analysis_results(self, tsla_df, analysis, dw_analysis):
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

        # Evaluate signals if scanner is ready and price is fresh (< 5s old)
        price_age = time.time() - self._price_updated_at if self._price_updated_at > 0 else 999
        if price_age > 5:
            logger.warning("Skipping signal eval — price is %.1fs stale (limit 5s)", price_age)
        elif self.scanner and self.tsla_price > 0:
            self._apply_analysis_results_with(
                tsla_df=self.current_tsla,
                analysis=analysis,
                dw_analysis=dw_analysis,
                native_tf_data=self.native_tf_data,
                scanner_cs=self.scanner,
                scanner_dw=self.scanner_dw,
                scanner_ml=self.scanner_ml,
                scanner_intra=self.scanner_intra,
                scanner_14a=self.scanner_14a,
                scanner_14a_dw=self.scanner_14a_dw,
                scanner_14a_ml=self.scanner_14a_ml,
                scanner_14a_intra=self.scanner_14a_intra,
                label='IB',
            )
            # Evaluate OE Signals_5 (daily bars — IB only, no yf equivalent)
            self._evaluate_oe_signals5()

    def _apply_analysis_results_with(self, *, tsla_df, analysis, dw_analysis,
                                      native_tf_data, scanner_cs, scanner_dw,
                                      scanner_ml, scanner_intra,
                                      scanner_14a, scanner_14a_dw,
                                      scanner_14a_ml, scanner_14a_intra,
                                      label='',
                                      ml_history_buffer=None,
                                      intraday_trade_state=None,
                                      price_override=None):
        """Shared signal evaluation logic for both IB and yfinance paths."""
        price = price_override if price_override and price_override > 0 else self.tsla_price
        sig = analysis.signal
        # --- CS-5TF ---
        if sig.action != 'HOLD':
            entry_alert = scanner_cs.evaluate_signal(
                analysis, price, signal_source='CS-5TF')
            if entry_alert and entry_alert.alert_type == 'ENTRY':
                self.positions_version += 1
                self.trades_version += 1
                self.load_model_data()
            if scanner_14a:
                a14 = scanner_14a.evaluate_signal(
                    analysis, price, signal_source='CS-5TF')
                if a14 and a14.alert_type == 'ENTRY':
                    self.positions_version += 1
                    self.trades_version += 1
                    self.load_model_data()

        # --- CS-DW ---
        if scanner_dw and dw_analysis and price > 0:
            try:
                dw_sig = dw_analysis.signal
                if dw_sig.action != 'HOLD':
                    dw_alert = scanner_dw.evaluate_signal(
                        dw_analysis, price, signal_source='CS-DW')
                    if dw_alert and dw_alert.alert_type == 'ENTRY':
                        self.positions_version += 1
                        self.trades_version += 1
                        self.load_model_data()
                    if scanner_14a_dw:
                        a14dw = scanner_14a_dw.evaluate_signal(
                            dw_analysis, price, signal_source='CS-DW')
                        if a14dw and a14dw.alert_type == 'ENTRY':
                            self.positions_version += 1
                            self.trades_version += 1
                            self.load_model_data()
                logger.info("[%s] DW analysis: %s %s (%.0f%%)",
                            label, dw_sig.action, dw_sig.primary_tf,
                            dw_sig.confidence * 100)
            except Exception as e:
                logger.warning("[%s] DW analysis failed: %s", label, e)

        # --- Surfer ML ---
        self._evaluate_surfer_ml_with(
            analysis=analysis, current_tsla=tsla_df,
            native_tf_data=native_tf_data,
            scanner_ml=scanner_ml, scanner_14a_ml=scanner_14a_ml,
            label=label,
            history_buffer=ml_history_buffer,
            price_override=price_override,
        )

        # --- Intraday ---
        self._evaluate_intraday_with(
            analysis=analysis, current_tsla=tsla_df,
            scanner_intra=scanner_intra, scanner_14a_intra=scanner_14a_intra,
            label=label,
            intraday_trade_state=intraday_trade_state,
            price_override=price_override,
        )

    def start_background_loops(self):
        """Start daemon threads for price/analysis/model — run without browser."""
        if getattr(self, '_bg_started', False):
            return
        self._bg_started = True

        def _price_loop():
            _min_interval = 0.1  # 100ms throttle (~10 updates/sec max)
            _last_update = 0.0
            while True:
                # Wait for IB tick (instant) or timeout at 2s (disconnected fallback)
                if self.ib_client and hasattr(self.ib_client, 'tick_event'):
                    self.ib_client.tick_event.wait(timeout=2.0)
                    self.ib_client.tick_event.clear()
                else:
                    time.sleep(2)
                # Throttle: skip if <100ms since last update
                now = time.time()
                if now - _last_update < _min_interval:
                    continue
                _last_update = now
                try:
                    self.update_prices()
                except Exception as e:
                    logger.error("Price loop error: %s", e)

        def _analysis_loop():
            time.sleep(30)  # Initial delay — let prices stabilize
            while True:
                if self._bar_aggregator:
                    # Wait for 5-min bar close (instant trigger) or timeout at 300s
                    self._bar_aggregator.bar_close_event.wait(timeout=300)
                    self._bar_aggregator.bar_close_event.clear()
                else:
                    time.sleep(150)  # fallback if no aggregator
                try:
                    self._run_analysis_bg()
                except Exception as e:
                    logger.error("Analysis loop error: %s", e)

        def _model_loop():
            while True:
                time.sleep(3600)
                try:
                    self.load_model_data()
                except Exception as e:
                    logger.error("Model reload error: %s", e)

        def _tf_refresh_loop():
            """Refresh higher TF bars from IB every 30 min."""
            while True:
                time.sleep(1800)
                if self.ib_client and self.ib_client.is_connected() and self.data_source == 'IB':
                    try:
                        new_data = self._load_ib_historical()
                        if new_data and 'TSLA' in new_data:
                            self.native_tf_data = new_data
                            logger.info("Higher TF data refreshed from IB")
                    except Exception as e:
                        logger.warning("Higher TF refresh failed: %s", e)

        def _yf_analysis_loop():
            """yfinance A/B: 150s timer, staggered 45s from IB loop."""
            time.sleep(75)  # Initial delay (staggered from IB's 30s)
            while True:
                time.sleep(150)
                try:
                    self._run_yf_analysis_bg()
                except Exception as e:
                    logger.error("yf A/B analysis loop error: %s", e)

        def _yf_tf_refresh_loop():
            """Refresh yfinance higher TF data every 30 min."""
            while True:
                time.sleep(1800)
                try:
                    self._load_yf_data()
                except Exception as e:
                    logger.warning("yf A/B TF refresh failed: %s", e)

        loops = [(_price_loop, 'price'), (_analysis_loop, 'analysis'),
                 (_model_loop, 'model'), (_tf_refresh_loop, 'tf-refresh'),
                 (_yf_analysis_loop, 'yf-analysis'),
                 (_yf_tf_refresh_loop, 'yf-tf-refresh')]
        for fn, name in loops:
            t = threading.Thread(target=fn, daemon=True, name=f'x14-{name}')
            t.start()
        logger.info("Background loops started (price/analysis/model/tf-refresh/yf-analysis/yf-tf-refresh)")

    def _run_analysis_bg(self):
        """Background-safe analysis: compute + apply without requiring Panel session."""
        if not self.native_tf_data:
            return
        if self._analysis_running:
            return
        self._analysis_running = True
        try:
            results = self._run_analysis_core()
            if results:
                # Try Bokeh thread first (if browser connected), else direct
                try:
                    pn.state.execute(lambda r=results: self._apply_analysis_results(*r))
                except Exception:
                    self._apply_analysis_results(*results)
        except Exception as e:
            logger.error("Analysis failed: %s\n%s", e, traceback.format_exc())
        finally:
            self._analysis_running = False

    def _load_ib_historical(self):
        """Load multi-TF historical data from IB for all symbols."""
        data = {}
        for symbol in ['TSLA', 'SPY']:
            tf_data = self.ib_client.fetch_all_tf_history(symbol)
            # Resample 2h/3h/4h from 1h
            if '1h' in tf_data:
                tf_data['2h'] = self._resample_ohlcv(tf_data['1h'], '2h')
                tf_data['3h'] = self._resample_ohlcv(tf_data['1h'], '3h')
                tf_data['4h'] = self._resample_ohlcv(tf_data['1h'], '4h')
            data[symbol] = tf_data

        # VIX: fetch daily only (all we need — 2 ML features: level + 5d change)
        vix_data = self.ib_client.fetch_all_tf_history('VIX', use_rth=True)
        data['^VIX'] = vix_data

        # Log summary
        for sym, tfs in data.items():
            tf_summary = ", ".join(f"{tf}={len(df)}" for tf, df in tfs.items() if len(df) > 0)
            logger.info("IB historical %s: %s", sym, tf_summary)
        return data

    def _load_yf_data(self):
        """Load native TF data + 5-min bars from yfinance for A/B comparison."""
        try:
            from v15.data.native_tf import load_native_tf_data
            from pathlib import Path
            cache_dir = Path('/tmp/.x14_native_tf_cache') if os.environ.get('SPACE_ID') else None
            self._yf_native_tf_data = load_native_tf_data(
                symbols=['TSLA', 'SPY', '^VIX'],
                timeframes=['daily', 'weekly', 'monthly', '1h', '2h', '3h', '4h'],
                verbose=False,
                cache_dir=cache_dir,
            )
            logger.info("yfinance A/B: TF data loaded: %s", list(self._yf_native_tf_data.keys()))
        except Exception as e:
            logger.warning("yfinance A/B: TF data load failed: %s", e)
            self._yf_native_tf_data = {}

        try:
            from v15.live_data import fetch_live_data
            tsla_df, spy_df, vix_df = fetch_live_data(period='5d', interval='5m')
            self._yf_current_tsla = tsla_df
            logger.info("yfinance A/B: 5-min data loaded: %d bars",
                        len(tsla_df) if tsla_df is not None else 0)
        except Exception as e:
            logger.warning("yfinance A/B: 5-min data load failed: %s", e)

    @staticmethod
    def _resample_ohlcv(df, rule):
        """Resample OHLCV data to a coarser timeframe."""
        return df.resample(rule).agg(
            {'open': 'first', 'high': 'max', 'low': 'min',
             'close': 'last', 'volume': 'sum'}
        ).dropna(subset=['close'])

    def _init_ib(self):
        """Connect to IB Gateway for real-time price streaming. Fails loudly."""
        try:
            import random
            from v15.ib.client import IBClient
            cid = random.randint(10, 99)  # Avoid stale client_id=1 conflicts
            self.ib_client = IBClient(host='127.0.0.1', port=4002, client_id=cid)
            self.ib_client.connect()
            self.ib_client.subscribe('TSLA')
            self.ib_client.subscribe('SPY')
            self.ib_client.subscribe('VIX')
            self.ib_connected = True
            logger.info("IB connected — streaming TSLA + SPY + VIX")
        except Exception as e:
            logger.error("IB CONNECTION FAILED — no live price source! %s", e)
            self.ib_client = None
            self.ib_connected = False

    def _init_scanner(self):
        """Create SurferLiveScanners with per-model local state files."""
        try:
            from v15.trading.surfer_live_scanner import SurferLiveScanner, ScannerConfig

            # --- c16 generation: flat $100K, trail^12, full-day intraday, no AM block ---
            config = ScannerConfig(initial_capital=self.scanner_capital)
            self.scanner = SurferLiveScanner(config, model_tag='c16')
            self.scanner_dw = SurferLiveScanner(config, model_tag='c16-dw')
            ml_config = ScannerConfig(
                initial_capital=self.scanner_capital,
                min_confidence=0.01,
            )
            self.scanner_ml = SurferLiveScanner(ml_config, model_tag='c16-ml')
            self.scanner_intra = SurferLiveScanner(config, model_tag='c16-intra')
            self.scanner_oe = SurferLiveScanner(config, model_tag='c16-oe')

            # --- c14a generation: confidence/risk sizing, trail^8, PM intraday, AM block ---
            c14a_config = ScannerConfig(
                initial_capital=self.scanner_capital,
                trail_power=8,
                flat_sizing=False,
                am_block_hour=10,           # Block entries after 10:30 ET
                intraday_start_hour=13,     # PM-only intraday window
                intraday_start_minute=0,
            )
            self.scanner_14a = SurferLiveScanner(c14a_config, model_tag='c14a')
            self.scanner_14a_dw = SurferLiveScanner(c14a_config, model_tag='c14a-dw')
            c14a_ml_config = ScannerConfig(
                initial_capital=self.scanner_capital,
                min_confidence=0.01,
                trail_power=8,
                flat_sizing=False,
                am_block_hour=10,
                intraday_start_hour=13,
                intraday_start_minute=0,
            )
            self.scanner_14a_ml = SurferLiveScanner(c14a_ml_config, model_tag='c14a-ml')
            self.scanner_14a_intra = SurferLiveScanner(c14a_config, model_tag='c14a-intra')

            # Ensure state files exist for all scanners
            all_scanners = [
                self.scanner, self.scanner_dw, self.scanner_ml,
                self.scanner_intra, self.scanner_oe,
                self.scanner_14a, self.scanner_14a_dw,
                self.scanner_14a_ml, self.scanner_14a_intra,
            ]
            for scnr in all_scanners:
                scnr._save_state()
            logger.info("Scanners initialized: 5x c16 + 4x c14a (capital=$%.0f)",
                         self.scanner_capital)
        except Exception as e:
            self._scanner_init_error = f"{e}"
            logger.error("Scanner init failed: %s\n%s", e, traceback.format_exc())
            self.scanner = None
            self.scanner_dw = None
            self.scanner_ml = None
            self.scanner_intra = None
            self.scanner_oe = None
            self.scanner_14a = None
            self.scanner_14a_dw = None
            self.scanner_14a_ml = None
            self.scanner_14a_intra = None

        # Initialize yfinance A/B scanners (identical configs, yf- prefixed tags)
        self._init_yf_scanners()

        # Load ML model for Surfer ML path
        # NOTE: Cannot import surfer_ml directly — it pulls in torch which isn't on HF.
        # Instead, load the pickle directly and wrap in a minimal GBT shim.
        try:
            import pickle
            from pathlib import Path
            import numpy as np
            model_path = Path('surfer_models/gbt_model.pkl')
            logger.info("GBT model check: cwd=%s, path=%s, exists=%s",
                        Path.cwd(), model_path, model_path.exists())
            if not model_path.exists():
                model_path = Path(__file__).parent.parent.parent / 'surfer_models' / 'gbt_model.pkl'
                logger.info("GBT fallback path: %s, exists=%s", model_path, model_path.exists())
            if model_path.exists():
                fsize = model_path.stat().st_size
                logger.info("GBT model file: %s (%d bytes)", model_path, fsize)
                if fsize < 200:
                    with open(model_path, 'rb') as f:
                        head = f.read(200)
                    logger.warning("GBT model file too small (%d bytes) — likely LFS pointer: %s",
                                   fsize, head[:100])
                else:
                    with open(model_path, 'rb') as f:
                        data = pickle.load(f)
                    # Build a lightweight GBT wrapper (avoids importing surfer_ml + torch)
                    class _GBTShim:
                        def __init__(self, models, feature_names):
                            self.models = models
                            self.feature_names = feature_names
                        def predict(self, X):
                            results = {}
                            for name, mdl in self.models.items():
                                results[name] = mdl.predict(X)
                            return results
                    self._ml_model = _GBTShim(data['models'], data['feature_names'])
                    self._ml_feature_names = data['feature_names']
                    self._ml_history_buffer = []
                    self._yf_ml_history_buffer = []
                    # Verify with a test prediction
                    test_pred = self._ml_model.predict(
                        np.zeros((1, len(self._ml_feature_names)), dtype=np.float32))
                    logger.info("ML model loaded: %d features, keys=%s",
                                len(self._ml_feature_names), list(test_pred.keys()))
            else:
                logger.warning("ML model not found at %s", model_path)
        except Exception as e:
            self._gbt_load_error = f"{e}"
            logger.warning("ML model load failed: %s — %s", e, traceback.format_exc())
            self._ml_model = None

        # Load EL (Extreme Loser) + ER (Extended Run) sub-models
        try:
            import pickle
            from pathlib import Path
            from v15.core.surfer_ml import ExtremeLoserDetector, ExtendedRunPredictor

            el_path = Path('surfer_models/extreme_loser_model.pkl')
            if not el_path.exists():
                el_path = Path(__file__).parent.parent.parent / 'surfer_models' / 'extreme_loser_model.pkl'
            if el_path.exists() and el_path.stat().st_size > 200:
                self._el_model = ExtremeLoserDetector.load(str(el_path))
                logger.info("EL model loaded from %s", el_path)
            else:
                logger.warning("EL model not found at %s", el_path)

            er_path = Path('surfer_models/extended_run_model.pkl')
            if not er_path.exists():
                er_path = Path(__file__).parent.parent.parent / 'surfer_models' / 'extended_run_model.pkl'
            if er_path.exists() and er_path.stat().st_size > 200:
                self._er_model = ExtendedRunPredictor.load(str(er_path))
                logger.info("ER model loaded from %s", er_path)
            else:
                logger.warning("ER model not found at %s", er_path)
        except Exception as e:
            logger.warning("EL/ER model load failed: %s — %s", e, traceback.format_exc())

        # Load intraday ML model (LightGBM filter for intraday signals)
        try:
            import pickle
            from pathlib import Path
            intra_data = None
            intra_path = Path('surfer_models/intraday_ml_model.pkl')
            logger.info("Intraday model check: path=%s, exists=%s", intra_path, intra_path.exists())
            if not intra_path.exists():
                intra_path = Path(__file__).parent.parent.parent / 'surfer_models' / 'intraday_ml_model.pkl'
                logger.info("Intraday fallback path: %s, exists=%s", intra_path, intra_path.exists())
            if intra_path.exists():
                fsize = intra_path.stat().st_size
                logger.info("Intraday model file: %s (%d bytes)", intra_path, fsize)
                if fsize < 200:
                    with open(intra_path, 'rb') as f:
                        head = f.read(200)
                    logger.warning("Intraday model file too small (%d bytes) — likely LFS pointer: %s",
                                   fsize, head[:100])
                else:
                    with open(intra_path, 'rb') as f:
                        intra_data = pickle.load(f)
                    logger.info("Intraday ML model loaded from file: %s", intra_path)
            else:
                # Fall back to embedded base64 model (for HF Spaces)
                try:
                    import base64
                    from v15.trading.intraday_ml_data import MODEL_B64
                    raw = base64.b64decode(MODEL_B64.strip())
                    import io
                    intra_data = pickle.load(io.BytesIO(raw))
                    logger.info("Intraday ML model loaded from embedded base64")
                except ImportError:
                    logger.warning("Intraday ML model not found (no file or embedded data)")
            if intra_data:
                self._intraday_ml_model = intra_data['model']
                self._intraday_ml_features = intra_data['feature_names']
                self._intraday_ml_threshold = intra_data.get('threshold', 0.5)
                self._intraday_trade_state = {
                    'bars_since_last': 999, 'daily_trades': 0,
                    'consec_wins': 0, 'consec_losses': 0,
                    'last_trade_date': None,
                }
                self._yf_intraday_trade_state = {
                    'bars_since_last': 999, 'daily_trades': 0,
                    'consec_wins': 0, 'consec_losses': 0,
                    'last_trade_date': None,
                }
                logger.info("Intraday ML model ready: %d features, threshold=%.2f",
                            len(self._intraday_ml_features), self._intraday_ml_threshold)
        except Exception as e:
            logger.warning("Intraday ML model load failed: %s — %s", e, traceback.format_exc())
            self._intraday_ml_model = None

    def _init_yf_scanners(self):
        """Create yf-* prefixed scanners with identical configs to IB counterparts."""
        try:
            from v15.trading.surfer_live_scanner import SurferLiveScanner, ScannerConfig

            # c16 generation (yf- prefixed)
            config = ScannerConfig(initial_capital=self.scanner_capital)
            self.scanner_yf = SurferLiveScanner(config, model_tag='yf-c16')
            self.scanner_yf_dw = SurferLiveScanner(config, model_tag='yf-c16-dw')
            ml_config = ScannerConfig(
                initial_capital=self.scanner_capital,
                min_confidence=0.01,
            )
            self.scanner_yf_ml = SurferLiveScanner(ml_config, model_tag='yf-c16-ml')
            self.scanner_yf_intra = SurferLiveScanner(config, model_tag='yf-c16-intra')
            self.scanner_yf_oe = SurferLiveScanner(config, model_tag='yf-c16-oe')

            # c14a generation (yf- prefixed)
            c14a_config = ScannerConfig(
                initial_capital=self.scanner_capital,
                trail_power=8,
                flat_sizing=False,
                am_block_hour=10,
                intraday_start_hour=13,
                intraday_start_minute=0,
            )
            self.scanner_yf_14a = SurferLiveScanner(c14a_config, model_tag='yf-c14a')
            self.scanner_yf_14a_dw = SurferLiveScanner(c14a_config, model_tag='yf-c14a-dw')
            c14a_ml_config = ScannerConfig(
                initial_capital=self.scanner_capital,
                min_confidence=0.01,
                trail_power=8,
                flat_sizing=False,
                am_block_hour=10,
                intraday_start_hour=13,
                intraday_start_minute=0,
            )
            self.scanner_yf_14a_ml = SurferLiveScanner(c14a_ml_config, model_tag='yf-c14a-ml')
            self.scanner_yf_14a_intra = SurferLiveScanner(c14a_config, model_tag='yf-c14a-intra')

            yf_scanners = [
                self.scanner_yf, self.scanner_yf_dw, self.scanner_yf_ml,
                self.scanner_yf_intra, self.scanner_yf_oe,
                self.scanner_yf_14a, self.scanner_yf_14a_dw,
                self.scanner_yf_14a_ml, self.scanner_yf_14a_intra,
            ]
            for scnr in yf_scanners:
                scnr._save_state()
            logger.info("yfinance A/B scanners initialized: 5x yf-c16 + 4x yf-c14a")
        except Exception as e:
            logger.error("yfinance A/B scanner init failed: %s", e)
            self.scanner_yf = None
            self.scanner_yf_dw = None
            self.scanner_yf_ml = None
            self.scanner_yf_intra = None
            self.scanner_yf_oe = None
            self.scanner_yf_14a = None
            self.scanner_yf_14a_dw = None
            self.scanner_yf_14a_ml = None
            self.scanner_yf_14a_intra = None

    def _evaluate_surfer_ml(self, analysis):
        """Evaluate Surfer ML signal (IB path — thin wrapper)."""
        self._evaluate_surfer_ml_with(
            analysis=analysis, current_tsla=self.current_tsla,
            native_tf_data=self.native_tf_data,
            scanner_ml=self.scanner_ml, scanner_14a_ml=self.scanner_14a_ml,
            label='IB',
        )

    def _evaluate_surfer_ml_with(self, *, analysis, current_tsla, native_tf_data,
                                  scanner_ml, scanner_14a_ml, label='',
                                  history_buffer=None, price_override=None):
        """Evaluate Surfer ML signal: physics signal + ML gate.

        ML model must agree with the signal direction to accept the trade.
        Action map: 0=HOLD, 1=BUY, 2=SELL.
        """
        if not scanner_ml or not self._ml_model or not analysis:
            return
        _price = price_override if price_override and price_override > 0 else self.tsla_price
        if _price <= 0:
            return

        sig = analysis.signal
        if sig.action == 'HOLD':
            return

        action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        physics_action_id = 1 if sig.action == 'BUY' else 2
        # Extract ML features
        try:
            import numpy as np
            from v15.core.surfer_backtest import _extract_signal_features

            tsla_df = current_tsla
            if tsla_df is None or len(tsla_df) == 0:
                logger.warning("[%s] Surfer ML: no TSLA data for feature extraction", label)
                return

            bar = len(tsla_df) - 1
            closes = tsla_df['close'].values

            spy_df = None
            vix_df = None
            if native_tf_data:
                spy_data = native_tf_data.get('SPY', {})
                spy_df = spy_data.get('daily')
                vix_data = native_tf_data.get('^VIX', {})
                vix_df = vix_data.get('daily')
            logger.info("[%s] Surfer ML features: spy_df=%s (%d rows), vix_df=%s (%d rows)",
                         label,
                         'YES' if spy_df is not None else 'NO',
                         len(spy_df) if spy_df is not None else 0,
                         'YES' if vix_df is not None else 'NO',
                         len(vix_df) if vix_df is not None else 0)

            closed = scanner_ml.closed_trades
            wins = sum(1 for t in closed[-10:] if t.pnl >= 0) if closed else 0
            losses = sum(1 for t in closed[-10:] if t.pnl < 0) if closed else 0

            feature_vec, feat_dict = _extract_signal_features(
                analysis, tsla_df, bar, closes,
                spy_df=spy_df, vix_df=vix_df,
                feature_names=self._ml_feature_names,
                history_buffer=history_buffer if history_buffer is not None else self._ml_history_buffer,
                eval_interval=3,
                closed_trades=[t.to_dict() for t in closed[-50:]] if closed else [],
                consecutive_wins=wins,
                consecutive_losses=losses,
                daily_pnl=scanner_ml.daily_pnl,
                equity=scanner_ml.equity,
            )

            # Log feature health: count NaN/zero/valid
            n_features = len(feature_vec)
            n_nan = int(np.isnan(feature_vec).sum())
            n_zero = int((feature_vec == 0).sum())
            n_valid = n_features - n_nan
            logger.info("[%s] Surfer ML features: %d total, %d valid, %d NaN, %d zero",
                         label, n_features, n_valid, n_nan, n_zero)
            if n_nan > n_features * 0.3:
                logger.warning("[%s] Surfer ML: >30%% features are NaN (%d/%d) — prediction unreliable",
                               label, n_nan, n_features)
            # Log top features by name
            if feat_dict:
                sample = {k: f'{v:.4f}' for k, v in list(feat_dict.items())[:10]}
                logger.info("[%s] Surfer ML feature sample: %s", label, sample)

            # Run ML prediction
            ml_pred = self._ml_model.predict(feature_vec.reshape(1, -1))
            ml_action = int(ml_pred.get('action', [0])[0]) if 'action' in ml_pred else 0
            ml_lifetime = float(ml_pred.get('lifetime', [0])[0]) if 'lifetime' in ml_pred else 0
            ml_action_probs = ml_pred.get('action_probs', [[0, 0, 0]])[0]

            logger.info("[%s] Surfer ML prediction: signal=%s (id=%d), ML=%s (id=%d), "
                         "probs=[HOLD=%.3f, BUY=%.3f, SELL=%.3f], lifetime=%.0f bars, conf=%.0f%%",
                         label, sig.action, physics_action_id,
                         action_map.get(ml_action, '?'), ml_action,
                         ml_action_probs[0], ml_action_probs[1], ml_action_probs[2],
                         ml_lifetime, sig.confidence * 100)

            # --- ML soft gate (informational only) ---
            if ml_action == 0:
                logger.info("[%s] Surfer ML INFO: ML predicts HOLD (signal was %s) — proceeding anyway", label, sig.action)
            elif ml_action != physics_action_id:
                logger.info("[%s] Surfer ML INFO: ML predicts %s but signal is %s — proceeding anyway",
                             label, action_map.get(ml_action, '?'), sig.action)
            else:
                logger.info("[%s] Surfer ML INFO: ML agrees with %s signal", label, sig.action)

            # --- EL / ER sub-model predictions ---
            el_flagged = False
            trail_width = 1.0
            if self._el_model and feature_vec is not None:
                try:
                    el_pred = self._el_model.predict(feature_vec.reshape(1, -1))
                    el_prob = float(el_pred['loser_prob'][0])
                    el_flagged = el_prob > 0.18
                    logger.info("[%s] EL sub-model: loser_prob=%.3f, flagged=%s", label, el_prob, el_flagged)
                except Exception as e:
                    logger.warning("[%s] EL prediction failed: %s", label, e)
            if self._er_model and feature_vec is not None:
                try:
                    er_pred = self._er_model.predict(feature_vec.reshape(1, -1))
                    er_prob = float(er_pred.get('run_prob', [0.5])[0])
                    if er_prob > 0.70:
                        trail_width = 2.0
                    elif er_prob > 0.50:
                        trail_width = 1.5
                    elif er_prob < 0.30:
                        trail_width = 0.7
                    logger.info("[%s] ER sub-model: run_prob=%.3f, trail_width=%.1f", label, er_prob, trail_width)
                except Exception as e:
                    logger.warning("[%s] ER prediction failed: %s", label, e)

        except Exception as e:
            logger.warning("[%s] Surfer ML feature extraction failed: %s — proceeding anyway", label, e)
            el_flagged = False
            trail_width = 1.0

        # Evaluate through scanner_ml (ML informs but doesn't gate)
        try:
            entry_alert = scanner_ml.evaluate_signal(
                analysis, _price, signal_source='surfer_ml',
                el_flagged=el_flagged, trail_width_mult=trail_width)
            if entry_alert and entry_alert.alert_type == 'ENTRY':
                logger.info("[%s] Surfer ML trade OPENED: %s @ $%.2f",
                             label, sig.action, _price)
                self.positions_version += 1
                self.trades_version += 1
                self.load_model_data()
            elif entry_alert:
                logger.info("[%s] Surfer ML evaluate_signal returned: %s", label, entry_alert.alert_type)
            else:
                logger.info("[%s] Surfer ML evaluate_signal returned None (scanner rejected)", label)
        except Exception as e:
            logger.warning("[%s] Surfer ML eval failed: %s", label, e)

        # c14a-ml: same signal, different config
        if scanner_14a_ml:
            try:
                a14ml = scanner_14a_ml.evaluate_signal(
                    analysis, _price, signal_source='surfer_ml')
                if a14ml and a14ml.alert_type == 'ENTRY':
                    self.positions_version += 1
                    self.trades_version += 1
                    self.load_model_data()
            except Exception as e:
                logger.warning("[%s] c14a ML eval failed: %s", label, e)

    def _evaluate_intraday(self, analysis):
        """Evaluate intraday signal (IB path — thin wrapper)."""
        self._evaluate_intraday_with(
            analysis=analysis, current_tsla=self.current_tsla,
            scanner_intra=self.scanner_intra,
            scanner_14a_intra=self.scanner_14a_intra,
            label='IB',
        )

    def _evaluate_intraday_with(self, *, analysis, current_tsla,
                                 scanner_intra, scanner_14a_intra, label='',
                                 intraday_trade_state=None, price_override=None):
        """Extract 5-min features from analysis and evaluate intraday signal."""
        if analysis is None or not analysis.tf_states:
            return
        if current_tsla is None or len(current_tsla) == 0:
            return
        _price = price_override if price_override and price_override > 0 else self.tsla_price
        if scanner_intra is None or _price <= 0:
            return

        tf_states = analysis.tf_states
        state_5m = tf_states.get('5min')
        state_1h = tf_states.get('1h')
        state_4h = tf_states.get('4h')
        state_daily = tf_states.get('daily')
        state_15m = tf_states.get('15min')
        state_30m = tf_states.get('30min')

        if not state_5m or not state_5m.valid:
            return

        cp5 = state_5m.position_pct
        h1_cp = state_1h.position_pct if state_1h and state_1h.valid else float('nan')
        h4_cp = state_4h.position_pct if state_4h and state_4h.valid else float('nan')
        daily_cp = state_daily.position_pct if state_daily and state_daily.valid else float('nan')

        # Compute VWAP distance and other 5-min bar features
        import numpy as np
        vwap_dist = float('nan')
        spread_pct = float('nan')
        vol_ratio = float('nan')
        vwap_slope = float('nan')
        gap_pct = float('nan')
        range_today_pct = float('nan')
        volume_today_ratio = float('nan')
        atr_5m_pct = float('nan')
        return_5bar = float('nan')
        return_20bar = float('nan')
        rsi_5m_val = float('nan')
        bvc_5m_val = float('nan')
        rsi_slope_val = float('nan')
        try:
            close_arr = current_tsla['close'].values
            high_arr = current_tsla['high'].values
            low_arr = current_tsla['low'].values
            vol_arr = current_tsla['volume'].values
            n = len(close_arr)
            tp = (high_arr + low_arr + close_arr) / 3.0
            dates = current_tsla.index.date
            today = dates[-1]
            today_mask = dates == today

            # VWAP distance
            if today_mask.sum() > 0:
                today_tp = tp[today_mask]
                today_vol = vol_arr[today_mask]
                cum_tv = (today_tp * today_vol).cumsum()
                cum_v = today_vol.cumsum()
                valid_v = cum_v > 0
                if valid_v.any():
                    vwap_val = cum_tv[valid_v][-1] / cum_v[valid_v][-1]
                    vwap_dist = (close_arr[-1] - vwap_val) / vwap_val * 100.0
                    # VWAP slope (last 5 bars)
                    if valid_v.sum() >= 5:
                        vwap_5 = cum_tv[valid_v][-5:] / cum_v[valid_v][-5:]
                        vwap_slope = (vwap_5[-1] - vwap_5[0]) / max(vwap_5[0], 0.01) * 100.0

            # Spread
            if n > 0 and close_arr[-1] > 0:
                spread_pct = (high_arr[-1] - low_arr[-1]) / close_arr[-1] * 100.0

            # Volume ratio
            today_count = int(today_mask.sum())
            if today_count > 0 and n >= 100:
                today_vol_total = float(np.sum(vol_arr[today_mask]))
                avg_daily_vol = float(np.mean(vol_arr[max(0, n - 100):n])) * 78
                vol_ratio = today_vol_total / max(1.0, avg_daily_vol)

            # Gap %
            if n >= 2 and close_arr[-2] > 0:
                yesterday_mask = dates == dates[-1 - today_count] if today_count < n else np.zeros_like(dates, dtype=bool)
                if yesterday_mask.any():
                    prev_close = close_arr[yesterday_mask][-1]
                    today_open = current_tsla['open'].values[today_mask][0] if today_mask.any() else close_arr[-1]
                    gap_pct = (today_open - prev_close) / prev_close * 100.0

            # ATR 5m %
            if n >= 21 and close_arr[-1] > 0:
                hl = high_arr[-20:] - low_arr[-20:]
                hc = np.abs(high_arr[-20:] - close_arr[-21:-1])
                lc = np.abs(low_arr[-20:] - close_arr[-21:-1])
                tr = np.maximum(np.maximum(hl, hc), lc)
                atr_5m_pct = float(np.mean(tr)) / close_arr[-1] * 100.0

            # Range today %
            if today_mask.any() and close_arr[-1] > 0:
                range_today_pct = (float(np.max(high_arr[today_mask])) - float(np.min(low_arr[today_mask]))) / close_arr[-1] * 100.0

            # Volume today ratio
            if today_count > 0 and n >= 100:
                volume_today_ratio = vol_ratio  # same calculation

            # Returns
            if n >= 6 and close_arr[-6] > 0:
                return_5bar = (close_arr[-1] - close_arr[-6]) / close_arr[-6] * 100.0
            if n >= 21 and close_arr[-21] > 0:
                return_20bar = (close_arr[-1] - close_arr[-21]) / close_arr[-21] * 100.0

            # RSI(14) from 5-min close -- Wilder smoothing (matches training)
            if n >= 16:
                deltas = np.diff(close_arr)
                gains = np.maximum(deltas, 0)
                losses = np.maximum(-deltas, 0)
                p = 14
                ag = gains[:p].mean()
                al = losses[:p].mean()
                for ki in range(p, len(deltas)):
                    ag = (ag * (p - 1) + gains[ki]) / p
                    al = (al * (p - 1) + losses[ki]) / p
                rsi_5m_val = 100.0 if al < 1e-10 else 100.0 - 100.0 / (1.0 + ag / al)

                # RSI slope: linear regression over last 5 RSI values
                if n >= 20:
                    rsi_arr = np.full(n, np.nan)
                    ag2 = gains[:p].mean()
                    al2 = losses[:p].mean()
                    for ki in range(p, len(deltas)):
                        ag2 = (ag2 * (p - 1) + gains[ki]) / p
                        al2 = (al2 * (p - 1) + losses[ki]) / p
                        rsi_arr[ki + 1] = 100.0 if al2 < 1e-10 else 100.0 - 100.0 / (1.0 + ag2 / al2)
                    seg = rsi_arr[-5:]
                    if not np.any(np.isnan(seg)):
                        x = np.arange(5, dtype=np.float64)
                        mx, my = x.mean(), seg.mean()
                        rsi_slope_val = float(np.sum((x - mx) * (seg - my)) / np.sum((x - mx) ** 2))

            # BVC(20) -- bullish volume confirmed (matches training)
            if n >= 21:
                from scipy.stats import norm as _norm
                dp = np.diff(close_arr)
                dp = np.concatenate([[0], dp])
                lb = 20
                s = pd.Series(dp).rolling(lb, min_periods=lb).std().values
                z = np.zeros(n)
                vl = s > 1e-10
                z[vl] = dp[vl] / s[vl]
                bp = _norm.cdf(z)
                nf = vol_arr * (2 * bp - 1)
                nc = pd.Series(nf).rolling(lb, min_periods=lb).sum().values
                tv = pd.Series(vol_arr.astype(float)).rolling(lb, min_periods=lb).sum().values
                valid_tv = tv > 0
                if valid_tv[-1]:
                    bvc_5m_val = float(nc[-1] / tv[-1])
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
            # Run ML filter BEFORE opening the trade (fix #2: filter was no-op)
            # Use a default confidence of 0.5 for pre-filter (scanner may adjust)
            ml_pass = self._intraday_ml_filter(
                cp5=cp5, vwap_dist=vwap_dist, vol_ratio=vol_ratio,
                vwap_slope=vwap_slope, spread_pct=spread_pct, gap_pct=gap_pct,
                daily_cp=daily_cp, h1_cp=h1_cp, h4_cp=h4_cp,
                daily_slope=daily_slope, h1_slope=h1_slope, h4_slope=h4_slope,
                confidence=0.5,
                atr_5m_pct=atr_5m_pct, range_today_pct=range_today_pct,
                volume_today_ratio=volume_today_ratio,
                return_5bar=return_5bar, return_20bar=return_20bar,
                rsi_5m=rsi_5m_val, bvc_5m=bvc_5m_val, rsi_slope=rsi_slope_val,
                cp_15m=state_15m.position_pct if state_15m and state_15m.valid else float('nan'),
                cp_30m=state_30m.position_pct if state_30m and state_30m.valid else float('nan'),
                trade_state=intraday_trade_state,
            )
            if not ml_pass:
                logger.info("[%s] Intraday signal REJECTED by ML filter (pre-entry)", label)
            else:
                alert = scanner_intra.evaluate_intraday_signal(
                    current_price=_price,
                    cp5=cp5, vwap_dist=vwap_dist,
                    daily_cp=daily_cp, h1_cp=h1_cp, h4_cp=h4_cp,
                    daily_slope=daily_slope, h1_slope=h1_slope, h4_slope=h4_slope,
                    vol_ratio=vol_ratio, vwap_slope=vwap_slope,
                    spread_pct=spread_pct, gap_pct=gap_pct,
                )
                if alert and alert.alert_type == 'ENTRY':
                    self.positions_version += 1
                    self.trades_version += 1
                    self.load_model_data()
        except Exception as e:
            logger.warning("[%s] Intraday eval failed: %s", label, e)

        # c14a-intra: same signal, different config (PM-only window, confidence sizing)
        if scanner_14a_intra:
            try:
                a14i = scanner_14a_intra.evaluate_intraday_signal(
                    current_price=_price,
                    cp5=cp5, vwap_dist=vwap_dist,
                    daily_cp=daily_cp, h1_cp=h1_cp, h4_cp=h4_cp,
                    daily_slope=daily_slope, h1_slope=h1_slope, h4_slope=h4_slope,
                    vol_ratio=vol_ratio, vwap_slope=vwap_slope,
                    spread_pct=spread_pct, gap_pct=gap_pct,
                )
                if a14i and a14i.alert_type == 'ENTRY':
                    self.positions_version += 1
                    self.trades_version += 1
                    self.load_model_data()
            except Exception as e:
                logger.warning("c14a intraday eval failed: %s", e)

    def _run_yf_analysis_core(self):
        """Compute analysis from yfinance data. Returns (tsla_df, analysis, dw_analysis) or None."""
        from v15.core.channel_surfer import prepare_multi_tf_analysis

        # Refresh yfinance 5-min bars
        tsla_df = None
        try:
            from v15.live_data import fetch_live_data
            tsla_df, spy_df, vix_df = fetch_live_data(period='5d', interval='5m')
        except Exception as e:
            logger.warning("yf A/B: failed to refresh 5-min data: %s", e)

        effective_tsla = tsla_df if tsla_df is not None and len(tsla_df) > 0 else self._yf_current_tsla

        if not self._yf_native_tf_data or effective_tsla is None:
            return None

        analysis = prepare_multi_tf_analysis(
            native_data=self._yf_native_tf_data,
            live_5min_tsla=effective_tsla,
            target_tfs=['5min', '1h', '4h', 'daily', 'weekly'],
        )

        dw_analysis = None
        if self.scanner_yf_dw:
            dw_analysis = prepare_multi_tf_analysis(
                native_data=self._yf_native_tf_data,
                live_5min_tsla=effective_tsla,
                target_tfs=['daily', 'weekly'],
            )

        return tsla_df, analysis, dw_analysis

    def _apply_yf_analysis_results(self, tsla_df, analysis, dw_analysis):
        """Apply yfinance analysis results — route to yf-* scanners."""
        self._yf_analysis_running = False

        if tsla_df is not None and len(tsla_df) > 0:
            self._yf_current_tsla = tsla_df

        logger.info("yf A/B analysis complete: %s %s (%.0f%%)",
                     analysis.signal.action,
                     analysis.signal.primary_tf,
                     analysis.signal.confidence * 100)

        # Use IB live price for entry decisions; fall back to yfinance last close if IB stale
        price_age = time.time() - self._price_updated_at if self._price_updated_at > 0 else 999
        yf_price = 0.0
        if tsla_df is not None and len(tsla_df) > 0:
            yf_price = float(tsla_df['close'].iloc[-1])
        if price_age > 5 and yf_price > 0:
            entry_price = yf_price
            logger.info("[yf] IB price stale (%.0fs) — using yfinance price $%.2f", price_age, yf_price)
        elif self.tsla_price > 0:
            entry_price = self.tsla_price
        else:
            entry_price = yf_price  # last resort
        if self.scanner_yf and entry_price > 0:
            self._apply_analysis_results_with(
                tsla_df=self._yf_current_tsla,
                analysis=analysis,
                dw_analysis=dw_analysis,
                native_tf_data=self._yf_native_tf_data,
                scanner_cs=self.scanner_yf,
                scanner_dw=self.scanner_yf_dw,
                scanner_ml=self.scanner_yf_ml,
                scanner_intra=self.scanner_yf_intra,
                scanner_14a=self.scanner_yf_14a,
                scanner_14a_dw=self.scanner_yf_14a_dw,
                scanner_14a_ml=self.scanner_yf_14a_ml,
                scanner_14a_intra=self.scanner_yf_14a_intra,
                label='yf',
                ml_history_buffer=self._yf_ml_history_buffer,
                intraday_trade_state=self._yf_intraday_trade_state,
                price_override=entry_price,
            )
            # OE signals for yf path
            if self.scanner_yf_oe and self._yf_native_tf_data and entry_price > 0:
                try:
                    from v15.core.oe_signals5 import check_oe_signal
                    if check_oe_signal(self._yf_native_tf_data):
                        from types import SimpleNamespace
                        mock_signal = SimpleNamespace(
                            action='BUY', confidence=0.7, primary_tf='daily',
                            signal_type='bounce', reason='OE Signals_5',
                            suggested_stop_pct=0.03, suggested_tp_pct=0.50,
                            ou_half_life=5.0,
                        )
                        mock_analysis = SimpleNamespace(signal=mock_signal, atr=None)
                        alert = self.scanner_yf_oe.evaluate_signal(
                            mock_analysis, entry_price, signal_source='oe_signals5')
                        if alert and alert.alert_type == 'ENTRY':
                            self.positions_version += 1
                            self.trades_version += 1
                            self.load_model_data()
                except Exception as e:
                    logger.warning("yf A/B OE eval failed: %s", e)

    def _run_yf_analysis_bg(self):
        """Background-safe yfinance analysis: compute + apply."""
        if not self._yf_native_tf_data:
            return
        if self._yf_analysis_running:
            return
        self._yf_analysis_running = True
        try:
            results = self._run_yf_analysis_core()
            if results:
                try:
                    pn.state.execute(lambda r=results: self._apply_yf_analysis_results(*r))
                except Exception:
                    self._apply_yf_analysis_results(*results)
        except Exception as e:
            logger.error("yf A/B analysis failed: %s\n%s", e, traceback.format_exc())
        finally:
            self._yf_analysis_running = False

    def _evaluate_oe_signals5(self):
        """Evaluate OE Signals_5 on latest daily bars."""
        if not self.scanner_oe or not self.native_tf_data or self.tsla_price <= 0:
            return
        try:
            from v15.core.oe_signals5 import check_oe_signal
            if check_oe_signal(self.native_tf_data):
                from types import SimpleNamespace
                mock_signal = SimpleNamespace(
                    action='BUY', confidence=0.7, primary_tf='daily',
                    signal_type='bounce', reason='OE Signals_5',
                    suggested_stop_pct=0.03, suggested_tp_pct=0.50,
                    ou_half_life=5.0,
                )
                mock_analysis = SimpleNamespace(signal=mock_signal, atr=None)
                alert = self.scanner_oe.evaluate_signal(
                    mock_analysis, self.tsla_price, signal_source='oe_signals5')
                if alert and alert.alert_type == 'ENTRY':
                    self.positions_version += 1
                    self.trades_version += 1
                    self.load_model_data()
        except Exception as e:
            logger.warning("OE Signals_5 eval failed: %s", e)

    def _intraday_ml_filter(self, trade_state=None, **kwargs) -> bool:
        """Run intraday ML model to filter signals. Returns True to accept, False to reject."""
        if self._intraday_ml_model is None:
            return True  # No model = accept all

        import numpy as np
        from datetime import datetime
        import pytz

        try:
            now = datetime.now(pytz.timezone('US/Eastern'))
            market_close = now.replace(hour=16, minute=0, second=0)
            minutes_to_close = max(0, (market_close - now).total_seconds() / 60.0)

            # Use passed trade_state (IB vs yf), fallback to default
            ts = trade_state if trade_state is not None else self._intraday_trade_state
            today_str = now.strftime('%Y-%m-%d')
            if ts['last_trade_date'] != today_str:
                ts['daily_trades'] = 0
                ts['last_trade_date'] = today_str
            ts['bars_since_last'] += 1

            # Build feature vector matching FEATURE_NAMES order from training
            feat = np.full(len(self._intraday_ml_features), np.nan)
            feat_map = {name: i for i, name in enumerate(self._intraday_ml_features)}

            def _set(name, val):
                if name in feat_map:
                    feat[feat_map[name]] = val

            # 5-min features
            _set('cp_5m', kwargs.get('cp5', float('nan')))
            _set('rsi_5m', kwargs.get('rsi_5m', float('nan')))
            _set('bvc_5m', kwargs.get('bvc_5m', float('nan')))
            _set('vwap_dist', kwargs.get('vwap_dist', float('nan')))
            _set('vol_ratio', kwargs.get('vol_ratio', float('nan')))
            _set('vwap_slope', kwargs.get('vwap_slope', float('nan')))
            _set('spread_pct', kwargs.get('spread_pct', float('nan')))
            _set('rsi_slope', kwargs.get('rsi_slope', float('nan')))
            _set('gap_pct', kwargs.get('gap_pct', float('nan')))

            # Higher TF positions
            _set('cp_15m', kwargs.get('cp_15m', float('nan')))
            _set('cp_30m', kwargs.get('cp_30m', float('nan')))
            _set('cp_1h', kwargs.get('h1_cp', float('nan')))
            _set('cp_4h', kwargs.get('h4_cp', float('nan')))
            _set('cp_daily', kwargs.get('daily_cp', float('nan')))

            # Slopes
            _set('slope_1h', kwargs.get('h1_slope', float('nan')))
            _set('slope_4h', kwargs.get('h4_slope', float('nan')))
            _set('slope_daily', kwargs.get('daily_slope', float('nan')))

            # Cross-TF divergences
            cp5v = kwargs.get('cp5', float('nan'))
            dcp = kwargs.get('daily_cp', float('nan'))
            h1v = kwargs.get('h1_cp', float('nan'))
            h4v = kwargs.get('h4_cp', float('nan'))
            if not np.isnan(dcp) and not np.isnan(cp5v):
                _set('div_daily_5m', dcp - cp5v)
            if not np.isnan(h1v) and not np.isnan(cp5v):
                _set('div_1h_5m', h1v - cp5v)
            if not np.isnan(h4v) and not np.isnan(cp5v):
                _set('div_4h_5m', h4v - cp5v)
            vals = [v for v in [dcp, h4v, h1v] if not np.isnan(v)]
            if vals and not np.isnan(cp5v):
                weights = [0.35, 0.35, 0.30][:len(vals)]
                weighted_avg = sum(v * w for v, w in zip(vals, weights)) / sum(weights)
                _set('div_weighted', weighted_avg - cp5v)

            # Time features
            _set('hour', now.hour)
            _set('minute', now.minute)
            _set('minutes_to_close', minutes_to_close)
            _set('day_of_week', now.weekday())

            # Volatility / range
            _set('atr_5m_pct', kwargs.get('atr_5m_pct', float('nan')))
            _set('range_today_pct', kwargs.get('range_today_pct', float('nan')))
            _set('volume_today_ratio', kwargs.get('volume_today_ratio', float('nan')))

            # Signal quality
            _set('confidence', kwargs.get('confidence', float('nan')))

            # Momentum
            _set('return_5bar', kwargs.get('return_5bar', float('nan')))
            _set('return_20bar', kwargs.get('return_20bar', float('nan')))

            # Cross-TF alignment (derived)
            all_cps = [v for v in [cp5v, kwargs.get('cp_15m', float('nan')),
                        kwargs.get('cp_30m', float('nan')), h1v, h4v, dcp]
                       if not np.isnan(v)]
            _set('bullish_tf_count', sum(1 for v in all_cps if v > 0.5))
            _set('cp_dispersion', float(np.std(all_cps)) if len(all_cps) >= 2 else float('nan'))
            slope_vals = [v for v in [kwargs.get('h1_slope', float('nan')),
                          kwargs.get('h4_slope', float('nan')),
                          kwargs.get('daily_slope', float('nan'))]
                         if not np.isnan(v)]
            _set('slope_agreement', sum(1 for v in slope_vals if v > 0))

            # Trade state
            _set('bars_since_last_trade', ts['bars_since_last'])
            _set('daily_trade_count', ts['daily_trades'])
            _set('consecutive_wins', ts['consec_wins'])
            _set('consecutive_losses', ts['consec_losses'])

            # Run prediction
            prob = self._intraday_ml_model.predict(feat.reshape(1, -1))[0]
            thresh = getattr(self, '_intraday_ml_threshold', 0.5)
            accept = prob >= thresh
            logger.info("Intraday ML: prob=%.3f, thresh=%.2f, accept=%s", prob, thresh, accept)

            # Update trade state only on acceptance
            if accept:
                ts['daily_trades'] += 1
                ts['bars_since_last'] = 0

            return accept

        except Exception as e:
            logger.warning("Intraday ML filter error: %s", e)
            return True  # On error, accept the signal

    def send_notification(self, msg: str, title: str = '') -> str:
        """Send a push notification via Telegram. Returns status string."""
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
        chat_id = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
        if not bot_token or not chat_id:
            return 'NO_CHANNEL'

        try:
            import requests as _req
            text = f"*{title or 'c14a Alert'}*\n{msg}" if title else msg
            resp = _req.post(
                f'https://api.telegram.org/bot{bot_token}/sendMessage',
                json={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'},
                timeout=10,
            )
            logger.info("Telegram: HTTP %d", resp.status_code)
            if resp.status_code == 200:
                return 'OK'
            logger.warning("Telegram failed: %s", resp.text[:200])
            return f'HTTP {resp.status_code}'
        except Exception as e:
            logger.warning("Telegram error: %s", e)
            return f'ERROR: {e}'

    def send_test_notification(self):
        """Send a test push notification."""
        from datetime import datetime
        import pytz

        now = datetime.now(pytz.timezone('US/Eastern'))
        msg = (
            f"Dashboard: c14a Trading Dashboard\n"
            f"TSLA: ${self.tsla_price:.2f}\n"
            f"Time: {now.strftime('%Y-%m-%d %H:%M:%S ET')}\n"
            f"Scanner: {'OK' if self.scanner else 'NONE'}\n"
            f"Notifications are working!"
        )
        return self.send_notification(msg, title='Test Notification')

    def load_model_data(self):
        """Load multi-model state from per-model local JSON files."""
        import json
        from v15.trading.surfer_live_scanner import _STATE_DIR

        combined = {}
        for tag in ('c16', 'c16-dw', 'c16-ml', 'c16-intra', 'c16-oe',
                    'c14a', 'c14a-dw', 'c14a-ml', 'c14a-intra',
                    'yf-c16', 'yf-c16-dw', 'yf-c16-ml', 'yf-c16-intra', 'yf-c16-oe',
                    'yf-c14a', 'yf-c14a-dw', 'yf-c14a-ml', 'yf-c14a-intra'):
            fpath = _STATE_DIR / f"surfer_state_{tag}.json"
            if fpath.exists():
                try:
                    combined[tag] = json.loads(fpath.read_text())
                except Exception as e:
                    logger.warning("Failed to load %s: %s", fpath, e)
        if combined:
            combined['_last_updated'] = datetime.now().isoformat()
            self.model_data = combined
            self.model_data_version += 1
            logger.info("Model data loaded: %d models (local files)",
                        len([k for k in combined if not k.startswith('_')]))

    def flush_scanner_state(self):
        """One-time: write each scanner's in-memory state to its per-model local file."""
        count = 0
        for scnr in self._all_scanners:
            scnr._save_state()
            count += 1
            logger.info("Flushed state for %s: %d trades, equity=$%.2f",
                        scnr.model_tag, len(scnr.closed_trades), scnr.equity)
        self.load_model_data()
        return count


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
        _src_display = {'CS-5TF': 'CS-5TF', 'CS-DW': 'CS-DW',
                        'surfer_ml': 'Surfer ML', 'intraday': 'Intraday'}
        display_src = _src_display.get(ea.signal_source, ea.signal_source)
        source_tag = (f' <span style="color:#64b5f6;font-size:12px;'
                      f'background:#1a237e;padding:2px 6px;border-radius:4px">'
                      f'{display_src}</span>')
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
