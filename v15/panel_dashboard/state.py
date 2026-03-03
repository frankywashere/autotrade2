"""Dashboard state — central param.Parameterized class for reactive updates."""

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
    scanner_ml = param.Parameter(None)        # SurferLiveScanner (Surfer ML)
    positions_version = param.Integer(0)      # Bump to trigger position card re-render (price-sensitive)
    trades_version = param.Integer(0)        # Bump only on actual trade open/close (not price ticks)
    exit_alert_html = param.String('')        # Latest exit alert HTML

    # Config (sidebar widgets bind to these)
    scanner_capital = param.Number(100_000, bounds=(10_000, 1_000_000))
    kill_switch = param.Boolean(False)

    # Model comparisons (1-hour cache)
    model_data = param.Dict({})
    model_data_version = param.Integer(0)

    # Internal
    _prev_price = param.Number(0.0, precedence=-1)
    _ml_model = param.Parameter(None, precedence=-1)
    _ml_feature_names = param.Parameter(None, precedence=-1)
    _ml_history_buffer = param.Parameter(None, precedence=-1)
    _intraday_ml_model = param.Parameter(None, precedence=-1)
    _intraday_ml_features = param.Parameter(None, precedence=-1)
    _intraday_trade_state = param.Parameter(None, precedence=-1)
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
        self._analysis_running = False

        # Run initial analysis (synchronous at startup — no UI to block yet)
        try:
            results = self._run_analysis_core()
            if results:
                self._apply_analysis_results(*results)
        except Exception as e:
            logger.error("Initial analysis failed: %s\n%s", e, traceback.format_exc())

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
                if spy_tick and spy_tick.price > 0 and spy_tick.price != self.spy_price:
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

        # Check exits if positions are open (both scanners)
        html_parts = []
        all_exit_alerts = []
        for scnr in [self.scanner, self.scanner_dw, self.scanner_ml]:
            if scnr and scnr.positions and price > 0:
                try:
                    exit_alerts = scnr.check_exits(price, price, price)
                    if exit_alerts:
                        for ea in exit_alerts:
                            html_parts.append(_exit_alert_html(ea))
                            all_exit_alerts.append(ea)
                except Exception as e:
                    logger.warning("Exit check failed: %s", e)
        if all_exit_alerts:
            self.exit_alert_html = '\n'.join(html_parts)
            self.positions_version += 1  # Re-render banner to clear exited positions
            self.trades_version += 1     # Update trade history
            # Update intraday ML trade state from intraday exits
            if self._intraday_trade_state:
                for ea in all_exit_alerts:
                    if getattr(ea, 'signal_source', '') == 'intraday':
                        ts = self._intraday_trade_state
                        if getattr(ea, 'pnl', 0) > 0:
                            ts['consec_wins'] += 1
                            ts['consec_losses'] = 0
                        else:
                            ts['consec_losses'] += 1
                            ts['consec_wins'] = 0
        # Bump version for live P&L updates only when price actually changed
        elif not all_exit_alerts and (price_changed if price > 0 else False):
            has_positions = ((self.scanner and self.scanner.positions)
                             or (self.scanner_dw and self.scanner_dw.positions)
                             or (self.scanner_ml and self.scanner_ml.positions))
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

        # Refresh 5-min bars
        tsla_df = None
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

        # Evaluate signals if scanner is ready
        if self.scanner and self.tsla_price > 0:
            # --- CS-5TF: Original 5-timeframe signal ---
            sig = analysis.signal
            if sig.action != 'HOLD':
                entry_alert = self.scanner.evaluate_signal(
                    analysis, self.tsla_price, signal_source='CS-5TF')
                if entry_alert and entry_alert.alert_type == 'ENTRY':
                    self.positions_version += 1
                    self.trades_version += 1

            # --- CS-DW: Daily+Weekly only signal (separate scanner/model) ---
            if self.scanner_dw and dw_analysis and self.tsla_price > 0:
                try:
                    dw_sig = dw_analysis.signal
                    if dw_sig.action != 'HOLD':
                        dw_alert = self.scanner_dw.evaluate_signal(
                            dw_analysis, self.tsla_price, signal_source='CS-DW')
                        if dw_alert and dw_alert.alert_type == 'ENTRY':
                            self.positions_version += 1
                            self.trades_version += 1
                    logger.info("DW analysis: %s %s (%.0f%%)",
                                dw_sig.action, dw_sig.primary_tf,
                                dw_sig.confidence * 100)
                except Exception as e:
                    logger.warning("DW analysis failed: %s", e)

            # --- Surfer ML: physics signal + ML overlay ---
            self._evaluate_surfer_ml(analysis)

            # Evaluate intraday signal
            self._evaluate_intraday(analysis)

    def _init_scanner(self):
        """Create SurferLiveScanners with Gist credentials from env vars."""
        try:
            from v15.trading.surfer_live_scanner import SurferLiveScanner, ScannerConfig
            config = ScannerConfig(initial_capital=self.scanner_capital)
            gist_id = os.environ.get('GIST_ID', '')
            github_token = os.environ.get('GITHUB_TOKEN', '')
            logger.info("Scanner init: gist_id=%s, github_token=%s",
                         gist_id[:8] + '...' if gist_id else 'MISSING',
                         github_token[:8] + '...' if github_token else 'MISSING')
            self.scanner = SurferLiveScanner(
                config, gist_id=gist_id, github_token=github_token,
                model_tag='c14',
            )
            self.scanner_dw = SurferLiveScanner(
                config, gist_id=gist_id, github_token=github_token,
                model_tag='c14-dw',
            )
            # Surfer ML scanner: low confidence gate (matching surfer_backtest)
            ml_config = ScannerConfig(
                initial_capital=self.scanner_capital,
                min_confidence=0.01,  # Very low gate — ML uses all signals
            )
            self.scanner_ml = SurferLiveScanner(
                ml_config, gist_id=gist_id, github_token=github_token,
                model_tag='c14-ml',
            )
            logger.info("Scanners initialized (c14 + c14-dw + c14-ml, capital=$%,.0f)",
                         self.scanner_capital)
        except Exception as e:
            logger.error("Scanner init failed: %s\n%s", e, traceback.format_exc())
            self.scanner = None
            self.scanner_dw = None
            self.scanner_ml = None

        # Load ML model for Surfer ML path
        try:
            from v15.core.surfer_ml import GBTModel, get_feature_names
            from pathlib import Path
            model_path = Path('surfer_models/gbt_model.pkl')
            if not model_path.exists():
                model_path = Path(__file__).parent.parent.parent / 'surfer_models' / 'gbt_model.pkl'
            if model_path.exists():
                self._ml_model = GBTModel.load(str(model_path))
                self._ml_feature_names = get_feature_names()
                self._ml_history_buffer = []
                import numpy as np
                test_pred = self._ml_model.predict(
                    np.zeros((1, len(self._ml_feature_names)), dtype=np.float32))
                logger.info("ML model loaded: %d features, keys=%s",
                            len(self._ml_feature_names), list(test_pred.keys()))
            else:
                logger.warning("ML model not found at %s", model_path)
        except Exception as e:
            logger.warning("ML model load failed: %s", e)
            self._ml_model = None

        # Load intraday ML model (LightGBM filter for intraday signals)
        try:
            import pickle
            from pathlib import Path
            intra_data = None
            intra_path = Path('surfer_models/intraday_ml_model.pkl')
            if not intra_path.exists():
                intra_path = Path(__file__).parent.parent.parent / 'surfer_models' / 'intraday_ml_model.pkl'
            if intra_path.exists():
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
                logger.info("Intraday ML model ready: %d features, threshold=%.2f",
                            len(self._intraday_ml_features), self._intraday_ml_threshold)
        except Exception as e:
            logger.warning("Intraday ML model load failed: %s", e)
            self._intraday_ml_model = None

    def _evaluate_surfer_ml(self, analysis):
        """Evaluate Surfer ML signal: physics signal + ML feature extraction."""
        if not self.scanner_ml or not self._ml_model or not analysis:
            return
        if self.tsla_price <= 0:
            return

        sig = analysis.signal
        if sig.action == 'HOLD':
            return

        # Extract ML features for context
        try:
            import numpy as np
            from v15.core.surfer_backtest import _extract_signal_features

            # Build minimal inputs for feature extraction
            tsla_df = self.current_tsla
            if tsla_df is None or len(tsla_df) == 0:
                return

            bar = len(tsla_df) - 1
            closes = tsla_df['close'].values

            # SPY/VIX: use native TF data if available
            spy_df = None
            vix_df = None
            if self.native_tf_data:
                spy_data = self.native_tf_data.get('SPY', {})
                spy_df = spy_data.get('daily')
                vix_data = self.native_tf_data.get('^VIX', {})
                vix_df = vix_data.get('daily')

            # Get trade state from scanner
            closed = self.scanner_ml.closed_trades
            wins = sum(1 for t in closed[-10:] if t.pnl >= 0) if closed else 0
            losses = sum(1 for t in closed[-10:] if t.pnl < 0) if closed else 0

            feature_vec, _ = _extract_signal_features(
                analysis, tsla_df, bar, closes,
                spy_df=spy_df, vix_df=vix_df,
                feature_names=self._ml_feature_names,
                history_buffer=self._ml_history_buffer,
                eval_interval=3,
                closed_trades=[t.to_dict() for t in closed[-50:]] if closed else [],
                consecutive_wins=wins,
                consecutive_losses=losses,
                daily_pnl=self.scanner_ml.daily_pnl,
                equity=self.scanner_ml.equity,
            )

            # Run ML prediction (informational context)
            ml_pred = self._ml_model.predict(feature_vec.reshape(1, -1))
            ml_action = int(ml_pred.get('action', [0])[0]) if 'action' in ml_pred else 0
            ml_lifetime = float(ml_pred.get('lifetime', [0])[0]) if 'lifetime' in ml_pred else 0

            # Log ML context
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            logger.info("Surfer ML: signal=%s, ML=%s, lifetime=%.0f bars, conf=%.0f%%",
                         sig.action, action_map.get(ml_action, '?'),
                         ml_lifetime, sig.confidence * 100)

        except Exception as e:
            logger.warning("Surfer ML feature extraction failed: %s", e)

        # Evaluate through scanner_ml (uses surfer_ml signal_source path)
        try:
            entry_alert = self.scanner_ml.evaluate_signal(
                analysis, self.tsla_price, signal_source='surfer_ml')
            if entry_alert and entry_alert.alert_type == 'ENTRY':
                self.positions_version += 1
                self.trades_version += 1
        except Exception as e:
            logger.warning("Surfer ML eval failed: %s", e)

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
            close_arr = self.current_tsla['close'].values
            high_arr = self.current_tsla['high'].values
            low_arr = self.current_tsla['low'].values
            vol_arr = self.current_tsla['volume'].values
            n = len(close_arr)
            tp = (high_arr + low_arr + close_arr) / 3.0
            dates = self.current_tsla.index.date
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
                    today_open = self.current_tsla['open'].values[today_mask][0] if today_mask.any() else close_arr[-1]
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
            alert = self.scanner.evaluate_intraday_signal(
                current_price=self.tsla_price,
                cp5=cp5, vwap_dist=vwap_dist,
                daily_cp=daily_cp, h1_cp=h1_cp, h4_cp=h4_cp,
                daily_slope=daily_slope, h1_slope=h1_slope, h4_slope=h4_slope,
                vol_ratio=vol_ratio, vwap_slope=vwap_slope,
                spread_pct=spread_pct, gap_pct=gap_pct,
            )
            if alert and alert.alert_type == 'ENTRY':
                # ML filter: check if intraday ML model rejects this signal
                if self._intraday_ml_filter(
                    cp5=cp5, vwap_dist=vwap_dist, vol_ratio=vol_ratio,
                    vwap_slope=vwap_slope, spread_pct=spread_pct, gap_pct=gap_pct,
                    daily_cp=daily_cp, h1_cp=h1_cp, h4_cp=h4_cp,
                    daily_slope=daily_slope, h1_slope=h1_slope, h4_slope=h4_slope,
                    confidence=alert.confidence,
                    atr_5m_pct=atr_5m_pct, range_today_pct=range_today_pct,
                    volume_today_ratio=volume_today_ratio,
                    return_5bar=return_5bar, return_20bar=return_20bar,
                    rsi_5m=rsi_5m_val, bvc_5m=bvc_5m_val, rsi_slope=rsi_slope_val,
                    cp_15m=state_15m.position_pct if state_15m and state_15m.valid else float('nan'),
                    cp_30m=state_30m.position_pct if state_30m and state_30m.valid else float('nan'),
                ):
                    self.positions_version += 1
                else:
                    logger.info("Intraday signal REJECTED by ML filter")
        except Exception as e:
            logger.warning("Intraday eval failed: %s", e)

    def _intraday_ml_filter(self, **kwargs) -> bool:
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

            # Update trade state (daily reset only, counts updated AFTER accept/reject)
            ts = self._intraday_trade_state
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

    def send_telegram(self, msg: str) -> str:
        """Send a Telegram message directly via bot API. Returns status string."""
        token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
        chat_id = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
        logger.info("Telegram send: token=%s (%d chars), chat_id=%s",
                     'SET' if token else 'MISSING', len(token),
                     'SET' if chat_id else 'MISSING')
        if not token or not chat_id:
            return 'ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set'

        url = f'https://api.telegram.org/bot{token}/sendMessage'
        payload = {'chat_id': chat_id, 'text': msg, 'parse_mode': 'HTML'}

        # Try requests library first (better DNS/proxy handling)
        try:
            import requests as _req
            logger.info("Telegram: using requests library, URL=%s", url[:60])
            resp = _req.post(url, json=payload, timeout=15)
            logger.info("Telegram: HTTP %d, body=%s", resp.status_code, resp.text[:200])
            if resp.status_code == 200:
                return 'OK'
            return f'HTTP {resp.status_code}: {resp.text[:100]}'
        except ImportError:
            logger.info("Telegram: requests not available, trying urllib")
        except Exception as e:
            logger.error("Telegram requests failed: %s", e, exc_info=True)
            # Fall through to urllib

        # Fall back to urllib
        try:
            import json as _json
            import urllib.request
            data = _json.dumps(payload).encode()
            logger.info("Telegram: using urllib, payload=%d bytes", len(data))
            req = urllib.request.Request(
                url, data=data, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = resp.read().decode()
                logger.info("Telegram urllib: HTTP %d, body=%s", resp.status, body[:200])
                if resp.status == 200:
                    return 'OK'
                return f'HTTP {resp.status}'
        except Exception as e:
            logger.error("Telegram urllib failed: %s", e, exc_info=True)
            return f'ERROR: {e}'

    def send_test_telegram(self):
        """Send a test signal to Telegram with DNS diagnostics."""
        from datetime import datetime
        import pytz
        import socket

        # DNS diagnostic
        dns_info = ''
        try:
            ips = socket.getaddrinfo('api.telegram.org', 443, socket.AF_INET)
            ip_list = [addr[4][0] for addr in ips]
            dns_info = f'DNS OK: {ip_list[:3]}'
            logger.info("Telegram DNS resolve: %s", ip_list)
        except Exception as e:
            dns_info = f'DNS FAILED: {e}'
            logger.error("Telegram DNS resolve failed: %s", e)

        now = datetime.now(pytz.timezone('US/Eastern'))
        msg = (
            f"🧪 <b>TEST SIGNAL</b>\n"
            f"Dashboard: c14 Trading Dashboard\n"
            f"TSLA: ${self.tsla_price:.2f}\n"
            f"Time: {now.strftime('%Y-%m-%d %H:%M:%S ET')}\n"
            f"Scanner: {'OK' if self.scanner else 'NONE'}\n"
            f"Telegram direct sending is working!"
        )
        result = self.send_telegram(msg)
        if result != 'OK':
            return f'{result} | {dns_info}'
        return result

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
