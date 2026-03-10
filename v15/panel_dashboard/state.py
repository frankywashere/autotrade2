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
    data_source = param.String('NONE')  # 'IB' or 'NONE'

    # Market data (loaded on startup, refreshed every 5 min)
    current_tsla = param.Parameter(None)      # 5-min OHLCV DataFrame
    native_tf_data = param.Dict({})           # {symbol: {tf: DataFrame}}
    analysis = param.Parameter(None)          # ChannelAnalysis object
    last_analysis = param.String('')          # Timestamp string

    # UI version bumps (trigger reactive re-renders)
    positions_version = param.Integer(0)      # Bump to trigger position card re-render (price-sensitive)
    trades_version = param.Integer(0)        # Bump only on actual trade open/close (not price ticks)
    exit_alert_html = param.String('')        # Latest exit alert HTML (legacy)
    trade_alert_html = param.String('')       # Entry/exit alert card HTML
    trade_alert_type = param.String('')       # 'entry', 'exit_profit', 'exit_loss' (for audio)

    # Config (sidebar widgets bind to these)
    kill_switch = param.Boolean(False)
    algo_control_version = param.Integer(0)  # Bump on algo enable/disable/equity change

    order_version = param.Integer(0)       # Bump on order submit/fill/cancel

    # IB connection
    ib_connected = param.Boolean(False)
    ib_degraded = param.Boolean(False)      # IB/DB mismatch — block new entries

    # New infrastructure (rebuild plan Parts 1-8)
    # These are set during startup.full_init() and used by new tabs/loops
    trade_db = param.Parameter(None, precedence=-1)             # TradeDB instance
    price_manager = param.Parameter(None, precedence=-1)        # PriceManager instance
    ib_order_handler = param.Parameter(None, precedence=-1)     # IBOrderHandler instance
    fa_supported = param.Boolean(False, precedence=-1)          # IB FA account support

    # Internal
    _prev_price = param.Number(0.0, precedence=-1)
    _ml_model = param.Parameter(None, precedence=-1)
    _ml_feature_names = param.Parameter(None, precedence=-1)
    _ml_history_buffer = param.Parameter(None, precedence=-1)
    _intraday_ml_model = param.Parameter(None, precedence=-1)
    _intraday_ml_features = param.Parameter(None, precedence=-1)
    _intraday_trade_state = param.Parameter(None, precedence=-1)
    _el_model = param.Parameter(None, precedence=-1)   # ExtremeLoserDetector
    _er_model = param.Parameter(None, precedence=-1)   # ExtendedRunPredictor
    _ws_client = param.Parameter(None, precedence=-1)  # UNUSED — kept for param compat
    _price_updated_at = param.Number(0.0, precedence=-1)  # time.time() of last successful price fetch
    _last_intraday_exit_check = param.Number(0.0, precedence=-1)  # throttle intraday exits to 5s

    def load_market_data(self):
        """Startup: connect IB, load TF data. Fails loudly if IB unavailable."""
        logger.info("Loading market data...")

        # Price feed: IB only
        self._ws_client = None
        self._price_err_count = 0
        self.ib_client = None
        self._bar_aggregator = None
        self._historical_5min_tsla = None
        self._init_ib()

        # Load historical data from IB
        if self.ib_client and self.ib_client.is_connected():
            try:
                self.native_tf_data = self._load_ib_historical()
                if self.native_tf_data and 'TSLA' in self.native_tf_data:
                    self.data_source = 'IB'
                    logger.info("Data source: IB (all TFs)")
            except Exception as e:
                logger.error("IB historical load failed: %s", e)
                self.native_tf_data = {}
        else:
            logger.error("IB NOT CONNECTED — no data source available. "
                         "Dashboard will be empty until IB reconnects.")
            self.data_source = 'NONE'
            self.native_tf_data = {}

        # Fetch 5-min bars from IB
        if self.ib_client and self.ib_client.is_connected():
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
                logger.error("IB 5-min fetch failed: %s", e)

        # Set initial prices from loaded data
        if self.current_tsla is not None and len(self.current_tsla) > 0:
            self.tsla_price = float(self.current_tsla['close'].iloc[-1])
            if len(self.current_tsla) > 1:
                self._prev_price = float(self.current_tsla['close'].iloc[-2])
                self.price_delta = self.tsla_price - self._prev_price
            else:
                self._prev_price = self.tsla_price

        self._analysis_running = False

        # Run initial analysis for UI channel chart (synchronous at startup)
        try:
            results = self._run_analysis_core()
            if results:
                tsla_df, analysis, _dw = results
                if tsla_df is not None and len(tsla_df) > 0:
                    self.current_tsla = tsla_df
                self.analysis = analysis
                self.last_analysis = datetime.now().strftime('%H:%M:%S')
        except Exception as e:
            logger.error("Initial analysis failed: %s\n%s", e, traceback.format_exc())

    # Price updates handled by loops.py (ib_price_loop)

    def run_analysis(self):
        """Manual button: recompute channel analysis for UI display."""
        if not self.native_tf_data:
            logger.warning("No native TF data available for analysis")
            return
        if self._analysis_running:
            return

        self._analysis_running = True

        def _bg():
            try:
                results = self._run_analysis_core()
                if results:
                    tsla_df, analysis, _dw = results
                    def _apply():
                        if tsla_df is not None and len(tsla_df) > 0:
                            self.current_tsla = tsla_df
                        self.analysis = analysis
                        self.last_analysis = datetime.now().strftime('%H:%M:%S')
                    try:
                        pn.state.execute(_apply)
                    except Exception:
                        _apply()
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

        # Refresh 5-min bars from IB aggregator
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

        effective_tsla = tsla_df if tsla_df is not None and len(tsla_df) > 0 else self.current_tsla

        # CS-5TF analysis
        analysis = prepare_multi_tf_analysis(
            native_data=self.native_tf_data,
            live_5min_tsla=effective_tsla,
            target_tfs=['5min', '1h', '4h', 'daily', 'weekly'],
        )

        return tsla_df, analysis, None

    # Legacy scanner evaluation methods removed — LiveEngine handles all signal
    # generation via unified backtester algo classes.



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
            from v15.ib.client import IBClient
            cid = 7  # Fixed clientId so orders are always cancellable across restarts
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
                logger.info("Intraday ML model ready: %d features, threshold=%.2f",
                            len(self._intraday_ml_features), self._intraday_ml_threshold)
        except Exception as e:
            logger.warning("Intraday ML model load failed: %s — %s", e, traceback.format_exc())
            self._intraday_ml_model = None

    def send_notification(self, msg: str, title: str = '') -> str:
        """Send a push notification via Telegram. Returns status string."""
        bot_token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
        chat_id = os.environ.get('TELEGRAM_CHAT_ID', '').strip()
        if not bot_token or not chat_id:
            # Fall back to config/api_keys.json
            try:
                import json
                cfg_path = os.path.join(os.path.dirname(__file__), '..', '..',
                                        'config', 'api_keys.json')
                with open(cfg_path) as f:
                    cfg = json.load(f)
                bot_token = cfg.get('telegram', {}).get('bot_token', '')
                chat_id = cfg.get('telegram', {}).get('chat_id', '')
            except Exception as e:
                logger.warning("Failed to load Telegram config from api_keys.json: %s", e)
        if not bot_token or not chat_id:
            return 'NO_CHANNEL'

        try:
            import requests as _req
            text = f"{title or 'c14a Alert'}\n{'='*len(title or 'c14a Alert')}\n{msg}" if title else msg
            resp = _req.post(
                f'https://api.telegram.org/bot{bot_token}/sendMessage',
                json={'chat_id': chat_id, 'text': text},
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
            f"LiveEngine: {'OK' if getattr(self, 'live_engine', None) else 'NONE'}\n"
            f"Notifications are working!"
        )
        return self.send_notification(msg, title='Test Notification')
