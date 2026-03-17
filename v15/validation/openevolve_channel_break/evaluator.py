"""
Evaluator for OpenEvolve Phase B3: Channel break predictor evolution.

Train/holdout split:
- Training (scored): 2015-01-01 to 2024-12-31, 1-min bars, window stop mode
- Holdout (reported only): 2025-01-01 to 2026-03-16, 5-sec bars, sequential stops

Scoring uses ONLY the training period. Holdout metrics are reported for
monitoring overfitting but do NOT affect combined_score.
"""

import hashlib
import importlib
import os
import pickle
import sys
import traceback
import types

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

# ── Path resolution (Windows server first, then Mac) ─────────────────────

def _resolve_path(*candidates):
    """Return the first existing path from candidates."""
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]  # fallback to last (will error later)


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

# Training data (1-min, semicolon-delimited)
TSLA_1MIN_PATH = _resolve_path(
    r'C:\AI\x14\data\TSLAMin_yfinance_deprecated.txt',
    os.path.join(_PROJECT_ROOT, 'data', 'TSLAMin_yfinance_deprecated.txt'))
SPY_1MIN_PATH = _resolve_path(
    r'C:\AI\x14\data\SPYMin.txt',
    os.path.join(_PROJECT_ROOT, 'data', 'SPYMin.txt'))
VIX_1MIN_PATH = _resolve_path(
    r'C:\AI\x14\data\VIXMin_IB.txt',
    os.path.join(_PROJECT_ROOT, 'data', 'VIXMin_IB.txt'))

TRAIN_START = '2015-01-01'
TRAIN_END = '2024-12-31'

# Holdout data (5-sec bars)
BAR_DATA_DIR = _resolve_path(
    r'C:\AI\x14\data\bars_5s',
    os.path.join(_PROJECT_ROOT, 'data', 'bars_5s'))

HOLDOUT_START = '2025-01-01'
HOLDOUT_END = '2026-03-16'

# ── Cached data providers (loaded once per process) ──────────────────────

_TRAIN_DATA = None
_HOLDOUT_DATA = None


def _cache_key(*args) -> str:
    """Build a short hash from arguments for cache versioning."""
    h = hashlib.sha256('|'.join(str(a) for a in args).encode()).hexdigest()[:12]
    return h


def _pickle_path(name: str) -> str:
    """Return path to a pickle cache file in the output dir."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f'{name}.pkl')


def _load_train_data():
    """Load 1-min bar data for training period (2015-2024). Cached.

    Uses pickle cache on disk to avoid re-parsing 90MB+ text files on every
    subprocess evaluation.  First call builds the DataProvider and saves it;
    subsequent calls load from pickle (~2-5 s vs ~minutes).
    """
    global _TRAIN_DATA
    if _TRAIN_DATA is not None:
        return _TRAIN_DATA

    from v15.validation.unified_backtester.data_provider import DataProvider

    if not os.path.isfile(TSLA_1MIN_PATH):
        raise FileNotFoundError(f"TSLA 1-min data not found: {TSLA_1MIN_PATH}")

    spy_path = SPY_1MIN_PATH if os.path.isfile(SPY_1MIN_PATH) else None
    vix_path = VIX_1MIN_PATH if os.path.isfile(VIX_1MIN_PATH) else None

    # Cache version key: hash of paths + date range
    cache_key = _cache_key(TSLA_1MIN_PATH, spy_path, vix_path, TRAIN_START, TRAIN_END)
    pkl_path = _pickle_path(f'train_data_{cache_key}')

    # Try loading from pickle cache
    if os.path.isfile(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                _TRAIN_DATA = pickle.load(f)
            print(f"[evaluator] Loaded train data from pickle cache ({pkl_path})")
            return _TRAIN_DATA
        except Exception as e:
            print(f"[evaluator] Pickle cache load failed ({e}), rebuilding...")

    # DataProvider.__init__ loads 1-min, resamples all TFs, loads aux daily
    _TRAIN_DATA = DataProvider(
        tsla_1min_path=TSLA_1MIN_PATH,
        start=TRAIN_START,
        end=TRAIN_END,
        spy_path=spy_path,
        rth_only=False,
    )

    # Manually load VIX 1-min into the provider (DataProvider.__init__ doesn't
    # accept vix_path, but _init_from_df1m does via from_5sec_bars path).
    # We replicate what from_5sec_bars does for VIX.
    if vix_path and os.path.isfile(vix_path):
        from v15.validation.unified_backtester.data_provider import (
            load_1min, _resample_ohlcv, _RESAMPLE_RULES,
        )
        vix_1m = load_1min(vix_path, TRAIN_START, TRAIN_END, rth_only=False)
        if len(vix_1m) > 0:
            _TRAIN_DATA._vix1m = vix_1m
            _TRAIN_DATA._vix_tf_data = {'1min': vix_1m}
            for tf, rule in _RESAMPLE_RULES.items():
                if rule is not None:
                    _TRAIN_DATA._vix_tf_data[tf] = _resample_ohlcv(vix_1m, rule)

    # Save to pickle cache for subsequent subprocess evaluations
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(_TRAIN_DATA, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[evaluator] Saved train data pickle cache ({pkl_path})")
    except Exception as e:
        print(f"[evaluator] WARNING: failed to save pickle cache: {e}")

    return _TRAIN_DATA


def _load_holdout_data():
    """Load 5-sec bar data for holdout period (2025-2026). Cached.

    Uses pickle cache on disk to avoid re-parsing CSV files on every
    subprocess evaluation.
    """
    global _HOLDOUT_DATA
    if _HOLDOUT_DATA is not None:
        return _HOLDOUT_DATA

    from v15.validation.unified_backtester.data_provider import DataProvider

    tsla_5s = os.path.join(BAR_DATA_DIR, 'TSLA_5s.csv')
    spy_5s = os.path.join(BAR_DATA_DIR, 'SPY_5s.csv')
    if not os.path.isfile(tsla_5s):
        raise FileNotFoundError(f"TSLA 5-sec bars not found: {tsla_5s}")

    spy_path = spy_5s if os.path.isfile(spy_5s) else None
    vix_path = VIX_1MIN_PATH if os.path.isfile(VIX_1MIN_PATH) else None

    # Cache version key
    cache_key = _cache_key(tsla_5s, spy_path, vix_path, HOLDOUT_START, HOLDOUT_END)
    pkl_path = _pickle_path(f'holdout_data_{cache_key}')

    # Try loading from pickle cache
    if os.path.isfile(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                _HOLDOUT_DATA = pickle.load(f)
            print(f"[evaluator] Loaded holdout data from pickle cache ({pkl_path})")
            return _HOLDOUT_DATA
        except Exception as e:
            print(f"[evaluator] Pickle cache load failed ({e}), rebuilding...")

    _HOLDOUT_DATA = DataProvider.from_5sec_bars(
        tsla_5s_path=tsla_5s,
        start=HOLDOUT_START,
        end=HOLDOUT_END,
        spy_5s_path=spy_path,
        vix_path=vix_path,
        rth_only=False,
    )

    # Save to pickle cache
    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(_HOLDOUT_DATA, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[evaluator] Saved holdout data pickle cache ({pkl_path})")
    except Exception as e:
        print(f"[evaluator] WARNING: failed to save pickle cache: {e}")

    return _HOLDOUT_DATA


# ── Feature extraction helpers ───────────────────────────────────────────

def _extract_tf_state_features(tf_state) -> dict:
    """Extract all numeric features from a TFChannelState into a flat dict."""
    return {
        'position_pct': tf_state.position_pct,
        'center_distance': tf_state.center_distance,
        'potential_energy': tf_state.potential_energy,
        'kinetic_energy': tf_state.kinetic_energy,
        'momentum_direction': tf_state.momentum_direction,
        'total_energy': tf_state.total_energy,
        'binding_energy': tf_state.binding_energy,
        'entropy': tf_state.entropy,
        'oscillation_period': tf_state.oscillation_period,
        'bars_to_next_bounce': tf_state.bars_to_next_bounce,
        'channel_health': tf_state.channel_health,
        'slope_pct': tf_state.slope_pct,
        'width_pct': tf_state.width_pct,
        'r_squared': tf_state.r_squared,
        'bounce_count': tf_state.bounce_count,
        'channel_direction': tf_state.channel_direction,
        'ou_theta': tf_state.ou_theta,
        'ou_half_life': tf_state.ou_half_life,
        'ou_reversion_score': tf_state.ou_reversion_score,
        'break_prob': tf_state.break_prob,
        'break_prob_up': tf_state.break_prob_up,
        'break_prob_down': tf_state.break_prob_down,
        'volume_score': tf_state.volume_score,
        'momentum_turn_score': tf_state.momentum_turn_score,
        'momentum_is_turning': tf_state.momentum_is_turning,
        'squeeze_score': tf_state.squeeze_score,
    }


def _extract_channel_features(channel) -> dict:
    """Extract raw channel object features not in TFChannelState."""
    return {
        'alternation_ratio': channel.alternation_ratio,
        'false_break_rate': channel.false_break_rate,
        'complete_cycles': channel.complete_cycles,
        'quality_score': channel.quality_score,
        'bars_since_last_touch': channel.bars_since_last_touch,
        'upper_touches': channel.upper_touches,
        'lower_touches': channel.lower_touches,
    }


# ── ChannelBreakAlgo ─────────────────────────────────────────────────────

class ChannelBreakAlgo:
    """
    Backtester algo that uses candidate's predict_channel_break() for entries.

    Mirrors surfer_ml on_bar() channel detection setup, but instead of
    analyze_channels() for signal generation, extracts raw channel features
    and passes them to the candidate function.

    Exit logic uses breakout-style trailing stop.
    """

    def __init__(self, config, data, predict_fn):
        self.config = config
        self.data = data
        self._predict_fn = predict_fn
        self._pos_state = {}
        self._bar_count = 0
        # Precomputed channel cache (set externally)
        # Dict[pd.Timestamp, dict] with keys: analysis, best_ch, channels_by_tf, df_slice
        self._channel_cache = None

    @property
    def algo_id(self):
        return self.config.algo_id

    def warmup_bars(self) -> int:
        return 300

    def on_bar(self, time, bar, open_positions, context=None):
        """Run channel detection, extract features, call candidate predictor."""
        from v15.validation.unified_backtester.algo_base import Signal

        # Anti-pyramid: skip if already have position in same direction
        existing_dirs = {p.direction for p in open_positions}

        # ---- Channel detection: use precomputed cache or compute on-the-fly ----
        if self._channel_cache is not None:
            cached = self._channel_cache.get(time)
            if cached is None:
                return []
            analysis = cached['analysis']
            best_ch = cached['best_ch']
            channels_by_tf = cached['channels_by_tf']
            df_slice = cached['df_slice']
        else:
            # Slow path: compute channels on-the-fly (original behavior)
            from v15.core.channel import detect_channels_multi_window, select_best_channel
            from v15.core.channel_surfer import analyze_channels, TF_WINDOWS

            # Get 5-min data: last 100 bars (matching surfer_ml)
            df5 = self.data.get_bars('5min', time)
            if len(df5) < 20:
                return []
            df_slice = df5.tail(100)

            # Detect channels on 5-min slice (matching surfer_ml lines 226-228)
            try:
                multi = detect_channels_multi_window(df_slice, windows=[10, 15, 20, 30, 40])
                best_ch, _ = select_best_channel(multi)
            except Exception:
                return []

            if best_ch is None or not best_ch.valid:
                return []

            # Build multi-TF channel dict (matching surfer_ml lines 236-276)
            slice_closes = df_slice['close'].values
            channels_by_tf = {'5min': best_ch}
            prices_by_tf = {'5min': slice_closes}
            current_prices = {'5min': float(slice_closes[-1])}
            volumes_dict = {}
            if 'volume' in df_slice.columns:
                volumes_dict['5min'] = df_slice['volume'].values

            _TF_PERIOD = {
                '1h': pd.Timedelta(hours=1),
                '4h': pd.Timedelta(hours=4),
                'daily': pd.Timedelta(days=1),
            }
            for tf_label in ('1h', '4h', 'daily'):
                try:
                    tf_df = self.data.get_bars(tf_label, time)
                except (ValueError, KeyError):
                    continue
                if len(tf_df) == 0:
                    continue
                tf_period = _TF_PERIOD.get(tf_label, pd.Timedelta(hours=1))
                tf_available = tf_df[tf_df.index + tf_period <= time]
                tf_recent = tf_available.tail(100)
                if len(tf_recent) < 30:
                    continue
                tf_windows = TF_WINDOWS.get(tf_label, [20, 30, 40])
                try:
                    tf_multi = detect_channels_multi_window(tf_recent, windows=tf_windows)
                    tf_ch, _ = select_best_channel(tf_multi)
                    if tf_ch and tf_ch.valid:
                        channels_by_tf[tf_label] = tf_ch
                        prices_by_tf[tf_label] = tf_recent['close'].values
                        current_prices[tf_label] = float(tf_recent['close'].iloc[-1])
                        if 'volume' in tf_recent.columns:
                            volumes_dict[tf_label] = tf_recent['volume'].values
                except Exception:
                    continue

            # Run analyze_channels to get full TFChannelState objects
            try:
                analysis = analyze_channels(
                    channels_by_tf, prices_by_tf, current_prices,
                    volumes_by_tf=volumes_dict if volumes_dict else None,
                )
            except Exception:
                return []

        # Extract 5-min channel features
        primary_state = analysis.tf_states.get('5min')
        if primary_state is None or not primary_state.valid:
            return []

        channel_features = _extract_tf_state_features(primary_state)

        # Add raw channel features not in TFChannelState
        raw_ch_features = _extract_channel_features(best_ch)
        channel_features.update(raw_ch_features)

        # Compute energy_ratio explicitly (candidate expects it)
        binding = primary_state.binding_energy
        total_e = primary_state.total_energy
        channel_features['energy_ratio'] = total_e / max(binding, 0.01)

        # Build multi-TF features dict
        multi_tf_features = {}
        for tf_label in ('1h', '4h', 'daily'):
            tf_state = analysis.tf_states.get(tf_label)
            if tf_state is not None and tf_state.valid:
                tf_feats = _extract_tf_state_features(tf_state)
                # Add raw channel features if available
                if tf_label in channels_by_tf:
                    tf_feats.update(_extract_channel_features(channels_by_tf[tf_label]))
                multi_tf_features[tf_label] = tf_feats

        # Get recent 5-min bars as DataFrame for candidate
        recent_bars = df_slice.copy()

        # Call candidate predictor
        try:
            prediction = self._predict_fn(channel_features, multi_tf_features, recent_bars)
        except Exception:
            return []

        if not isinstance(prediction, dict):
            return []

        # Check if we should enter
        signal_action = prediction.get('signal')
        if signal_action not in ('BUY', 'SELL'):
            return []

        confidence = prediction.get('confidence', 0.0)
        if confidence < 0.05:
            return []

        direction = 'long' if signal_action == 'BUY' else 'short'

        # Anti-pyramid
        if direction in existing_dirs:
            return []

        stop_pct = prediction.get('stop_pct', 0.010)
        tp_pct = prediction.get('tp_pct', 0.015)

        # Clamp to reasonable ranges
        stop_pct = max(0.002, min(0.030, stop_pct))
        tp_pct = max(0.005, min(0.050, tp_pct))

        return [Signal(
            algo_id=self.config.algo_id,
            direction=direction,
            price=bar['close'],
            confidence=confidence,
            stop_pct=stop_pct,
            tp_pct=tp_pct,
            signal_type='break',
            metadata={
                'el_flagged': False,
                'trail_width_mult': 1.0,
                'fast_reversion': False,
                'ou_half_life': channel_features.get('ou_half_life', 5.0),
                'signal_bar_high': bar['high'],
                'signal_bar_low': bar['low'],
                'break_confidence': confidence,
                'break_direction': prediction.get('break_direction'),
            },
        )]

    def on_position_opened(self, position):
        """Initialize trail state (matching surfer_ml)."""
        sig_high = position.metadata.get('signal_bar_high', position.entry_price)
        sig_low = position.metadata.get('signal_bar_low', position.entry_price)
        self._pos_state[position.pos_id] = {
            'el_flagged': position.metadata.get('el_flagged', False),
            'trail_width_mult': position.metadata.get('trail_width_mult', 1.0),
            'fast_reversion': position.metadata.get('fast_reversion', False),
            'ou_half_life': position.metadata.get('ou_half_life', 5.0),
            'window_high': sig_high,
            'window_low': sig_low,
        }

    def on_fill(self, trade):
        """Clean up state on fill."""
        self._pos_state.pop(trade.pos_id, None)

    def serialize_state(self, pos_id: str) -> dict:
        return dict(self._pos_state.get(pos_id, {}))

    def restore_state(self, pos_id: str, state: dict):
        self._pos_state[pos_id] = state

    def get_effective_stop(self, position):
        """Breakout-style trailing stop (simplified for channel breaks).

        Uses the same profit-tier approach as surfer-ml but tuned for breakouts:
        tighter initial trail since we expect strong directional moves.
        """
        state = self._pos_state.get(position.pos_id, {})
        entry = position.entry_price
        if entry <= 0:
            return position.stop_price

        trailing = position.best_price
        initial_stop_dist = abs(position.stop_price - entry) / entry if entry > 0 else 0.01
        twm = state.get('trail_width_mult', 1.0)

        if position.direction == 'long':
            profit_from_best = (trailing - entry) / entry

            if profit_from_best > 0.015:
                # Deep in profit -- tight trail
                trail_price = trailing * (1 - initial_stop_dist * 0.01 * twm)
                return max(position.stop_price, trail_price)
            elif profit_from_best > 0.008:
                # Moderate profit -- medium trail
                trail_price = trailing * (1 - initial_stop_dist * 0.02 * twm)
                return max(position.stop_price, trail_price)
            elif profit_from_best > 0.003:
                # Small profit -- move to breakeven
                trail_price = trailing * (1 - initial_stop_dist * 0.05 * twm)
                return max(position.stop_price, trail_price)
            else:
                return position.stop_price
        else:
            profit_from_best = (entry - trailing) / entry

            if profit_from_best > 0.015:
                trail_price = trailing * (1 + initial_stop_dist * 0.01 * twm)
                return min(position.stop_price, trail_price)
            elif profit_from_best > 0.008:
                trail_price = trailing * (1 + initial_stop_dist * 0.02 * twm)
                return min(position.stop_price, trail_price)
            elif profit_from_best > 0.003:
                trail_price = trailing * (1 + initial_stop_dist * 0.05 * twm)
                return min(position.stop_price, trail_price)
            else:
                return position.stop_price

    def check_exits(self, time, bar, open_positions):
        """Check exits: stop/trail, TP, and timeouts."""
        from v15.validation.unified_backtester.algo_base import ExitSignal

        exits = []
        max_hold = self.config.max_hold_bars if self.config.max_hold_bars > 0 else 60
        eval_interval = self.config.eval_interval

        for pos in open_positions:
            state = self._pos_state.get(pos.pos_id, {})

            # Accumulate window high/low
            bar_high = bar['high']
            bar_low = bar['low']
            state.setdefault('window_high', bar_high)
            state.setdefault('window_low', bar_low)
            state['window_high'] = max(state['window_high'], bar_high)
            state['window_low'] = min(state['window_low'], bar_low)

            # Only evaluate exit every eval_interval bars
            state.setdefault('exit_bar_count', 0)
            state['exit_bar_count'] += 1
            if state['exit_bar_count'] < eval_interval:
                continue
            state['exit_bar_count'] = 0

            high = state['window_high']
            low = state['window_low']
            close = bar['close']
            state['window_high'] = bar['high']
            state['window_low'] = bar['low']

            # Stop/trail check
            effective_stop = self.get_effective_stop(pos)

            if pos.direction == 'long':
                if low <= effective_stop:
                    reason = 'stop' if effective_stop == pos.stop_price else 'trail'
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=effective_stop, reason=reason))
                    continue
                if high >= pos.tp_price:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.tp_price, reason='tp'))
                    continue
            else:
                if high >= effective_stop:
                    reason = 'stop' if effective_stop == pos.stop_price else 'trail'
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=effective_stop, reason=reason))
                    continue
                if low <= pos.tp_price:
                    exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.tp_price, reason='tp'))
                    continue

            # Hard timeout
            if pos.hold_bars >= max_hold:
                exits.append(ExitSignal(pos_id=pos.pos_id, price=close, reason='timeout'))

        return exits


# ── Channel cache ─────────────────────────────────────────────────────

def _load_channel_cache(data, label: str):
    """Load or build the precomputed channel cache for ChannelBreakAlgo.

    Uses precompute_channels_full which stores the raw Channel
    objects and df_slice needed for feature extraction.
    """
    cache_key = _cache_key(label, 'break_channel_cache_v1')
    pkl_path = _pickle_path(f'{label}_break_channel_cache_{cache_key}')

    if os.path.isfile(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                cache = pickle.load(f)
            print(f"[evaluator:break] Loaded channel cache from {pkl_path} "
                  f"({len(cache)} entries)")
            return cache
        except Exception as e:
            print(f"[evaluator:break] Channel cache load failed ({e}), rebuilding...")

    from v15.core.channel_cache import precompute_channels_full
    import logging
    logging.basicConfig(level=logging.INFO)

    print(f"[evaluator:break] Precomputing channel cache for {label}...")
    cache = precompute_channels_full(data, eval_interval=3)
    print(f"[evaluator:break] Channel cache built: {len(cache)} entries")

    try:
        with open(pkl_path, 'wb') as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
        print(f"[evaluator:break] Saved channel cache ({pkl_path}, {size_mb:.0f} MB)")
    except Exception as e:
        print(f"[evaluator:break] WARNING: failed to save channel cache: {e}")

    return cache


_TRAIN_CHANNEL_CACHE = None
_HOLDOUT_CHANNEL_CACHE = None


def _get_train_channel_cache(data):
    """Get or build the training channel cache."""
    global _TRAIN_CHANNEL_CACHE
    if _TRAIN_CHANNEL_CACHE is not None:
        return _TRAIN_CHANNEL_CACHE
    _TRAIN_CHANNEL_CACHE = _load_channel_cache(data, 'train')
    return _TRAIN_CHANNEL_CACHE


def _get_holdout_channel_cache(data):
    """Get or build the holdout channel cache."""
    global _HOLDOUT_CHANNEL_CACHE
    if _HOLDOUT_CHANNEL_CACHE is not None:
        return _HOLDOUT_CHANNEL_CACHE
    _HOLDOUT_CHANNEL_CACHE = _load_channel_cache(data, 'holdout')
    return _HOLDOUT_CHANNEL_CACHE


# ── Backtest runner (shared by train + holdout) ──────────────────────────

def _run_backtest(data, predict_fn, stop_check_mode='window', use_sequential=False,
                  channel_cache=None):
    """Run a single backtest on the given DataProvider, return (metrics, n_trades).

    Args:
        data: DataProvider instance
        predict_fn: candidate's predict_channel_break function
        stop_check_mode: 'window' for 1-min data (no sub-minute), 'sequential' for 5-sec
        use_sequential: True to enable sequential stop mode (holdout with 5-sec data)
    """
    from v15.validation.unified_backtester.algo_base import AlgoConfig, CostModel
    from v15.validation.unified_backtester.engine import BacktestEngine
    from v15.validation.unified_backtester.portfolio import PortfolioManager
    from v15.validation.unified_backtester.results import compute_metrics

    # Determine stop mode
    if use_sequential:
        mode = 'sequential'
    else:
        mode = 'current'  # 1-min data: check at bar close (no sub-minute)

    config = AlgoConfig(
        algo_id='channel-break',
        initial_equity=100_000.0,
        max_equity_per_trade=100_000.0,
        max_positions=2,
        primary_tf='5min',
        eval_interval=3,            # Every 3 bars = 15 min
        exit_check_tf='5min',
        cost_model=CostModel(
            slippage_pct=0.0,
            commission_per_share=0.0,
        ),
        stop_check_mode=mode,
        exit_grace_bars=5,
        stop_update_secs=60,
        stop_check_secs=5,
        grace_ratchet_secs=60,
        profit_activated_stop=True,
        max_underwater_mins=0,
        max_hold_bars=60,
    )

    algo = ChannelBreakAlgo(config, data, predict_fn)

    # Set precomputed channel cache (skips on-the-fly detection in on_bar)
    if channel_cache is not None:
        algo._channel_cache = channel_cache

    portfolio = PortfolioManager()
    portfolio.register_algo(
        algo_id=config.algo_id,
        initial_equity=config.initial_equity,
        max_per_trade=config.max_equity_per_trade,
        max_positions=config.max_positions,
        cost_model=config.cost_model,
    )

    engine = BacktestEngine(data, [algo], portfolio, verbose=False)
    results = engine.run()

    trades = portfolio.get_trades(algo_id='channel-break')
    if not trades:
        return None, 0

    m = compute_metrics(trades, config.initial_equity)
    return m, m['total_trades']


# ── Scoring ──────────────────────────────────────────────────────────────

def _compute_score(m):
    """Compute composite score from metrics dict. Returns 0.0 for bad results."""
    if m is None:
        return 0.0

    total_pnl = m['total_pnl']
    win_rate = m['win_rate']
    sharpe = m['sharpe_ratio']
    pf = m['profit_factor']
    dd = m['max_drawdown_pct']
    n_trades = m['total_trades']

    # Trade count penalty
    if n_trades < 50:
        trade_mult = 0.2
    elif n_trades < 200:
        trade_mult = 0.5 + 0.5 * (n_trades - 50) / 150
    elif n_trades <= 3000:
        trade_mult = 1.0
    elif n_trades <= 5000:
        trade_mult = 1.0 - 0.3 * (n_trades - 3000) / 2000
    else:
        trade_mult = 0.7

    # Drawdown penalty
    if dd <= 10:
        dd_mult = 1.0
    elif dd <= 20:
        dd_mult = 1.0 - 0.3 * (dd - 10) / 10
    elif dd <= 40:
        dd_mult = 0.7 - 0.4 * (dd - 20) / 20
    else:
        dd_mult = 0.3

    # Score
    if total_pnl <= 0:
        return 0.0

    score = (total_pnl
             * (1.0 + max(sharpe, 0) * 0.2)
             * (0.3 + win_rate * 0.7)
             * (1.0 + max(pf - 1, 0) * 0.1)
             * trade_mult
             * dd_mult)

    return max(score, 0.0)


# ── Main evaluate() entry point ──────────────────────────────────────────

def evaluate(program_path: str) -> dict:
    """Evaluate a channel break predictor candidate. Returns score dict.

    Runs training (2015-2024 on 1-min) and holdout (2025-2026 on 5-sec).
    combined_score comes ONLY from training. Holdout is reported for monitoring.
    """
    # ── Load training data ────────────────────────────────────────────
    try:
        train_data = _load_train_data()
    except Exception as e:
        return {'combined_score': 0.0, 'error': f'train data load failed: {e}'}

    # ── Import candidate ──────────────────────────────────────────────
    try:
        spec = importlib.util.spec_from_file_location('candidate', program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        candidate_predict = getattr(mod, 'predict_channel_break', None)
        if candidate_predict is None:
            return {'combined_score': 0.0, 'error': 'missing predict_channel_break()'}

    except Exception as e:
        return {'combined_score': 0.0, 'error': f'import failed: {e}'}

    # Load precomputed channel cache (built once, reused across evaluations)
    try:
        train_cc = _get_train_channel_cache(train_data)
    except Exception as e:
        print(f"[evaluator:break] WARNING: channel cache failed ({e}), running without")
        train_cc = None

    # ── Run training backtest (1-min, 2015-2024) ──────────────────────
    try:
        train_m, train_n = _run_backtest(
            train_data, candidate_predict,
            stop_check_mode='current',
            use_sequential=False,
            channel_cache=train_cc,
        )
    except Exception as e:
        return {'combined_score': 0.0, 'error': f'train backtest failed: {traceback.format_exc()}'}

    if train_m is None:
        return {'combined_score': 0.0, 'n_trades': 0, 'error': 'no trades (train)'}

    train_score = _compute_score(train_m)

    # ── Build result dict (training = scored) ─────────────────────────
    result = {
        'combined_score': train_score,
        'total_pnl': train_m['total_pnl'],
        'n_trades': train_m['total_trades'],
        'win_rate': train_m['win_rate'],
        'sharpe': train_m['sharpe_ratio'],
        'profit_factor': train_m['profit_factor'],
        'max_drawdown_pct': train_m['max_drawdown_pct'],
        'avg_pnl': train_m['avg_pnl'],
    }

    # ── Run holdout backtest (5-sec, 2025-2026) — reported only ───────
    try:
        holdout_data = _load_holdout_data()

        # Load holdout channel cache
        try:
            holdout_cc = _get_holdout_channel_cache(holdout_data)
        except Exception as e:
            print(f"[evaluator:break] WARNING: holdout channel cache failed ({e})")
            holdout_cc = None

        holdout_m, holdout_n = _run_backtest(
            holdout_data, candidate_predict,
            stop_check_mode='sequential',
            use_sequential=True,
            channel_cache=holdout_cc,
        )
        if holdout_m is not None:
            result['holdout_total_pnl'] = holdout_m['total_pnl']
            result['holdout_n_trades'] = holdout_m['total_trades']
            result['holdout_win_rate'] = holdout_m['win_rate']
            result['holdout_sharpe'] = holdout_m['sharpe_ratio']
            result['holdout_profit_factor'] = holdout_m['profit_factor']
            result['holdout_max_drawdown_pct'] = holdout_m['max_drawdown_pct']
            result['holdout_avg_pnl'] = holdout_m['avg_pnl']
            result['holdout_score'] = _compute_score(holdout_m)
        else:
            result['holdout_n_trades'] = 0
            result['holdout_error'] = 'no trades (holdout)'
    except Exception as e:
        result['holdout_error'] = f'holdout failed: {e}'

    return result
