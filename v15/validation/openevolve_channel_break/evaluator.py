"""
Evaluator for OpenEvolve Phase B3: Channel break predictor evolution.

LIGHTWEIGHT version — loads pre-computed channel features from pickle
(built by precompute_features.py) and walks forward with simple position
management. No BacktestEngine, no DataProvider, no channel detection.

Train/holdout split:
- Training (scored): 2015-01-01 to 2024-12-31
- Holdout: NOT run during training eval (too slow). Reported separately.

Scoring uses ONLY the training period.
"""

import importlib
import os
import pickle
import sys
import time as _time
import traceback

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))

# ── Config ────────────────────────────────────────────────────────────────

INITIAL_EQUITY = 100_000
MAX_EQUITY_PER_TRADE = 100_000
MAX_POSITIONS = 2
MAX_HOLD_EVALS = 60          # max hold in eval points (~60 * 15min = 15 hrs)
COST_PER_TRADE = 0.0         # no commissions (matches original)
SLIPPAGE_PCT = 0.0            # no slippage (matches original)

# Pre-computed features file
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_OUTPUT_DIR = os.path.join(_BASE_DIR, 'output')
_FEATURES_FILE = os.path.join(_OUTPUT_DIR, 'precomputed_features.pkl')

# ── Cached features (loaded once per process) ────────────────────────────

_CACHED_FEATURES = None


def _load_features():
    """Load pre-computed features from pickle. Cached after first call."""
    global _CACHED_FEATURES
    if _CACHED_FEATURES is not None:
        return _CACHED_FEATURES

    if not os.path.isfile(_FEATURES_FILE):
        raise FileNotFoundError(
            f"Pre-computed features not found: {_FEATURES_FILE}\n"
            f"Run precompute_features.py first.")

    t0 = _time.monotonic()
    with open(_FEATURES_FILE, 'rb') as f:
        _CACHED_FEATURES = pickle.load(f)
    elapsed = _time.monotonic() - t0
    print(f"[evaluator] Loaded {len(_CACHED_FEATURES)} pre-computed features "
          f"in {elapsed:.1f}s")
    return _CACHED_FEATURES


# ── Simple position tracking (mirrors B2 evaluator) ──────────────────────

class Position:
    __slots__ = ('pos_id', 'direction', 'entry_price', 'entry_eval_idx',
                 'stop_price', 'tp_price', 'size_dollars', 'shares',
                 'best_price', 'hold_evals', 'trail_width_mult',
                 'initial_stop_dist')

    def __init__(self, pos_id, direction, entry_price, entry_eval_idx,
                 stop_pct, tp_pct, size_dollars):
        self.pos_id = pos_id
        self.direction = direction
        self.entry_price = entry_price
        self.entry_eval_idx = entry_eval_idx
        self.size_dollars = size_dollars
        self.shares = size_dollars / entry_price
        self.trail_width_mult = 1.0
        self.initial_stop_dist = stop_pct

        if direction == 'long':
            self.stop_price = entry_price * (1 - stop_pct)
            self.tp_price = entry_price * (1 + tp_pct)
            self.best_price = entry_price
        else:
            self.stop_price = entry_price * (1 + stop_pct)
            self.tp_price = entry_price * (1 - tp_pct)
            self.best_price = entry_price

        self.hold_evals = 0


class Trade:
    __slots__ = ('direction', 'entry_price', 'exit_price', 'pnl',
                 'reason', 'hold_evals')

    def __init__(self, pos, exit_price, reason):
        self.direction = pos.direction
        self.entry_price = pos.entry_price
        self.exit_price = exit_price
        self.hold_evals = pos.hold_evals
        self.reason = reason

        if pos.direction == 'long':
            raw_pnl = (exit_price - pos.entry_price) * pos.shares
        else:
            raw_pnl = (pos.entry_price - exit_price) * pos.shares

        slippage = pos.size_dollars * SLIPPAGE_PCT * 2
        self.pnl = raw_pnl - COST_PER_TRADE - slippage


# ── Trailing stop (matches original evaluator) ───────────────────────────

def _get_effective_stop(pos):
    """Breakout-style trailing stop, same tiers as original."""
    entry = pos.entry_price
    if entry <= 0:
        return pos.stop_price

    trailing = pos.best_price
    isd = pos.initial_stop_dist
    twm = pos.trail_width_mult

    if pos.direction == 'long':
        profit_from_best = (trailing - entry) / entry
        if profit_from_best > 0.015:
            trail_price = trailing * (1 - isd * 0.01 * twm)
            return max(pos.stop_price, trail_price)
        elif profit_from_best > 0.008:
            trail_price = trailing * (1 - isd * 0.02 * twm)
            return max(pos.stop_price, trail_price)
        elif profit_from_best > 0.003:
            trail_price = trailing * (1 - isd * 0.05 * twm)
            return max(pos.stop_price, trail_price)
        else:
            return pos.stop_price
    else:
        profit_from_best = (entry - trailing) / entry
        if profit_from_best > 0.015:
            trail_price = trailing * (1 + isd * 0.01 * twm)
            return min(pos.stop_price, trail_price)
        elif profit_from_best > 0.008:
            trail_price = trailing * (1 + isd * 0.02 * twm)
            return min(pos.stop_price, trail_price)
        elif profit_from_best > 0.003:
            trail_price = trailing * (1 + isd * 0.05 * twm)
            return min(pos.stop_price, trail_price)
        else:
            return pos.stop_price


# ── Walk-forward backtest ─────────────────────────────────────────────────

def _run_backtest(features, predict_fn):
    """Walk forward through pre-computed features, return list of Trades.

    At each eval point:
    1. Check exits on open positions (using bar OHLC)
    2. Call predict_fn with channel_features, multi_tf_features, recent_bars
    3. Open positions if signal returned
    """
    positions = []
    trades = []
    pos_counter = 0
    err_logged = False

    for idx, row in enumerate(features):
        bar_high = row['bar_high']
        bar_low = row['bar_low']
        bar_close = row['bar_close']

        # ── Check exits on open positions ──
        closed_ids = set()
        for pos in positions:
            pos.hold_evals += 1

            # Update best price
            if pos.direction == 'long':
                pos.best_price = max(pos.best_price, bar_high)
            else:
                pos.best_price = min(pos.best_price, bar_low)

            exit_price = None
            reason = None

            effective_stop = _get_effective_stop(pos)

            if pos.direction == 'long':
                if bar_low <= effective_stop:
                    exit_price = effective_stop
                    reason = 'stop' if effective_stop == pos.stop_price else 'trail'
                elif bar_high >= pos.tp_price:
                    exit_price = pos.tp_price
                    reason = 'tp'
                elif pos.hold_evals >= MAX_HOLD_EVALS:
                    exit_price = bar_close
                    reason = 'timeout'
            else:
                if bar_high >= effective_stop:
                    exit_price = effective_stop
                    reason = 'stop' if effective_stop == pos.stop_price else 'trail'
                elif bar_low <= pos.tp_price:
                    exit_price = pos.tp_price
                    reason = 'tp'
                elif pos.hold_evals >= MAX_HOLD_EVALS:
                    exit_price = bar_close
                    reason = 'timeout'

            if exit_price is not None:
                trades.append(Trade(pos, exit_price, reason))
                closed_ids.add(pos.pos_id)

        if closed_ids:
            positions = [p for p in positions if p.pos_id not in closed_ids]

        # ── Signal generation ──
        if len(positions) >= MAX_POSITIONS:
            continue

        # Reconstruct recent_bars DataFrame from stored numpy arrays
        recent_bars = pd.DataFrame(
            row['recent_bars_values'],
            index=pd.DatetimeIndex(row['recent_bars_index']),
            columns=['open', 'high', 'low', 'close', 'volume'],
        )

        # Anti-pyramid: check existing directions
        existing_dirs = {p.direction for p in positions}

        try:
            prediction = predict_fn(
                row['channel_features'],
                row['multi_tf_features'],
                recent_bars,
            )
        except Exception:
            if not err_logged:
                print(f"PREDICT_ERROR: {traceback.format_exc()[-300:]}")
                err_logged = True
            continue

        if not isinstance(prediction, dict):
            continue

        signal_action = prediction.get('signal')
        if signal_action not in ('BUY', 'SELL'):
            continue

        confidence = prediction.get('confidence', 0.0)
        if confidence < 0.05:
            continue

        direction = 'long' if signal_action == 'BUY' else 'short'

        # Anti-pyramid
        if direction in existing_dirs:
            continue

        stop_pct = prediction.get('stop_pct', 0.010)
        tp_pct = prediction.get('tp_pct', 0.015)
        stop_pct = max(0.002, min(0.030, stop_pct))
        tp_pct = max(0.005, min(0.050, tp_pct))

        pos_counter += 1
        pos = Position(
            pos_id=f"p{pos_counter}",
            direction=direction,
            entry_price=bar_close,
            entry_eval_idx=idx,
            stop_pct=stop_pct,
            tp_pct=tp_pct,
            size_dollars=MAX_EQUITY_PER_TRADE,
        )
        positions.append(pos)

    # Close remaining positions at last bar's close
    if positions and len(features) > 0:
        last_close = features[-1]['bar_close']
        for pos in positions:
            trades.append(Trade(pos, last_close, 'eod_close'))

    return trades


# ── Metrics computation ───────────────────────────────────────────────────

def _compute_metrics(trades):
    """Compute performance metrics from trade list."""
    if not trades:
        return {
            'total_pnl': 0, 'total_trades': 0, 'win_rate': 0,
            'sharpe_ratio': 0, 'profit_factor': 0, 'max_drawdown_pct': 0,
            'avg_pnl': 0,
        }

    pnls = np.array([t.pnl for t in trades])
    total_pnl = float(np.sum(pnls))
    n_trades = len(trades)
    winners = int(np.sum(pnls > 0))
    win_rate = float(winners / n_trades * 100)
    avg_pnl = float(np.mean(pnls))

    # Sharpe (annualized from per-trade PnL)
    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Profit factor
    gross_profit = float(np.sum(pnls[pnls > 0]))
    gross_loss = float(np.abs(np.sum(pnls[pnls < 0])))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = 999.0 if gross_profit > 0 else 0.0

    # Max drawdown
    cumulative = np.cumsum(pnls)
    equity_curve = INITIAL_EQUITY + cumulative
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak * 100
    max_dd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    return {
        'total_pnl': total_pnl,
        'total_trades': n_trades,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'profit_factor': profit_factor,
        'max_drawdown_pct': max_dd,
        'avg_pnl': avg_pnl,
    }


# ── Scoring (same formula as original) ───────────────────────────────────

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


# ── Holdout: 5-sec honest backtest ────────────────────────────────────────

_HOLDOUT_DATA = None

def _resolve_path(*candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]

def _run_holdout_5sec(predict_fn):
    """Run 5-sec honest backtest on holdout period (2025-2026). Returns metrics or None."""
    global _HOLDOUT_DATA
    if _HOLDOUT_DATA is None:
        from v15.validation.unified_backtester.data_provider import DataProvider
        _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        tsla_5s = _resolve_path(r'C:\AI\x14\data\bars_5s\TSLA_5s.csv',
                                os.path.join(_PROJECT_ROOT, 'data', 'bars_5s', 'TSLA_5s.csv'))
        spy_5s = _resolve_path(r'C:\AI\x14\data\bars_5s\SPY_5s.csv',
                               os.path.join(_PROJECT_ROOT, 'data', 'bars_5s', 'SPY_5s.csv'))
        vix_1m = _resolve_path(r'C:\AI\x14\data\VIXMin_IB.txt',
                               os.path.join(_PROJECT_ROOT, 'data', 'VIXMin_IB.txt'))
        _HOLDOUT_DATA = DataProvider.from_5sec_bars(
            tsla_5s_path=tsla_5s, start='2025-01-01', end='2026-03-16',
            spy_5s_path=spy_5s if os.path.isfile(spy_5s) else None,
            vix_path=vix_1m if os.path.isfile(vix_1m) else None,
            rth_only=False)
        print(f"[evaluator] Holdout data loaded (5-sec)")

    from v15.validation.unified_backtester.algo_base import AlgoBase, AlgoConfig, Signal, ExitSignal, CostModel
    from v15.validation.unified_backtester.engine import BacktestEngine
    from v15.validation.unified_backtester.portfolio import PortfolioManager
    from v15.validation.unified_backtester.results import compute_metrics
    from v15.core.channel import detect_channels_multi_window, select_best_channel
    from v15.core.channel_surfer import analyze_channels, TF_WINDOWS

    def _ext_tf(ts):
        return {
            'position_pct': ts.position_pct, 'center_distance': ts.center_distance,
            'potential_energy': ts.potential_energy, 'kinetic_energy': ts.kinetic_energy,
            'momentum_direction': ts.momentum_direction, 'total_energy': ts.total_energy,
            'binding_energy': ts.binding_energy, 'entropy': ts.entropy,
            'oscillation_period': ts.oscillation_period, 'bars_to_next_bounce': ts.bars_to_next_bounce,
            'channel_health': ts.channel_health, 'slope_pct': ts.slope_pct,
            'width_pct': ts.width_pct, 'r_squared': ts.r_squared,
            'bounce_count': ts.bounce_count, 'channel_direction': ts.channel_direction,
            'ou_theta': ts.ou_theta, 'ou_half_life': ts.ou_half_life,
            'ou_reversion_score': ts.ou_reversion_score, 'break_prob': ts.break_prob,
            'break_prob_up': ts.break_prob_up, 'break_prob_down': ts.break_prob_down,
            'volume_score': ts.volume_score, 'momentum_turn_score': ts.momentum_turn_score,
            'momentum_is_turning': ts.momentum_is_turning, 'squeeze_score': ts.squeeze_score,
        }

    def _ext_ch(ch):
        return {
            'alternation_ratio': ch.alternation_ratio, 'false_break_rate': ch.false_break_rate,
            'complete_cycles': ch.complete_cycles, 'quality_score': ch.quality_score,
            'bars_since_last_touch': ch.bars_since_last_touch,
            'upper_touches': ch.upper_touches, 'lower_touches': ch.lower_touches,
        }

    class _HoldoutAlgo(AlgoBase):
        def __init__(self, config, data):
            super().__init__(config, data)
            self._ps = {}
        def warmup_bars(self): return 300
        def on_bar(self, time, bar, open_positions, context=None):
            try:
                df5 = self.data.get_bars('5min', time)
                if len(df5) < 20: return []
                df_slice = df5.tail(100)
                multi = detect_channels_multi_window(df_slice, windows=[10,15,20,30,40])
                best_ch, _ = select_best_channel(multi)
                if not best_ch or not best_ch.valid: return []
                cbt = {'5min': best_ch}; pbt = {'5min': df_slice['close'].values}
                cpr = {'5min': float(df_slice['close'].iloc[-1])}; vd = {}
                if 'volume' in df_slice.columns: vd['5min'] = df_slice['volume'].values
                for tfl, per in [('1h',pd.Timedelta(hours=1)),('4h',pd.Timedelta(hours=4)),('daily',pd.Timedelta(days=1))]:
                    try: tf_df = self.data.get_bars(tfl, time)
                    except: continue
                    if len(tf_df) == 0: continue
                    tfa = tf_df[tf_df.index + per <= time].tail(100)
                    if len(tfa) < 30: continue
                    try:
                        tm = detect_channels_multi_window(tfa, windows=TF_WINDOWS.get(tfl,[20,30,40]))
                        tc, _ = select_best_channel(tm)
                        if tc and tc.valid:
                            cbt[tfl] = tc; pbt[tfl] = tfa['close'].values
                            cpr[tfl] = float(tfa['close'].iloc[-1])
                            if 'volume' in tfa.columns: vd[tfl] = tfa['volume'].values
                    except: continue
                analysis = analyze_channels(cbt, pbt, cpr, volumes_by_tf=vd or None)
                ps = analysis.tf_states.get('5min')
                if not ps: return []
                cf = _ext_tf(ps); cf.update(_ext_ch(best_ch))
                cf['energy_ratio'] = cf['total_energy'] / max(cf['binding_energy'], 1e-10)
                mtf = {}
                for tfl in ('1h','4h','daily'):
                    ts = analysis.tf_states.get(tfl)
                    if ts: mtf[tfl] = _ext_tf(ts)
                r = predict_fn(cf, mtf, df_slice)
                if not r or not r.get('signal'): return []
                sa = r['signal']
                if sa not in ('BUY','SELL'): return []
                d = 'long' if sa == 'BUY' else 'short'
                if d in {p.direction for p in open_positions}: return []
                return [Signal(algo_id=self.config.algo_id, direction=d, price=bar['close'],
                    confidence=r.get('confidence',0.5),
                    stop_pct=float(np.clip(r.get('stop_pct',0.01),0.002,0.03)),
                    tp_pct=float(np.clip(r.get('tp_pct',0.02),0.005,0.05)),
                    signal_type='channel_break',
                    metadata={'signal_bar_high':bar['high'],'signal_bar_low':bar['low']})]
            except: return []
        def check_exits(self, time, bar, open_positions):
            exits = []
            for pos in open_positions:
                if pos.direction == 'long':
                    if bar['low'] <= pos.stop_price: exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.stop_price, reason='stop')); continue
                    if bar['high'] >= pos.tp_price: exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.tp_price, reason='tp')); continue
                else:
                    if bar['high'] >= pos.stop_price: exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.stop_price, reason='stop')); continue
                    if bar['low'] <= pos.tp_price: exits.append(ExitSignal(pos_id=pos.pos_id, price=pos.tp_price, reason='tp')); continue
                if pos.hold_bars >= 60: exits.append(ExitSignal(pos_id=pos.pos_id, price=bar['close'], reason='timeout'))
            return exits
        def on_fill(self, t): self._ps.pop(t.pos_id, None)
        def on_position_opened(self, p): self._ps[p.pos_id] = {}
        def serialize_state(self, pid): return {}
        def restore_state(self, pid, s): pass

    config = AlgoConfig(algo_id='ch-break-holdout', initial_equity=100_000.0,
        max_equity_per_trade=100_000.0, max_positions=2, primary_tf='5min',
        eval_interval=3, exit_check_tf='5min', cost_model=CostModel(),
        stop_check_mode='sequential', exit_grace_bars=5, stop_update_secs=60,
        stop_check_secs=5, grace_ratchet_secs=60, profit_activated_stop=True,
        max_hold_bars=60)

    algo = _HoldoutAlgo(config, _HOLDOUT_DATA)
    portfolio = PortfolioManager()
    portfolio.register_algo('ch-break-holdout', 100_000.0, 100_000.0, 2, config.cost_model)

    t0 = _time.monotonic()
    engine = BacktestEngine(_HOLDOUT_DATA, [algo], portfolio, verbose=False)
    engine.run()
    elapsed = _time.monotonic() - t0

    trades_list = portfolio.get_trades(algo_id='ch-break-holdout')
    if not trades_list:
        print(f"[evaluator] Holdout: 0 trades ({elapsed:.0f}s)")
        return None

    hm = compute_metrics(trades_list, 100_000.0)
    print(f"[evaluator] Holdout done in {elapsed:.0f}s")
    return hm


# ── Main evaluate() entry point ──────────────────────────────────────────

def evaluate(program_path: str) -> dict:
    """Evaluate a channel break predictor candidate. Returns score dict.

    Loads pre-computed features (from precompute_features.py), walks forward
    with simple position management. No channel detection during eval.
    """
    t_total = _time.monotonic()

    # ── Load pre-computed features ───────────────────────────────────
    try:
        features = _load_features()
    except Exception as e:
        return {'combined_score': 0.0, 'error': f'feature load failed: {e}'}

    t_load = _time.monotonic() - t_total
    print(f"[evaluator] Data loaded in {t_load:.1f}s, {len(features)} eval points")

    # ── Import candidate ─────────────────────────────────────────────
    try:
        spec = importlib.util.spec_from_file_location('candidate', program_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        candidate_predict = getattr(mod, 'predict_channel_break', None)
        if candidate_predict is None:
            return {'combined_score': 0.0, 'error': 'missing predict_channel_break()'}

    except Exception as e:
        return {'combined_score': 0.0, 'error': f'import failed: {e}'}

    # ── Run walk-forward backtest ────────────────────────────────────
    t_bt = _time.monotonic()
    try:
        trades = _run_backtest(features, candidate_predict)
    except Exception as e:
        return {'combined_score': 0.0,
                'error': f'backtest failed: {traceback.format_exc()[-500:]}'}

    bt_time = _time.monotonic() - t_bt
    n_trades = len(trades)
    print(f"[evaluator] Backtest done in {bt_time:.1f}s, {n_trades} trades")

    if n_trades == 0:
        return {'combined_score': 0.0, 'n_trades': 0, 'error': 'no trades'}

    # ── Compute metrics and score ────────────────────────────────────
    m = _compute_metrics(trades)
    score = _compute_score(m)

    total_time = _time.monotonic() - t_total
    print(f"[evaluator] Total eval time: {total_time:.1f}s | "
          f"score={score:.0f} PnL=${m['total_pnl']:.0f} trades={m['total_trades']} "
          f"WR={m['win_rate']:.1f}% Sharpe={m['sharpe_ratio']:.3f} "
          f"DD={m['max_drawdown_pct']:.1f}%")

    result = {
        'combined_score': score,
        'total_pnl': m['total_pnl'],
        'n_trades': m['total_trades'],
        'win_rate': m['win_rate'],
        'sharpe': m['sharpe_ratio'],
        'profit_factor': m['profit_factor'],
        'max_drawdown_pct': m['max_drawdown_pct'],
        'avg_pnl': m['avg_pnl'],
    }

    # ── Holdout: 5-sec honest backtest (2025-2026) — reported only ────
    try:
        holdout_m = _run_holdout_5sec(candidate_predict)
        if holdout_m:
            result['holdout_total_pnl'] = holdout_m['total_pnl']
            result['holdout_n_trades'] = holdout_m['total_trades']
            result['holdout_win_rate'] = holdout_m['win_rate']
            result['holdout_sharpe'] = holdout_m['sharpe_ratio']
            result['holdout_max_drawdown_pct'] = holdout_m['max_drawdown_pct']
            print(f"[evaluator] Holdout: PnL=${holdout_m['total_pnl']:.0f} "
                  f"trades={holdout_m['total_trades']} "
                  f"Sharpe={holdout_m['sharpe_ratio']:.3f} "
                  f"DD={holdout_m['max_drawdown_pct']:.1f}%")
    except Exception as e:
        print(f"[evaluator] Holdout failed: {e}")

    return result
