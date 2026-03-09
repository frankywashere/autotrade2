"""
OE-Sig5 Algorithm — Plug-in for unified backtester.

Evolved daily bounce signal from OpenEvolve. Uses TSLA/SPY/VIX daily +
TSLA weekly bars to detect long entries near lower channel boundaries
with ATR compression and multi-condition branches.

Exit logic: same as CS-combo (exponential trail, 10-day hold, 3% stop).
Entry: next-day RTH open (delayed_entry=True).
"""

import logging
import time as _time_mod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..algo_base import AlgoBase, AlgoConfig, Signal, ExitSignal, CostModel
from ..data_provider import DataProvider
from ..portfolio import Position

logger = logging.getLogger(__name__)


DEFAULT_OE_SIG5_CONFIG = AlgoConfig(
    algo_id='oe-sig5',
    initial_equity=100_000.0,
    max_equity_per_trade=100_000.0,
    max_positions=1,
    primary_tf='daily',
    eval_interval=1,
    exit_check_tf='5min',            # Check exits intraday (matching live)
    cost_model=CostModel(
        slippage_pct=0.0001,
        commission_per_share=0.005,
    ),
    params={
        'flat_sizing': True,
        'trail_base': 0.025,
        'trail_power': 12,
        'stop_pct': 0.03,
        'tp_pct': 0.04,
        'max_hold_days': 10,
        'cooldown_days': 0,
        'default_confidence': 0.7,
    },
)


def _compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _channel_at(df_slice):
    if len(df_slice) < 10:
        return None
    try:
        from v15.core.channel import detect_channel
        ch = detect_channel(df_slice)
        return ch if (ch and ch.valid) else None
    except Exception as e:
        logger.warning("OE-Sig5 channel detection failed: %s", e)
        return None


def _near_lower(price, ch, frac=0.25):
    if ch is None:
        return False
    lower = ch.lower_line[-1]
    upper = ch.upper_line[-1]
    w = upper - lower
    if w <= 0:
        return False
    return (price - lower) / w < frac


def _evolved_signal(i, tsla, spy, vix, tw, rt):
    """Core evolved signal logic. Returns 1 (long) or 0 (no signal)."""
    if i < 35 or tw is None or len(tw) < 50:
        return 0

    closes = tsla['close'].iloc[i - 20:i + 1].values.astype(float)
    highs = tsla['high'].iloc[i - 20:i + 1].values.astype(float)
    lows = tsla['low'].iloc[i - 20:i + 1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    atr_5 = tr[-5:].mean()
    atr_20 = tr.mean()
    if atr_5 >= 0.75 * atr_20:
        return 0

    vix_now = float(vix['close'].iloc[i])

    daily_date = tsla.index[i]
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 50:
        return 0
    close_w = float(tw['close'].iloc[wk_idx])

    in_channel_lower = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if _near_lower(close_w, ch, 0.25):
                in_channel_lower = True
                break

    if not in_channel_lower:
        if 18 <= vix_now <= 50 and i >= 20:
            in_ch30 = False
            for window in (20, 30, 40, 50):
                if wk_idx >= window:
                    ch30 = _channel_at(tw.iloc[wk_idx - window:wk_idx])
                    if _near_lower(close_w, ch30, 0.30):
                        in_ch30 = True
                        break
            if in_ch30:
                t_rsi_c = float(rt.iloc[i]) if not np.isnan(rt.iloc[i]) else 50.0
                t20 = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - 20]) - 1.0
                s20 = float(spy['close'].iloc[i]) / float(spy['close'].iloc[i - 20]) - 1.0
                if t_rsi_c < 33 and (s20 - t20) >= 0.08:
                    return 1
                if t_rsi_c < 38 and (s20 - t20) >= 0.06:
                    return 1
        return 0

    if 18 <= vix_now <= 50:
        if i >= 20:
            tsla_ret = (float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - 20])) - 1.0
            spy_ret = (float(spy['close'].iloc[i]) / float(spy['close'].iloc[i - 20])) - 1.0
            if (spy_ret - tsla_ret) >= 0.05:
                return 1
        if i >= 3:
            c_now = float(tsla['close'].iloc[i])
            c_3d = float(tsla['close'].iloc[i - 3])
            if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
                return 1
        if i >= 20:
            spy_now = float(spy['close'].iloc[i])
            spy_20d = float(spy['close'].iloc[i - 20])
            if spy_20d > 0 and spy_now > spy_20d:
                return 1
        if i >= 5:
            spy_now = float(spy['close'].iloc[i])
            spy_5d = float(spy['close'].iloc[i - 5])
            if spy_5d > 0 and spy_now > spy_5d:
                return 1
        if i >= 10:
            c_now_a = float(tsla['close'].iloc[i])
            c_10d = float(tsla['close'].iloc[i - 10])
            if c_10d > 0 and (c_now_a - c_10d) / c_10d < -0.06:
                return 1
        if i >= 20:
            ma_20 = float(tsla['close'].iloc[i - 20:i].astype(float).mean())
            c_now_a6 = float(tsla['close'].iloc[i])
            if ma_20 > 0 and (c_now_a6 - ma_20) / ma_20 < -0.08:
                return 1

    if 15 <= vix_now <= 50:
        if i >= 35:
            close_series = tsla['close'].iloc[:i + 1].astype(float)
            ema_12 = close_series.ewm(span=12, adjust=False).mean()
            ema_26 = close_series.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            if float(macd_line.iloc[-1] - signal_line.iloc[-1]) < 0:
                return 1
        t_rsi = float(rt.iloc[i]) if not np.isnan(rt.iloc[i]) else 50.0
        if t_rsi < 40:
            return 1
        if i >= 3:
            c_now = float(tsla['close'].iloc[i])
            c_3d = float(tsla['close'].iloc[i - 3])
            if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
                return 1
        if i >= 4:
            if all(float(tsla['close'].iloc[j]) < float(tsla['close'].iloc[j - 1])
                   for j in range(i - 3, i + 1)):
                return 1
        if i >= 20:
            vol_now = float(tsla['volume'].iloc[i])
            vol_avg = float(tsla['volume'].iloc[i - 20:i].astype(float).mean())
            o_now = float(tsla['open'].iloc[i])
            c_close = float(tsla['close'].iloc[i])
            if vol_avg > 0 and vol_now > 1.5 * vol_avg and o_now > 0 and (c_close - o_now) / o_now < -0.01:
                return 1
        if i >= 5:
            vix_5h = max(float(vix['close'].iloc[i - k]) for k in range(1, 6))
            if vix_5h >= 25 and vix_now <= vix_5h * 0.90:
                return 1
        if i >= 20:
            bb_closes = tsla['close'].iloc[i - 20:i].values.astype(float)
            bb_mean = bb_closes.mean()
            bb_std = bb_closes.std()
            if bb_std > 0 and float(tsla['close'].iloc[i]) < bb_mean - 2.0 * bb_std:
                return 1

    if 10 <= vix_now < 17:
        t_rsi_c = float(rt.iloc[i]) if not np.isnan(rt.iloc[i]) else 50.0
        if t_rsi_c < 43 and i >= 20:
            t_ret_c = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - 20]) - 1.0
            s_ret_c = float(spy['close'].iloc[i]) / float(spy['close'].iloc[i - 20]) - 1.0
            _div_c = s_ret_c - t_ret_c
            if 0.04 <= _div_c < 0.12:
                return 1

    if 10 <= vix_now < 15:
        t_rsi_ext = float(rt.iloc[i]) if not np.isnan(rt.iloc[i]) else 50.0
        if t_rsi_ext < 32:
            return 1

    return 0


class OESig5Algo(AlgoBase):
    """OE-Sig5: OpenEvolve daily bounce signal with combo-style exit."""

    def __init__(self, config: AlgoConfig = None, data: DataProvider = None):
        super().__init__(config or DEFAULT_OE_SIG5_CONFIG, data)
        self._cooldown_remaining = 0
        self._current_day = None

        if data is not None and not getattr(data, 'is_live', False):
            # Backtest: precompute from native_tf files
            print("  Loading OE-Sig5 signals...")
            t0 = _time_mod.time()
            self._day_signals = self._precompute_signals()
            print(f"  Done: {len(self._day_signals)} signal days in {_time_mod.time() - t0:.1f}s")
        else:
            # Live: compute incrementally from DataProvider
            self._day_signals = {}

    def _precompute_signals(self) -> Dict:
        """Precompute OE-Sig5 signals for all trading days."""
        from v15.data.native_tf import fetch_native_tf

        # Load native daily/weekly data (needs multi-year history for weekly channels)
        start_native = '2015-01-01'
        end_native = str(self.data.end_time.date() + pd.Timedelta(days=1))[:10]

        tsla_d = fetch_native_tf('TSLA', 'daily', start_native, end_native)
        spy_d = fetch_native_tf('SPY', 'daily', start_native, end_native)
        vix_d = fetch_native_tf('^VIX', 'daily', start_native, end_native)
        tsla_w = fetch_native_tf('TSLA', 'weekly', start_native, end_native)

        for df in [tsla_d, spy_d, vix_d, tsla_w]:
            df.columns = [c.lower() for c in df.columns]

        # Strip timezone if present (match backtest tz-naive convention)
        for df in [tsla_d, spy_d, vix_d, tsla_w]:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

        # Inner-join on common dates (matching live path which does the same)
        common_dates = tsla_d.index.intersection(spy_d.index).intersection(vix_d.index)
        tsla_d = tsla_d.loc[common_dates]
        spy_d = spy_d.loc[common_dates]
        vix_d = vix_d.loc[common_dates]

        tsla_rsi = _compute_rsi(tsla_d['close'], 14)

        # Align indices: find which daily bars are in our test period
        trading_days = self.data.trading_days
        signals = {}
        default_conf = self.config.params.get('default_confidence', 0.7)
        stop_pct = self.config.params.get('stop_pct', 0.03)

        for day in trading_days:
            day_ts = pd.Timestamp(day)
            # Find index in tsla_d
            if day_ts not in tsla_d.index:
                # Try date match
                matches = tsla_d.index[tsla_d.index.date == day]
                if len(matches) == 0:
                    continue
                idx = tsla_d.index.get_loc(matches[0])
            else:
                idx = tsla_d.index.get_loc(day_ts)

            sig = _evolved_signal(idx, tsla_d, spy_d, vix_d, tsla_w, tsla_rsi)
            if sig == 1:
                signals[day] = {
                    'action': 'BUY',
                    'confidence': default_conf,
                    'stop_pct': stop_pct,
                    'signal_type': 'oe_sig5',
                }

        return signals

    def warmup_bars(self) -> int:
        return 0

    def _compute_today_signal(self, time, day):
        """Compute OE-Sig5 signal for today using live data.

        Critical: align TSLA/SPY/VIX DataFrames by date before calling
        _evolved_signal, which uses positional iloc indexing.
        """
        try:
            tsla_d = self.data.get_bars('daily', time, symbol='TSLA')
            spy_d = self.data.get_bars('daily', time, symbol='SPY')
            vix_d = self.data.get_bars('daily', time, symbol='VIX')
        except Exception as e:
            logger.error("OE-Sig5: failed to get daily bars: %s", e)
            return

        if len(tsla_d) < 36 or len(spy_d) < 36 or len(vix_d) < 36:
            logger.debug("OE-Sig5: insufficient daily bars (TSLA=%d, SPY=%d, VIX=%d)",
                          len(tsla_d), len(spy_d), len(vix_d))
            return

        # Ensure lowercase columns
        for df in [tsla_d, spy_d, vix_d]:
            df.columns = [c.lower() for c in df.columns]

        # DATE ALIGNMENT: inner-join on DatetimeIndex so iloc[i] on all three
        # DataFrames refers to the same calendar date. Without this,
        # different-length DataFrames produce wrong cross-symbol comparisons.
        common_dates = tsla_d.index.intersection(spy_d.index).intersection(vix_d.index)
        if len(common_dates) < 36:
            logger.debug("OE-Sig5: insufficient aligned dates (%d)", len(common_dates))
            return
        tsla_d = tsla_d.loc[common_dates]
        spy_d = spy_d.loc[common_dates]
        vix_d = vix_d.loc[common_dates]

        # Get weekly bars (native from IB seeding, not resampled)
        try:
            tsla_w = self.data.get_bars('weekly', time, symbol='TSLA')
        except Exception as e:
            logger.debug("OE-Sig5: weekly bars not available, will resample from daily: %s", e)
            tsla_w = None

        # Fallback: resample from daily if no native weekly
        if tsla_w is None or len(tsla_w) < 51:
            tsla_w = tsla_d.resample('W-FRI').agg({
                'open': 'first', 'high': 'max', 'low': 'min',
                'close': 'last', 'volume': 'sum',
            }).dropna()

        if len(tsla_w) < 51:
            logger.debug("OE-Sig5: insufficient weekly bars (%d)", len(tsla_w))
            return

        tsla_rsi = _compute_rsi(tsla_d['close'], 14)

        # Evaluate signal at last aligned daily bar
        idx = len(tsla_d) - 1
        sig = _evolved_signal(idx, tsla_d, spy_d, vix_d, tsla_w, tsla_rsi)

        if sig == 1:
            default_conf = self.config.params.get('default_confidence', 0.7)
            stop_pct = self.config.params.get('stop_pct', 0.03)
            self._day_signals[day] = {
                'action': 'BUY',
                'confidence': default_conf,
                'stop_pct': stop_pct,
                'signal_type': 'oe_sig5',
            }
            logger.info("OE-Sig5 signal for %s: BUY conf=%.2f", day, default_conf)

    def on_bar(self, time: pd.Timestamp, bar: dict,
               open_positions: list,
               context=None) -> List[Signal]:
        """At end of trading day: look up precomputed OE signal."""
        params = self.config.params

        day = time.date()

        # Live mode: compute today's signal from available data
        if getattr(self.data, 'is_live', False) and day != self._current_day:
            self._compute_today_signal(time, day)

        if day != self._current_day:
            self._current_day = day
            if self._cooldown_remaining > 0:
                self._cooldown_remaining -= 1
                return []

        if self._cooldown_remaining > 0:
            return []
        if open_positions:
            return []

        sig_data = self._day_signals.get(day)
        if sig_data is None:
            return []

        return [Signal(
            algo_id=self.config.algo_id,
            direction='long',
            price=bar['close'],
            confidence=sig_data['confidence'],
            stop_pct=sig_data['stop_pct'],
            tp_pct=params.get('tp_pct', 0.04),
            signal_type='oe_sig5',
            delayed_entry=True,
        )]

    def check_exits(self, time: pd.Timestamp, bar: dict,
                    open_positions: list) -> List[ExitSignal]:
        """Same exit logic as CS-combo: exponential trail + stop + timeout."""
        exits = []
        params = self.config.params
        trail_base = params.get('trail_base', 0.025)
        trail_power = params.get('trail_power', 12)
        max_hold = params.get('max_hold_days', 10)

        for pos in open_positions:
            high = bar['high']
            low = bar['low']
            close = bar['close']

            trail_pct = trail_base * (1.0 - pos.confidence) ** trail_power

            if pos.direction == 'long':
                # Causal: use best_price from PRIOR bars only (engine updates after exits)
                best = pos.best_price
                trailing_stop = best * (1.0 - trail_pct)
                if best > pos.entry_price:
                    effective_stop = max(pos.stop_price, trailing_stop)
                else:
                    effective_stop = pos.stop_price

                if low <= effective_stop:
                    is_trailing = best > pos.entry_price and trailing_stop > pos.stop_price
                    reason = 'trailing' if is_trailing else 'stop'
                    exits.append(ExitSignal(
                        pos_id=pos.pos_id,
                        price=effective_stop,
                        reason=reason,
                    ))
                    self._cooldown_remaining = params.get('cooldown_days', 0)
                    continue

                # Check take profit
                if high >= pos.tp_price:
                    exits.append(ExitSignal(
                        pos_id=pos.pos_id, price=pos.tp_price, reason='tp'))
                    self._cooldown_remaining = params.get('cooldown_days', 0)
                    continue

                # Timeout: convert max_hold_days to 5-min bars (78 per day)
                max_hold_5m = max_hold * 78
                if pos.hold_bars >= max_hold_5m:
                    exits.append(ExitSignal(
                        pos_id=pos.pos_id,
                        price=close,
                        reason='timeout',
                    ))
                    self._cooldown_remaining = params.get('cooldown_days', 0)

        return exits

    def get_effective_stop(self, position) -> Optional[float]:
        """Return current effective stop for broker-side sync."""
        params = self.config.params
        trail_base = params.get('trail_base', 0.025)
        trail_power = params.get('trail_power', 12)
        trail_pct = trail_base * (1.0 - position.confidence) ** trail_power

        if position.direction == 'long':
            trailing_stop = position.best_price * (1.0 - trail_pct)
            if position.best_price > position.entry_price:
                return max(position.stop_price, trailing_stop)
            return position.stop_price
        else:
            trailing_stop = position.best_price * (1.0 + trail_pct)
            if position.best_price < position.entry_price:
                return min(position.stop_price, trailing_stop)
            return position.stop_price

    def on_fill(self, trade):
        pass

    def on_position_opened(self, position):
        pass
