"""
Intraday Enhanced-Union Algorithm — Plug-in for unified backtester.

Replicates intraday_v14b_janfeb.py config I (FD eUnion m30 FLAT $100K).

Signal: sig_union_enhanced with WIDER_PARAMS
Sizing: Flat $100K / price
Trail: 0.006 * (1-conf)^6 with floor 0.002
Window: 9:30-15:25 ET (full day)
Max trades per day: 30
"""

import datetime as dt
import time as _time_mod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..algo_base import AlgoBase, AlgoConfig, Signal, ExitSignal, CostModel
from ..data_provider import DataProvider
from ..portfolio import Position


# Default config matching backtest config I
DEFAULT_INTRADAY_CONFIG = AlgoConfig(
    algo_id='intraday',
    initial_equity=100_000.0,
    max_equity_per_trade=100_000.0,
    max_positions=1,            # Only 1 position at a time (matching backtest)
    primary_tf='5min',
    eval_interval=1,            # Every 5-min bar
    exit_check_tf='5min',       # Check exits on 5-min bars (matching backtest default)
    active_start=dt.time(9, 30),  # Engine-level hint: skip on_bar before 9:30
    active_end=dt.time(15, 25),   # Engine-level hint: skip on_bar after 15:25
    cost_model=CostModel(
        slippage_pct=0.0002,
        commission_per_share=0.005,  # Matching intraday_v14b_janfeb.py
    ),
    params={
        'flat_sizing': True,
        'trail_base': 0.006,
        'trail_power': 6,
        'trail_floor': 0.0,  # No floor — matching original backtest
        'stop_pct': 0.008,
        'tp_pct': 0.020,
        'max_trades_per_day': 30,
        'intraday_start': dt.time(9, 30),
        'intraday_end': dt.time(15, 25),
        'signal_params': {   # WIDER_PARAMS matching backtest config I
            'vwap_thresh': -0.10,
            'd_min': 0.20,
            'h1_min': 0.15,
            'f5_thresh': 0.35,
            'div_thresh': 0.20,
            'div_f5_thresh': 0.35,  # Match original (uses f5_thresh for both)
            'min_vol_ratio': 0.8,
            'stop': 0.008,
            'tp': 0.020,
        },
        'ml_model_path': None,
        'ml_threshold': 0.5,
    },
)


def _channel_position(close_arr, window=60):
    """Compute channel position using linear regression (matching backtest)."""
    n = len(close_arr)
    close = close_arr.astype(np.float64)
    w = window
    sx = w * (w - 1) / 2.0
    sx2 = (w - 1) * w * (2 * w - 1) / 6.0
    denom = w * sx2 - sx ** 2
    cy = np.cumsum(close)
    sy = np.full(n, np.nan)
    sy[w - 1] = cy[w - 1]
    if n > w:
        sy[w:] = cy[w:] - cy[:n - w]
    idx = np.arange(n, dtype=np.float64)
    cwy = np.cumsum(idx * close)
    sxy = np.full(n, np.nan)
    sxy[w - 1] = cwy[w - 1]
    if n > w:
        si = np.arange(w, n, dtype=np.float64) - w + 1
        sxy[w:] = (cwy[w:] - cwy[:n - w]) - si * sy[w:]
    slope = (w * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / w
    fl = slope * (w - 1) + intercept
    cy2 = np.cumsum(close ** 2)
    sy2 = np.full(n, np.nan)
    sy2[w - 1] = cy2[w - 1]
    if n > w:
        sy2[w:] = cy2[w:] - cy2[:n - w]
    vy = (sy2 - sy ** 2 / w) / w
    vx = denom / (w ** 2)
    vr = np.maximum(vy - slope ** 2 * vx, 0)
    sr = np.sqrt(vr)
    u = fl + 2 * sr
    l_ = fl - 2 * sr
    wi = u - l_
    pos = np.full(n, np.nan)
    v = (wi > 1e-10) & ~np.isnan(wi)
    pos[v] = (close[v] - l_[v]) / wi[v]
    return np.clip(pos, 0.0, 1.0)


def _channel_slope(close_arr, window=60):
    """Compute channel slope (matching backtest)."""
    n = len(close_arr)
    close = close_arr.astype(np.float64)
    w = window
    sx = w * (w - 1) / 2.0
    sx2 = (w - 1) * w * (2 * w - 1) / 6.0
    denom = w * sx2 - sx ** 2
    cy = np.cumsum(close)
    sy = np.full(n, np.nan)
    sy[w - 1] = cy[w - 1]
    if n > w:
        sy[w:] = cy[w:] - cy[:n - w]
    idx = np.arange(n, dtype=np.float64)
    cwy = np.cumsum(idx * close)
    sxy = np.full(n, np.nan)
    sxy[w - 1] = cwy[w - 1]
    if n > w:
        si = np.arange(w, n, dtype=np.float64) - w + 1
        sxy[w:] = (cwy[w:] - cwy[:n - w]) - si * sy[w:]
    slope_arr = (w * sxy - sx * sy) / denom
    ns = np.full(n, np.nan)
    valid = close > 0
    ns[valid] = slope_arr[valid] / close[valid]
    return ns


def _resample_ohlcv(df, rule):
    return df.resample(rule).agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open'])


def _compute_vwap(o, h, l, c, v, dates):
    n = len(c)
    vwap = np.full(n, np.nan)
    vd = np.full(n, np.nan)
    tp = (h + l + c) / 3.0
    ctv = cv = 0.0
    pd_ = None
    for i in range(n):
        d = dates[i]
        if d != pd_:
            ctv = cv = 0.0
            pd_ = d
        ctv += tp[i] * v[i]
        cv += v[i]
        if cv > 0:
            vwap[i] = ctv / cv
            vd[i] = (c[i] - vwap[i]) / vwap[i] * 100.0
    return vwap, vd


def _compute_volume_ratio(v, lb=20):
    n = len(v)
    vr = np.full(n, np.nan)
    rv = pd.Series(v).rolling(lb, min_periods=lb).mean().values
    valid = (rv > 0) & ~np.isnan(rv)
    vr[valid] = v[valid] / rv[valid]
    return vr


def _compute_vwap_slope(vd, lb=5):
    n = len(vd)
    vs = np.full(n, np.nan)
    for i in range(lb, n):
        seg = vd[i - lb + 1:i + 1]
        if np.any(np.isnan(seg)):
            continue
        x = np.arange(lb, dtype=np.float64)
        mx = x.mean()
        my = seg.mean()
        vs[i] = np.sum((x - mx) * (seg - my)) / np.sum((x - mx) ** 2)
    return vs


def _compute_spread_pct(h, l, c):
    sp = np.full(len(c), np.nan)
    valid = c > 0
    sp[valid] = (h[valid] - l[valid]) / c[valid] * 100.0
    return sp


def _compute_gap_pct(c, dates):
    n = len(c)
    gap = np.full(n, np.nan)
    prev_close = None
    prev_date = None
    for i in range(n):
        d = dates[i]
        if d != prev_date:
            if prev_close is not None and prev_close > 0:
                gap[i] = (c[i] - prev_close) / prev_close * 100.0
            prev_date = d
        prev_close = c[i]
    cur_gap = np.nan
    for i in range(n):
        if not np.isnan(gap[i]):
            cur_gap = gap[i]
        gap[i] = cur_gap
    return gap


class IntradayAlgo(AlgoBase):
    """Intraday VWAP/Divergence union signal algorithm.

    Precomputes all features from 5-min data at init (matching backtest),
    then looks them up during the walk-forward simulation.
    """

    def __init__(self, config: AlgoConfig = None, data: DataProvider = None):
        super().__init__(config or DEFAULT_INTRADAY_CONFIG, data)
        self._trades_today = 0
        self._current_day = None

        # Precompute features (matching backtest build_features + precompute_all)
        print("  Precomputing intraday features...")
        t0 = _time_mod.time()
        self._precompute_features()
        print(f"  Done in {_time_mod.time() - t0:.1f}s")

    def _precompute_features(self):
        """Precompute all features from 5-min data (matching backtest exactly)."""
        f5m = self.data._tf_data['5min']
        n = len(f5m)
        self._f5m = f5m
        self._n = n
        self._bar_index = {}
        idx_1m = self.data._tf_data['1min'].index
        dates_1m = idx_1m.date
        for i, ts in enumerate(f5m.index):
            self._bar_index[ts] = i
            if i + 1 < len(f5m.index):
                next_ts = f5m.index[i + 1]
                end_pos = idx_1m.searchsorted(next_ts, side='left') - 1
            else:
                day_mask = dates_1m == ts.date()
                end_pos = np.flatnonzero(day_mask)[-1] if day_mask.any() else -1
            if end_pos >= 0:
                self._bar_index[idx_1m[end_pos]] = i

        c = f5m['close'].values.astype(np.float64)
        h = f5m['high'].values.astype(np.float64)
        l = f5m['low'].values.astype(np.float64)
        o = f5m['open'].values.astype(np.float64)
        v = f5m['volume'].values.astype(np.float64)
        dates = np.array([t.date() for t in f5m.index])

        # Channel windows matching backtest
        wins = {'5m': 60, '15m': 40, '30m': 30, '1h': 24, '4h': 20, 'daily': 40}

        # 5-min features
        self._cp5 = _channel_position(c, wins['5m'])

        # VWAP features
        _, self._vwap_dist = _compute_vwap(o, h, l, c, v, dates)
        self._vol_ratio = _compute_volume_ratio(v)
        self._vwap_slope = _compute_vwap_slope(self._vwap_dist)
        self._spread_pct = _compute_spread_pct(h, l, c)
        self._gap_pct = _compute_gap_pct(c, dates)

        # Higher TF features (precomputed and mapped to 5-min bars)
        # Resample from 1-min data
        df1m = self.data._tf_data['1min']
        tfs = {}
        for rule, name in [('1h', '1h'), ('4h', '4h'), ('15min', '15m'), ('30min', '30m')]:
            tfs[name] = _resample_ohlcv(df1m, rule)
        tfs['daily'] = _resample_ohlcv(df1m, '1D')

        # Compute channel positions for each TF
        for name, df in tfs.items():
            c_tf = df['close'].values.astype(np.float64)
            df_feat = pd.DataFrame(index=df.index)
            df_feat['chan_pos'] = _channel_position(c_tf, wins[name])
            if name in ('1h', '4h', 'daily'):
                df_feat['chan_slope'] = _channel_slope(c_tf, wins[name])
            tfs[name] = df_feat

        # Map higher TF values to 5-min bars (point-in-time, no lookahead)
        self._h1_cp = np.full(n, np.nan)
        self._h1_slope = np.full(n, np.nan)
        self._h4_cp = np.full(n, np.nan)
        self._h4_slope = np.full(n, np.nan)
        self._daily_cp = np.full(n, np.nan)
        self._daily_slope = np.full(n, np.nan)

        # Daily: use PRIOR day's value (no lookahead)
        daily_f = tfs['daily']
        daily_cp = daily_f['chan_pos'].values
        daily_slope = daily_f['chan_slope'].values
        daily_dates = np.array([idx.date() if hasattr(idx, 'date') else idx for idx in daily_f.index])
        bar_dates = dates
        d2d = {}
        unique_days = sorted(set(bar_dates))
        for d in unique_days:
            best = -1
            for k in range(len(daily_dates)):
                if daily_dates[k] < d:
                    best = k
                elif daily_dates[k] >= d:
                    break
            if best >= 0:
                d2d[d] = best
        for i in range(n):
            d = bar_dates[i]
            if d in d2d:
                k = d2d[d]
                self._daily_cp[i] = daily_cp[k]
                self._daily_slope[i] = daily_slope[k]

        # Intraday TFs: map using point-in-time lookup
        configs = [
            (tfs['1h'], {'1h_cp': 'chan_pos', '1h_slope': 'chan_slope'}),
            (tfs['4h'], {'4h_cp': 'chan_pos', '4h_slope': 'chan_slope'}),
        ]
        for tf_feat, col_map in configs:
            ti = tf_feat.index.values
            ca = {k: tf_feat[v].values for k, v in col_map.items()}
            j = 0
            for i in range(n):
                t = f5m.index[i]
                while j < len(ti) - 1 and ti[j + 1] <= t:
                    j += 1
                if j < len(ti) and ti[j] <= t:
                    for k in col_map:
                        if k == '1h_cp':
                            self._h1_cp[i] = ca[k][j]
                        elif k == '1h_slope':
                            self._h1_slope[i] = ca[k][j]
                        elif k == '4h_cp':
                            self._h4_cp[i] = ca[k][j]
                        elif k == '4h_slope':
                            self._h4_slope[i] = ca[k][j]

    def warmup_bars(self) -> int:
        return 100

    def on_bar(self, time: pd.Timestamp, bar: dict,
               open_positions: list) -> List[Signal]:
        """Evaluate intraday signal on each 5-min bar."""
        params = self.config.params
        sig_params = params.get('signal_params', {})
        bar_ts = pd.Timestamp(bar.get('time', time))

        # Reset daily counter
        day = bar_ts.date()
        if day != self._current_day:
            self._trades_today = 0
            self._current_day = day

        # Skip if already in a trade (matching backtest: `if in_trade: continue`)
        if open_positions:
            return []

        # Check max trades per day
        if self._trades_today >= params.get('max_trades_per_day', 30):
            return []

        # Check intraday window
        t = bar_ts.time()
        if not (params['intraday_start'] <= t <= params['intraday_end']):
            return []

        # Look up bar index
        idx = self._bar_index.get(time)
        if idx is None or idx < 60:
            return []

        # Get precomputed features
        cp5 = self._cp5[idx]
        vwap_dist = self._vwap_dist[idx]
        daily_cp = self._daily_cp[idx]
        h1_cp = self._h1_cp[idx]
        h4_cp = self._h4_cp[idx]
        vol_ratio = self._vol_ratio[idx]
        vwap_slope = self._vwap_slope[idx]
        gap_pct = self._gap_pct[idx]
        spread_pct = self._spread_pct[idx]
        daily_slope = self._daily_slope[idx]
        h1_slope = self._h1_slope[idx]
        h4_slope = self._h4_slope[idx]

        # Run signal function
        try:
            from v15.trading.intraday_signals import sig_union_enhanced
        except ImportError:
            return []

        result = sig_union_enhanced(
            cp5=cp5, vwap_dist=vwap_dist,
            daily_cp=daily_cp, h1_cp=h1_cp, h4_cp=h4_cp,
            vol_ratio=vol_ratio, vwap_slope=vwap_slope,
            gap_pct=gap_pct, daily_slope=daily_slope,
            h1_slope=h1_slope, h4_slope=h4_slope,
            spread_pct=spread_pct,
            params=sig_params or None,
        )

        if result is None:
            return []

        direction_str, confidence, stop_pct, tp_pct = result
        self._trades_today += 1

        return [Signal(
            algo_id=self.config.algo_id,
            direction='long',  # Intraday is long-only
            price=bar['close'],  # Will be overridden to next bar's open by engine
            confidence=confidence,
            stop_pct=stop_pct,
            tp_pct=tp_pct,
            signal_type='intraday',
        )]

    def check_exits(self, time: pd.Timestamp, bar: dict,
                    open_positions: list) -> List[ExitSignal]:
        """Check trailing stop, stop loss, and take profit."""
        exits = []
        params = self.config.params
        trail_base = params.get('trail_base', 0.006)
        trail_power = params.get('trail_power', 6)
        trail_floor = params.get('trail_floor', 0.002)

        for pos in open_positions:
            high = bar['high']
            low = bar['low']
            close = bar['close']

            if pos.direction == 'long':
                # Update best price
                best = max(pos.best_price, high)

                # Trailing stop — ratchets from bar 1 (matching original backtest)
                trail_pct = trail_base * (1.0 - pos.confidence) ** trail_power
                if trail_floor > 0:
                    trail_pct = max(trail_floor, trail_pct)
                trail_stop = best * (1.0 - trail_pct)

                # Effective stop: max of initial stop and trail (always, not just in profit)
                effective_stop = max(pos.stop_price, trail_stop)

                # Check stop hit
                if low <= effective_stop:
                    exit_price = effective_stop  # Fill at stop level (matching original)
                    reason = 'trail' if effective_stop > pos.stop_price else 'stop'
                    exits.append(ExitSignal(
                        pos_id=pos.pos_id,
                        price=exit_price,
                        reason=reason,
                    ))
                    continue

                # Check take profit
                if high >= pos.tp_price:
                    exits.append(ExitSignal(
                        pos_id=pos.pos_id,
                        price=pos.tp_price,
                        reason='tp',
                    ))
                    continue

                # Timeout: 78 5-min bars (matching original backtest)
                # hold_bars counts 5-min bars since exit_check_tf='5min'
                if pos.hold_bars >= 78:
                    exits.append(ExitSignal(
                        pos_id=pos.pos_id, price=close, reason='timeout'))
                    continue

                # EOD: close at end of active window
                bar_time = pd.Timestamp(bar.get('time', time)).time()
                intraday_end = params.get('intraday_end', dt.time(15, 25))
                if bar_time >= intraday_end:
                    exits.append(ExitSignal(
                        pos_id=pos.pos_id, price=close, reason='eod'))
                    continue

        return exits

    def on_fill(self, trade):
        """Track win/loss streaks."""
        pass
