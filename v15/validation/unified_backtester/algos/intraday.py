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
import logging
import time as _time_mod
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..algo_base import AlgoBase, AlgoConfig, Signal, ExitSignal, CostModel
from ..data_provider import DataProvider
from ..portfolio import Position

logger = logging.getLogger(__name__)


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
    """Compute channel position using linear regression (matching backtest).

    Adapts window down to fit available data (min 10 bars for meaningful
    regression). Matches detect_channel() behavior in channel.py.
    """
    n = len(close_arr)
    if n < 10:
        return np.full(n, np.nan)
    close = close_arr.astype(np.float64)
    w = min(window, n)
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
    """Compute channel slope (matching backtest).

    Adapts window down to fit available data (min 10 bars).
    """
    n = len(close_arr)
    if n < 10:
        return np.full(n, np.nan)
    close = close_arr.astype(np.float64)
    w = min(window, n)
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

        if data is not None and not getattr(data, 'is_live', False):
            # Backtest mode: precompute all features for speed
            print("  Precomputing intraday features...")
            t0 = _time_mod.time()
            self._precompute_features()
            print(f"  Done in {_time_mod.time() - t0:.1f}s")
        else:
            # Live mode: compute features incrementally in on_bar()
            logger.info("Intraday algo: live mode, features computed per-bar")
            self._bar_index = {}
            self._live_mode = True

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

        # RSI(14) on 5-min closes + 5-bar slope
        delta = pd.Series(c).diff()
        gain = delta.clip(lower=0).rolling(14).mean().values
        loss = (-delta.clip(upper=0)).rolling(14).mean().values
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / np.where(loss == 0, np.nan, loss)
        rsi_arr = 100.0 - (100.0 / (1.0 + rs))
        self._rsi_slope = np.full(n, np.nan)
        for i in range(18, n):  # 14 warmup + 5 slope
            seg = rsi_arr[i - 4:i + 1]
            if not np.any(np.isnan(seg)):
                x = np.arange(5, dtype=np.float64)
                mx = x.mean()
                my = seg.mean()
                self._rsi_slope[i] = np.sum((x - mx) * (seg - my)) / np.sum((x - mx) ** 2)

        # Bullish 1-min candle count (last 5 1-min bars at each 5-min boundary)
        df1m = self.data._tf_data['1min']
        df1m_c = df1m['close'].values
        df1m_o = df1m['open'].values
        self._bullish_1m = np.full(n, np.nan)
        for i, ts in enumerate(f5m.index):
            pos = df1m.index.searchsorted(ts, side='right')
            start = max(0, pos - 5)
            if pos > start:
                seg_c = df1m_c[start:pos]
                seg_o = df1m_o[start:pos]
                self._bullish_1m[i] = float(np.sum(seg_c > seg_o))

        # Higher TF features (precomputed and mapped to 5-min bars)
        # Resample from 1-min data
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

        # Intraday TFs: map using completion-aware lookup
        # A 1h bar starting at 10:00 only becomes visible after 10:59 (its last 1-min bar)
        configs = [
            (tfs['1h'], {'1h_cp': 'chan_pos', '1h_slope': 'chan_slope'}),
            (tfs['4h'], {'4h_cp': 'chan_pos', '4h_slope': 'chan_slope'}),
        ]
        df1m_idx = df1m.index
        for tf_feat, col_map in configs:
            ti = tf_feat.index
            # Compute bar completion: last 1-min bar before next TF bar start
            completion = np.empty(len(ti), dtype='datetime64[ns]')
            for k in range(len(ti)):
                if k + 1 < len(ti):
                    pos = df1m_idx.searchsorted(ti[k + 1], side='left') - 1
                    completion[k] = df1m_idx[pos] if pos >= 0 else ti[k]
                else:
                    day = ti[k].date()
                    day_mask = df1m_idx.date == day
                    if day_mask.any():
                        completion[k] = df1m_idx[np.flatnonzero(day_mask)[-1]]
                    else:
                        completion[k] = ti[k]

            ca = {k: tf_feat[v].values for k, v in col_map.items()}
            j = -1
            for i in range(n):
                t = f5m.index[i]
                # Advance j to latest bar whose completion <= current 5-min start
                while j + 1 < len(ti) and completion[j + 1] <= t:
                    j += 1
                if j >= 0:
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

    def _compute_live_features(self, time: pd.Timestamp) -> Optional[dict]:
        """Compute all features from live bar data for current 5-min bar.

        Returns dict of feature values, or None if insufficient data.
        """
        wins = {'5m': 60, '1h': 24, '4h': 20, 'daily': 40}

        # 5-min bars up to current time
        f5m = self.data.get_bars('5min', time)
        if len(f5m) < 60:
            return None

        c = f5m['close'].values.astype(np.float64)
        h = f5m['high'].values.astype(np.float64)
        l = f5m['low'].values.astype(np.float64)
        o = f5m['open'].values.astype(np.float64)
        v = f5m['volume'].values.astype(np.float64)
        dates = np.array([t.date() for t in f5m.index])

        # 5-min channel position (last value)
        cp5_arr = _channel_position(c, wins['5m'])
        cp5 = cp5_arr[-1]

        # VWAP features (last values)
        _, vwap_dist_arr = _compute_vwap(o, h, l, c, v, dates)
        vwap_dist = vwap_dist_arr[-1]
        vol_ratio_arr = _compute_volume_ratio(v)
        vol_ratio = vol_ratio_arr[-1]
        vwap_slope_arr = _compute_vwap_slope(vwap_dist_arr)
        vwap_slope = vwap_slope_arr[-1]
        spread_pct_arr = _compute_spread_pct(h, l, c)
        spread_pct = spread_pct_arr[-1]
        gap_pct_arr = _compute_gap_pct(c, dates)
        gap_pct = gap_pct_arr[-1]

        # RSI(14) slope (last value)
        delta = pd.Series(c).diff()
        gain = delta.clip(lower=0).rolling(14).mean().values
        loss = (-delta.clip(upper=0)).rolling(14).mean().values
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / np.where(loss == 0, np.nan, loss)
        rsi_arr = 100.0 - (100.0 / (1.0 + rs))
        rsi_slope = np.nan
        n = len(c)
        if n >= 19:  # 14 warmup + 5 slope
            seg = rsi_arr[-5:]
            if not np.any(np.isnan(seg)):
                x = np.arange(5, dtype=np.float64)
                mx = x.mean()
                my = seg.mean()
                rsi_slope = float(np.sum((x - mx) * (seg - my)) / np.sum((x - mx) ** 2))

        # Bullish 1-min candle count (last 5 1-min bars)
        df1m = self.data.get_bars('1min', time)
        bullish_1m = np.nan
        if len(df1m) >= 5:
            last5 = df1m.iloc[-5:]
            bullish_1m = float((last5['close'] > last5['open']).sum())

        # Daily channel (prior day — no lookahead)
        daily = self.data.get_bars('daily', time)
        daily_cp = np.nan
        daily_slope = np.nan
        if len(daily) >= 2:
            # Use all days except today (prior day's value)
            today = time.date()
            prior = daily[daily.index.date < today] if hasattr(daily.index, 'date') else daily.iloc[:-1]
            if len(prior) >= 10:
                dc = prior['close'].values.astype(np.float64)
                cp_arr = _channel_position(dc, wins['daily'])
                sl_arr = _channel_slope(dc, wins['daily'])
                daily_cp = cp_arr[-1]
                daily_slope = sl_arr[-1]

        # 1h channel (last completed bar)
        h1 = self.data.get_bars('1h', time)
        h1_cp = np.nan
        h1_slope = np.nan
        if len(h1) >= 10:
            hc = h1['close'].values.astype(np.float64)
            cp_arr = _channel_position(hc, wins['1h'])
            sl_arr = _channel_slope(hc, wins['1h'])
            h1_cp = cp_arr[-1]
            h1_slope = sl_arr[-1]

        # 4h channel (last completed bar)
        h4 = self.data.get_bars('4h', time)
        h4_cp = np.nan
        h4_slope = np.nan
        if len(h4) >= 10:
            hc = h4['close'].values.astype(np.float64)
            cp_arr = _channel_position(hc, wins['4h'])
            sl_arr = _channel_slope(hc, wins['4h'])
            h4_cp = cp_arr[-1]
            h4_slope = sl_arr[-1]

        return {
            'cp5': cp5, 'vwap_dist': vwap_dist, 'daily_cp': daily_cp,
            'h1_cp': h1_cp, 'h4_cp': h4_cp, 'vol_ratio': vol_ratio,
            'vwap_slope': vwap_slope, 'gap_pct': gap_pct,
            'spread_pct': spread_pct, 'daily_slope': daily_slope,
            'h1_slope': h1_slope, 'h4_slope': h4_slope,
            'rsi_slope': rsi_slope, 'bullish_1m': bullish_1m,
        }

    def on_bar(self, time: pd.Timestamp, bar: dict,
               open_positions: list,
               context=None) -> List[Signal]:
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
        if not (params.get('intraday_start', dt.time(9, 30)) <= t <= params.get('intraday_end', dt.time(15, 25))):
            return []

        # Get features — live or precomputed
        if getattr(self, '_live_mode', False):
            feats = self._compute_live_features(time)
            if feats is None:
                return []
            cp5 = feats['cp5']
            vwap_dist = feats['vwap_dist']
            daily_cp = feats['daily_cp']
            h1_cp = feats['h1_cp']
            h4_cp = feats['h4_cp']
            vol_ratio = feats['vol_ratio']
            vwap_slope = feats['vwap_slope']
            gap_pct = feats['gap_pct']
            spread_pct = feats['spread_pct']
            daily_slope = feats['daily_slope']
            h1_slope = feats['h1_slope']
            h4_slope = feats['h4_slope']
            rsi_slope = feats['rsi_slope']
            bullish_1m = feats['bullish_1m']
        else:
            # Backtest: look up precomputed index
            idx = self._bar_index.get(time)
            if idx is None or idx < 60:
                return []
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
            rsi_slope = self._rsi_slope[idx]
            bullish_1m = self._bullish_1m[idx]

        # Run signal function
        try:
            from v15.trading.intraday_signals import sig_union_enhanced
        except ImportError as e:
            logger.error("Intraday: ImportError for sig_union_enhanced: %s", e)
            return []

        result = sig_union_enhanced(
            cp5=cp5, vwap_dist=vwap_dist,
            daily_cp=daily_cp, h1_cp=h1_cp, h4_cp=h4_cp,
            vol_ratio=vol_ratio, vwap_slope=vwap_slope,
            bullish_1m=bullish_1m,
            gap_pct=gap_pct, rsi_slope=rsi_slope,
            daily_slope=daily_slope,
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
        trail_floor = params.get('trail_floor', 0.0)

        for pos in open_positions:
            high = bar['high']
            low = bar['low']
            close = bar['close']

            if pos.direction == 'long':
                # Causal: use best_price from PRIOR bars only (engine updates after exits)
                best = pos.best_price

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

    def get_effective_stop(self, position) -> Optional[float]:
        """Return current effective stop for broker-side sync."""
        params = self.config.params
        trail_base = params.get('trail_base', 0.006)
        trail_power = params.get('trail_power', 6)
        trail_floor = params.get('trail_floor', 0.0)

        trail_pct = trail_base * (1.0 - position.confidence) ** trail_power
        if trail_floor > 0:
            trail_pct = max(trail_floor, trail_pct)

        if position.direction == 'long':
            trail_stop = position.best_price * (1.0 - trail_pct)
            return max(position.stop_price, trail_stop)
        else:
            trail_stop = position.best_price * (1.0 + trail_pct)
            return min(position.stop_price, trail_stop)

    def on_fill(self, trade):
        """Track win/loss streaks."""
        pass
