#!/usr/bin/env python3
"""
c11 Swing Backtest — multi-day/week TSLA trading via channel + SPY/RSI patterns.

Three modes:
  daily  — S01-S40 on daily bars, 10yr history (2015-2025). IS=2015-2024, OOS=2025.
  weekly — W01-W10 on weekly bars, 10yr history. Wider stops, multi-week holds.
  hourly — H01-H10 on 1h bars, ~2yr yfinance data (2023-2025). IS=2023-2024, OOS=2025.

Entry: next bar open after signal fires. No look-ahead bias.
Sizing: fixed $1M per trade. Costs: 0.05% slippage + 0.01% commission per side.

Usage:
    python3 -m v15.validation.swing_backtest                          # daily IS
    python3 -m v15.validation.swing_backtest --end-year 2025          # daily full
    python3 -m v15.validation.swing_backtest --mode weekly            # weekly IS
    python3 -m v15.validation.swing_backtest --mode weekly --end-year 2025
    python3 -m v15.validation.swing_backtest --mode hourly            # 1h IS 2023-2024
    python3 -m v15.validation.swing_backtest --mode hourly --end-year 2025
    python3 -m v15.validation.swing_backtest --sweep S32              # param sweep
    python3 -m v15.validation.swing_backtest --detail S32_union       # trade detail
"""

import argparse
import sys
import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v15.data.native_tf import fetch_native_tf
from v15.core.channel import detect_channel

# ── Config ────────────────────────────────────────────────────────────────────
# Overridable via --trade-usd CLI arg. Default $1M matches historical runs.
# Use --trade-usd 10000 to match c10/c9 sizing ($10K/trade, $100K equity).
MAX_TRADE_USD = 1_000_000
SLIPPAGE_PCT  = 0.0005    # 0.05% per side
COMM_PCT      = 0.0001    # 0.01% per side
COST_PER_SIDE = SLIPPAGE_PCT + COMM_PCT


# ── Data types ────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    signal:      str
    direction:   int          # +1 long, -1 short (all long for now)
    entry_date:  pd.Timestamp
    entry_price: float
    exit_date:   pd.Timestamp
    exit_price:  float
    exit_reason: str          # 'signal', 'stop', 'timeout'
    hold_days:   int
    pnl_usd:     float
    pnl_pct:     float


@dataclass
class BacktestResult:
    signal:     str
    trades:     List[Trade] = field(default_factory=list)

    @property
    def n_trades(self):
        return len(self.trades)

    @property
    def win_rate(self):
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl_usd > 0) / len(self.trades)

    @property
    def total_pnl(self):
        return sum(t.pnl_usd for t in self.trades)

    @property
    def avg_pnl(self):
        return self.total_pnl / max(self.n_trades, 1)

    @property
    def profit_factor(self):
        wins   = sum(t.pnl_usd for t in self.trades if t.pnl_usd > 0)
        losses = abs(sum(t.pnl_usd for t in self.trades if t.pnl_usd < 0))
        return wins / max(losses, 0.01)

    @property
    def avg_hold_days(self):
        return np.mean([t.hold_days for t in self.trades]) if self.trades else 0.0

    @property
    def max_loss(self):
        if not self.trades:
            return 0.0
        return min(t.pnl_usd for t in self.trades)

    def by_year(self) -> Dict[int, float]:
        years: Dict[int, float] = {}
        for t in self.trades:
            y = t.entry_date.year
            years[y] = years.get(y, 0.0) + t.pnl_usd
        return dict(sorted(years.items()))

    def profitable_years(self) -> int:
        return sum(1 for v in self.by_year().values() if v > 0)

    def exit_breakdown(self) -> Dict[str, int]:
        d: Dict[str, int] = {}
        for t in self.trades:
            d[t.exit_reason] = d.get(t.exit_reason, 0) + 1
        return d


# ── New feature helpers (Phase 7: unorthodox signals) ─────────────────────────
import math as _math
import calendar as _calendar

def _hurst(prices: np.ndarray) -> float:
    """Hurst exponent via R/S analysis. H>0.5 trending, H<0.5 mean-reverting.
    Uses geometric lag spacing to get sufficient data points for regression.
    Requires ~60+ prices for reliable output; returns 0.5 if insufficient."""
    n = len(prices)
    if n < 20:
        return 0.5
    lr = np.diff(np.log(np.maximum(prices, 1e-10)))
    rs_vals, lag_vals = [], []
    # Geometrically spaced lags from 4 → n//2 (gives ~10 points for n=60)
    lag = 4
    max_lag = max(8, n // 2)
    while lag <= max_lag:
        rs_lag = []
        for start in range(0, len(lr) - lag + 1, lag):
            sub = lr[start:start + lag]
            if len(sub) < 3:
                continue
            mu = np.mean(sub)
            dev = np.cumsum(sub - mu)
            R = float(np.max(dev) - np.min(dev))
            S = float(np.std(sub, ddof=1))
            if S > 1e-12:
                rs_lag.append(R / S)
        if len(rs_lag) >= 2:
            rs_vals.append(_math.log(float(np.mean(rs_lag))))
            lag_vals.append(_math.log(lag))
        lag = max(lag + 1, int(lag * 1.4))
    if len(rs_vals) < 4:
        return 0.5
    return float(np.clip(np.polyfit(lag_vals, rs_vals, 1)[0], 0.0, 1.0))


def _perm_entropy(prices: np.ndarray, order: int = 3) -> float:
    """Permutation entropy (Bandt & Pompe). Low=ordered/predictable, High=chaotic."""
    n = len(prices)
    if n < order + 2:
        return 1.0
    counts: dict = {}
    for i in range(n - order + 1):
        pat = tuple(int(x) for x in np.argsort(prices[i:i + order]))
        counts[pat] = counts.get(pat, 0) + 1
    total = sum(counts.values())
    entropy = -sum((c / total) * _math.log2(c / total) for c in counts.values() if c)
    max_e = _math.log2(_math.factorial(order))
    return float(entropy / max_e) if max_e > 0 else entropy


def _efficiency_ratio(prices: np.ndarray) -> float:
    """Kaufman ER: 1.0=linear trend, 0.0=choppy. Measures directional efficiency."""
    if len(prices) < 3:
        return 0.5
    net  = abs(float(prices[-1]) - float(prices[0]))
    path = float(np.sum(np.abs(np.diff(prices))))
    return net / path if path > 1e-10 else 0.0


_OPEX_DATES_CACHE = None
def _get_opex_dates():
    global _OPEX_DATES_CACHE
    if _OPEX_DATES_CACHE is not None:
        return _OPEX_DATES_CACHE
    dates = []
    for year in range(2014, 2027):
        for month in range(1, 13):
            c = _calendar.monthcalendar(year, month)
            fri = [w[4] for w in c if w[4] != 0]
            dates.append(pd.Timestamp(year, month, fri[2]))
    _OPEX_DATES_CACHE = dates
    return dates

def _opex_proximity(dt):
    """(days_to_next_opex, days_from_last_opex)."""
    opex = _get_opex_dates()
    d = pd.Timestamp(dt).normalize()
    future = [x for x in opex if x >= d]
    past   = [x for x in opex if x <= d]
    to_next   = int((future[0] - d).days) if future else 30
    from_last = int((d - past[-1]).days)  if past   else 30
    return to_next, from_last


_MOON_CACHE: dict = {}
def _moon_phase(dt) -> float:
    """Moon illumination percentage (0=new, 100=full). Uses ephem if available."""
    key = str(pd.Timestamp(dt).date())
    if key in _MOON_CACHE:
        return _MOON_CACHE[key]
    try:
        import ephem
        m = ephem.Moon(key)
        val = float(m.phase)
    except Exception:
        val = 50.0
    _MOON_CACHE[key] = val
    return val


# ── Helpers ───────────────────────────────────────────────────────────────────
def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI (ewm-based)."""
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=True).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=True).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _channel_at(df_slice: pd.DataFrame) -> Optional[object]:
    """Detect channel on a bar slice, return None on failure."""
    if len(df_slice) < 10:
        return None
    try:
        ch = detect_channel(df_slice)
        return ch if (ch and ch.valid) else None
    except Exception:
        return None


def _near_lower(price: float, ch, frac: float = 0.25) -> bool:
    """True if price is within bottom `frac` fraction of channel width."""
    if ch is None:
        return False
    lower = ch.lower_line[-1]
    upper = ch.upper_line[-1]
    w = upper - lower
    if w <= 0:
        return False
    return (price - lower) / w < frac


def _near_upper(price: float, ch, frac: float = 0.25) -> bool:
    if ch is None:
        return False
    lower = ch.lower_line[-1]
    upper = ch.upper_line[-1]
    w = upper - lower
    if w <= 0:
        return False
    return (upper - price) / w < frac


def _normalize_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone info from index (convert to UTC first, then tz-naive date).
    Use for daily/weekly/monthly bars — normalises each bar to midnight UTC."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').normalize().tz_localize(None)
    return df


def _strip_tz_intraday(df: pd.DataFrame) -> pd.DataFrame:
    """Strip timezone from sub-daily (1h/5m) bars — preserves intraday time."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    return df


def _align_daily_to_hourly(vix_daily: pd.DataFrame,
                            hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill daily VIX close to match hourly bar timestamps.

    Both inputs must be tz-naive (vix_daily after _normalize_tz,
    hourly_index after _strip_tz_intraday).
    """
    # Get UTC midnight date for each hourly bar
    daily_dates = hourly_index.floor('D')
    # Forward-fill daily VIX onto the hourly grid
    aligned = vix_daily['close'].reindex(daily_dates, method='ffill').values
    return pd.DataFrame({
        'open':   aligned,
        'high':   aligned,
        'low':    aligned,
        'close':  aligned,
        'volume': 0.0,
    }, index=hourly_index)


def _resample_weekly(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV → weekly (week ending Friday)."""
    return daily_df.resample('W-FRI').agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }).dropna()


# ── Core backtester ───────────────────────────────────────────────────────────
def _record_trade(result: BacktestResult, signal: str, direction: int,
                  entry_date, entry_price: float,
                  exit_date, exit_price: float,
                  reason: str, hold_days: int,
                  trade_usd: float = None) -> float:
    """Record a closed trade. Returns pnl_usd. trade_usd defaults to MAX_TRADE_USD."""
    if trade_usd is None:
        trade_usd = MAX_TRADE_USD
    eff_entry = entry_price * (1 + COST_PER_SIDE * direction)
    eff_exit  = exit_price  * (1 - COST_PER_SIDE * direction)
    pnl_pct   = direction * (eff_exit - eff_entry) / eff_entry
    pnl_usd   = pnl_pct * trade_usd
    result.trades.append(Trade(
        signal=signal, direction=direction,
        entry_date=entry_date, entry_price=entry_price,
        exit_date=exit_date, exit_price=exit_price,
        exit_reason=reason, hold_days=hold_days,
        pnl_usd=pnl_usd, pnl_pct=pnl_pct,
    ))
    return pnl_usd


def run_swing_backtest(
    tsla:          pd.DataFrame,
    spy:           pd.DataFrame,
    vix:           pd.DataFrame,
    tsla_weekly:   pd.DataFrame,
    spy_weekly:    pd.DataFrame,
    signal_fn:     Callable,
    signal_name:   str,
    max_hold_days: int   = 10,
    stop_pct:      float = 0.05,
    channel_window: int  = 50,
    warmup_bars:   int   = 70,
    start_year:    int   = 2015,
    end_year:      int   = 2024,
    compound:      bool  = False,
    trail_pct:     float = 0.0,    # > 0 enables trailing stop (e.g. 0.03 = 3%)
    persist_bars:  int   = 1,      # signal must fire N consecutive bars before entry
) -> BacktestResult:
    """Run swing backtest.
    compound:     position scales with equity (c10/c9 model).
    trail_pct:    trailing stop from highest close since entry (0 = disabled, use fixed stop_pct).
    persist_bars: require signal to fire N consecutive days before entering (default=1=immediate).
    """
    result = BacktestResult(signal=signal_name)

    # Compounding state
    initial_equity = MAX_TRADE_USD * 10
    equity         = initial_equity

    # Align to common daily dates
    common = tsla.index.intersection(spy.index).intersection(vix.index)
    tsla = tsla.loc[common]
    spy  = spy.loc[common]
    vix  = vix.loc[common]

    n = len(tsla)
    if n < warmup_bars + 5:
        return result

    # Precompute RSI on full data
    rsi_tsla = _rsi(tsla['close'], 14)
    rsi_spy  = _rsi(spy['close'], 14)
    tsla_full = tsla
    spy_full  = spy
    vix_full  = vix
    trade_start = warmup_bars

    in_trade       = False
    entry_price    = 0.0
    entry_bar      = 0
    entry_date     = None
    direction      = 0
    open_trade_usd = MAX_TRADE_USD   # size locked in at entry
    highest_close  = 0.0             # for trailing stop
    consec_signal  = 0               # persistence counter (bars signal has been active)

    for i in range(trade_start, n - 1):
        bar_year = tsla_full.index[i].year
        in_window = (bar_year >= start_year and bar_year <= end_year)

        # Always manage open trades regardless of year window
        # (a trade entered in Dec 2024 must be stopped/timed-out on Jan 2025 bars)
        if in_trade:
            price     = tsla_full['close'].iloc[i]
            hold_days = i - entry_bar
            highest_close = max(highest_close, price)

            # Stop loss — trailing or fixed
            if trail_pct > 0:
                stop_level = highest_close * (1 - trail_pct)
            else:
                stop_level = entry_price * (1 - stop_pct)

            if direction == 1 and price < stop_level:
                exit_price = tsla_full['open'].iloc[i + 1]
                pnl = _record_trade(result, signal_name, direction,
                                    entry_date, entry_price,
                                    tsla_full.index[i + 1], exit_price,
                                    'stop', hold_days, open_trade_usd)
                if compound:
                    equity = max(equity + pnl, initial_equity * 0.10)
                in_trade = False
                consec_signal = 0
                continue

            # Timeout
            if hold_days >= max_hold_days:
                exit_price = tsla_full['open'].iloc[i + 1]
                pnl = _record_trade(result, signal_name, direction,
                                    entry_date, entry_price,
                                    tsla_full.index[i + 1], exit_price,
                                    'timeout', hold_days, open_trade_usd)
                if compound:
                    equity = max(equity + pnl, initial_equity * 0.10)
                in_trade = False
                consec_signal = 0
                continue

            # Exit signal (must hold at least 2 days to avoid whipsaw)
            if hold_days >= 2:
                sig = signal_fn(i, tsla_full, spy_full, vix_full,
                                tsla_weekly, spy_weekly,
                                rsi_tsla, rsi_spy, channel_window)
                if sig != direction:
                    exit_price = tsla_full['open'].iloc[i + 1]
                    pnl = _record_trade(result, signal_name, direction,
                                        entry_date, entry_price,
                                        tsla_full.index[i + 1], exit_price,
                                        'signal', hold_days, open_trade_usd)
                    if compound:
                        equity = max(equity + pnl, initial_equity * 0.10)
                    in_trade = False
                    consec_signal = 0
                    continue

        elif in_window:
            # Only open new trades within the trading window
            sig = signal_fn(i, tsla_full, spy_full, vix_full,
                            tsla_weekly, spy_weekly,
                            rsi_tsla, rsi_spy, channel_window)
            if sig != 0:
                consec_signal += 1
            else:
                consec_signal = 0

            if sig != 0 and consec_signal >= persist_bars:
                # Lock in position size at entry (compounding: scale with equity growth)
                equity_scale   = (equity / initial_equity) if compound else 1.0
                open_trade_usd = MAX_TRADE_USD * equity_scale
                entry_price    = tsla_full['open'].iloc[i + 1]
                entry_date     = tsla_full.index[i + 1]
                entry_bar      = i + 1
                direction      = sig
                in_trade       = True
                highest_close  = entry_price
                consec_signal  = 0

    # Close any still-open trade at end of data
    if in_trade:
        exit_price = tsla_full['close'].iloc[-1]
        _record_trade(result, signal_name, direction,
                      entry_date, entry_price,
                      tsla_full.index[-1], exit_price,
                      'timeout', n - 1 - entry_bar, open_trade_usd)

    return result


# ── Signal functions ──────────────────────────────────────────────────────────
# Signature: fn(i, tsla, spy, vix, tsla_weekly, spy_weekly, rsi_tsla, rsi_spy, window) -> int

def sig_s01_daily_bounce(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S01: TSLA daily channel bounce — baseline."""
    if i < w:
        return 0
    ch = _channel_at(tsla.iloc[i - w:i])
    if ch is None:
        return 0
    return 1 if _near_lower(tsla['close'].iloc[i], ch, 0.25) else 0


def sig_s02_bounce_spy_filter(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S02: TSLA daily bounce + SPY above 20d MA."""
    if i < w:
        return 0
    ch = _channel_at(tsla.iloc[i - w:i])
    if ch is None:
        return 0
    if not _near_lower(tsla['close'].iloc[i], ch, 0.25):
        return 0
    spy_ma20 = spy['close'].iloc[max(0, i - 20):i].mean()
    if spy['close'].iloc[i] < spy_ma20:
        return 0
    return 1


def sig_s03_rsi_divergence(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S03: TSLA RSI < 35 + SPY RSI > 50."""
    if i < 20:
        return 0
    t_rsi = rt.iloc[i]
    s_rsi = rs.iloc[i]
    if pd.isna(t_rsi) or pd.isna(s_rsi):
        return 0
    return 1 if (t_rsi < 35 and s_rsi > 50) else 0


def sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S04: SPY at/near 20d high but TSLA lagging by >3%."""
    lookback = 20
    if i < lookback:
        return 0
    spy_now  = spy['close'].iloc[i]
    spy_high = spy['high'].iloc[i - lookback:i].max()
    tsla_now = tsla['close'].iloc[i]
    tsla_high = tsla['high'].iloc[i - lookback:i].max()
    spy_strong  = spy_now >= spy_high * 0.98   # SPY within 2% of 20d high
    tsla_lagging = tsla_now < tsla_high * 0.97 # TSLA at least 3% below its 20d high
    return 1 if (spy_strong and tsla_lagging) else 0


def sig_s05_high_quality_channel(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S05: Channel bounce only on high r² channels (>0.80)."""
    if i < w:
        return 0
    ch = _channel_at(tsla.iloc[i - w:i])
    if ch is None:
        return 0
    if ch.r_squared < 0.80:
        return 0
    return 1 if _near_lower(tsla['close'].iloc[i], ch, 0.25) else 0


def sig_s06_multi_window(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S06: Near lower band on 2+ of [30d, 50d, 70d] windows."""
    if i < 70:
        return 0
    near_count = 0
    for win in [30, 50, 70]:
        if i < win:
            continue
        ch = _channel_at(tsla.iloc[i - win:i])
        if ch is not None and _near_lower(tsla['close'].iloc[i], ch, 0.25):
            near_count += 1
    return 1 if near_count >= 2 else 0


def sig_s07_combined(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S07: Channel bottom + TSLA RSI < 45 + SPY RSI > 50 + SPY above 20d MA."""
    if i < w:
        return 0
    ch = _channel_at(tsla.iloc[i - w:i])
    if ch is None:
        return 0
    if not _near_lower(tsla['close'].iloc[i], ch, 0.25):
        return 0
    t_rsi = rt.iloc[i]
    s_rsi = rs.iloc[i]
    if pd.isna(t_rsi) or pd.isna(s_rsi):
        return 0
    if t_rsi > 45:
        return 0
    if s_rsi < 50:
        return 0
    spy_ma20 = spy['close'].iloc[max(0, i - 20):i].mean()
    if spy['close'].iloc[i] < spy_ma20:
        return 0
    return 1


def sig_s08_vix_spike_bounce(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S08: VIX spike (>25) + TSLA near lower channel band."""
    if i < w:
        return 0
    vix_now = vix['close'].iloc[i]
    if vix_now < 25:
        return 0
    ch = _channel_at(tsla.iloc[i - w:i])
    if ch is None:
        return 0
    return 1 if _near_lower(tsla['close'].iloc[i], ch, 0.30) else 0


def sig_s09_weekly_bounce(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S09: TSLA weekly channel bounce."""
    if tw is None or len(tw) < 20:
        return 0
    # Find the weekly bar that corresponds to current daily bar's date
    daily_date = tsla.index[i]
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 20:
        return 0
    ch = _channel_at(tw.iloc[wk_idx - 20:wk_idx])
    if ch is None:
        return 0
    return 1 if _near_lower(tw['close'].iloc[wk_idx], ch, 0.25) else 0


def sig_s10_spy_channel_break_tsla_entry(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S10: SPY breaks above its daily channel upper band → buy TSLA."""
    if i < w:
        return 0
    spy_ch = _channel_at(spy.iloc[i - w:i])
    if spy_ch is None:
        return 0
    # SPY breaking UP out of channel (above upper band)
    spy_price = spy['close'].iloc[i]
    spy_upper = spy_ch.upper_line[-1]
    if spy_price <= spy_upper:
        return 0
    # TSLA not already at its upper channel (avoid chasing)
    tsla_ch = _channel_at(tsla.iloc[i - w:i])
    if tsla_ch is not None and _near_upper(tsla['close'].iloc[i], tsla_ch, 0.25):
        return 0
    return 1


# ── Phase 2 signals ───────────────────────────────────────────────────────────

def sig_s11_spy_lag_rsi(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S11: SPY-TSLA lag + TSLA RSI < 55 (not already overbought)."""
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if pd.isna(t_rsi) or t_rsi > 55:
        return 0
    return 1


def sig_s12_weekly_bounce_spy_filter(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S12: Weekly bounce + SPY above 20d MA (trend confirmation)."""
    if sig_s09_weekly_bounce(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    spy_ma20 = spy['close'].iloc[max(0, i - 20):i].mean()
    if spy['close'].iloc[i] < spy_ma20:
        return 0
    return 1


def sig_s13_either_s04_or_s09(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S13: Fire on S04 OR S09 — union of two best signals."""
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s09_weekly_bounce(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return 0


def sig_s14_both_s04_and_s09(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S14: Both S04 AND S09 agree — intersection, higher confidence."""
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if sig_s09_weekly_bounce(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1


def sig_s15_spy_lag_high_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S15: SPY-TSLA lag when VIX is elevated (>18) — more volatile, bigger moves."""
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if vix['close'].iloc[i] < 18:
        return 0
    return 1


def sig_s16_spy_lag_low_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S16: SPY-TSLA lag when VIX is low (<18) — calm markets, steady catch-up."""
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if vix['close'].iloc[i] >= 18:
        return 0
    return 1


def sig_s17_spy_lag_tighter(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S17: SPY-TSLA lag tighter: SPY within 1% of 30d high, TSLA 5%+ behind."""
    lookback = 30
    if i < lookback:
        return 0
    spy_now   = spy['close'].iloc[i]
    spy_high  = spy['high'].iloc[i - lookback:i].max()
    tsla_now  = tsla['close'].iloc[i]
    tsla_high = tsla['high'].iloc[i - lookback:i].max()
    spy_strong   = spy_now >= spy_high * 0.99
    tsla_lagging = tsla_now < tsla_high * 0.95  # 5% lag threshold
    return 1 if (spy_strong and tsla_lagging) else 0


def sig_s18_weekly_rsi_divergence(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S18: Weekly channel bounce + TSLA daily RSI < 40."""
    if sig_s09_weekly_bounce(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if pd.isna(t_rsi) or t_rsi > 40:
        return 0
    return 1


def sig_s19_spy_lag_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S19: SPY lag + SPY itself bouncing (SPY RSI 50-65, not overbought)."""
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    s_rsi = rs.iloc[i]
    if pd.isna(s_rsi) or not (50 <= s_rsi <= 65):
        return 0
    return 1


def sig_s20_spy_channel_break_rsi(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S20: SPY channel break up + TSLA RSI oversold (<50)."""
    if sig_s10_spy_channel_break_tsla_entry(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if pd.isna(t_rsi) or t_rsi > 50:
        return 0
    return 1


# ── Experiment registry ───────────────────────────────────────────────────────
# Phase 1: (name, fn, max_hold_days, stop_pct, channel_window)
SIGNALS_P1: List[Tuple] = [
    ('S01_daily_bounce_baseline',    sig_s01_daily_bounce,            10, 0.05, 50),
    ('S02_bounce_spy_uptrend',       sig_s02_bounce_spy_filter,       10, 0.05, 50),
    ('S03_rsi_divergence',           sig_s03_rsi_divergence,          15, 0.05, 50),
    ('S04_spy_tsla_lag',             sig_s04_spy_lag,                  5, 0.03, 50),  # best params
    ('S05_high_quality_channel',     sig_s05_high_quality_channel,    10, 0.05, 50),
    ('S06_multi_window_bounce',      sig_s06_multi_window,            10, 0.05, 50),
    ('S07_combined_rsi_channel_spy', sig_s07_combined,                15, 0.05, 50),
    ('S08_vix_spike_bounce',         sig_s08_vix_spike_bounce,        10, 0.05, 50),
    ('S09_weekly_bounce',            sig_s09_weekly_bounce,            5, 0.05, 50),  # best params
    ('S10_spy_channel_break_entry',  sig_s10_spy_channel_break_tsla_entry, 7, 0.07, 50),  # best params
]

# Phase 2: combinations and refinements
SIGNALS_P2: List[Tuple] = [
    ('S11_spy_lag_rsi_filter',       sig_s11_spy_lag_rsi,              5, 0.03, 50),
    ('S12_weekly_bounce_spy_filter', sig_s12_weekly_bounce_spy_filter,  5, 0.05, 50),
    ('S13_union_s04_or_s09',         sig_s13_either_s04_or_s09,        5, 0.04, 50),
    ('S14_intersect_s04_and_s09',    sig_s14_both_s04_and_s09,         5, 0.03, 50),
    ('S15_spy_lag_high_vix',         sig_s15_spy_lag_high_vix,         5, 0.03, 50),
    ('S16_spy_lag_low_vix',          sig_s16_spy_lag_low_vix,          5, 0.03, 50),
    ('S17_spy_lag_tighter_5pct',     sig_s17_spy_lag_tighter,          5, 0.03, 50),
    ('S18_weekly_rsi_div',           sig_s18_weekly_rsi_divergence,    10, 0.05, 50),
    ('S19_spy_lag_momentum_rsi',     sig_s19_spy_lag_momentum,          5, 0.03, 50),
    ('S20_spy_break_rsi_filter',     sig_s20_spy_channel_break_rsi,    7, 0.07, 50),
]

# ── Phase 3 signals — VIX regime gating + trend guards ───────────────────────

def sig_s21_union_high_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S21: S13 (union) but only when VIX > 15 (exclude very calm markets)."""
    if vix['close'].iloc[i] <= 15:
        return 0
    return sig_s13_either_s04_or_s09(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s22_spy_lag_tsla_uptrend(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S22: SPY lag + TSLA above its 50d MA (avoid catching falling knife)."""
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    tsla_ma50 = tsla['close'].iloc[max(0, i - 50):i].mean()
    if tsla['close'].iloc[i] < tsla_ma50 * 0.97:
        return 0
    return 1


def sig_s23_spy_lag_spy_not_collapsing(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S23: SPY lag but SPY not in freefall (SPY not >10% below 50d high)."""
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    spy_50d_high = spy['high'].iloc[max(0, i - 50):i].max()
    if spy['close'].iloc[i] < spy_50d_high * 0.90:
        return 0
    return 1


def sig_s24_high_vix_lag_longer(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S24: Same entry as S15 (high VIX lag) — for sweep with longer hold."""
    return sig_s15_spy_lag_high_vix(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s25_weekly_no_bear(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S25: Weekly bounce but SPY not making new 20d lows (no bear crash)."""
    if sig_s09_weekly_bounce(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    spy_20d_low = spy['low'].iloc[max(0, i - 20):i].min()
    if spy['close'].iloc[i] <= spy_20d_low * 1.02:
        return 0
    return 1


def sig_s26_spy_lag_recovery(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S26: SPY lag + TSLA already showing 1-day recovery (close > prior close)."""
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 0
    if tsla['close'].iloc[i] <= tsla['close'].iloc[i - 1]:
        return 0
    return 1


def sig_s27_spy_lag_both_rsi(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S27: SPY lag + TSLA RSI 30-55 + SPY RSI 45-65 (sweet spot ranges)."""
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    s_rsi = rs.iloc[i]
    if pd.isna(t_rsi) or pd.isna(s_rsi):
        return 0
    if not (30 <= t_rsi <= 55):
        return 0
    if not (45 <= s_rsi <= 65):
        return 0
    return 1


def sig_s28_vix_extreme_bounce(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S28: VIX > 25 + TSLA RSI < 40 (extreme fear + oversold = strong bounce)."""
    if vix['close'].iloc[i] < 25:
        return 0
    t_rsi = rt.iloc[i]
    if pd.isna(t_rsi) or t_rsi > 40:
        return 0
    return 1


def sig_s29_spy_lag_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S29: SPY lag tiered by VIX — 20d high lag (any VIX), but VIX must be 15-35."""
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = vix['close'].iloc[i]
    if not (15 <= vix_now <= 35):
        return 0
    return 1


def sig_s30_union_spy_filter(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S30: Union of S04+S09 + SPY in medium-term uptrend (above 50d MA)."""
    if sig_s13_either_s04_or_s09(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    spy_ma50 = spy['close'].iloc[max(0, i - 50):i].mean()
    if spy['close'].iloc[i] < spy_ma50:
        return 0
    return 1


# Phase 3: VIX regime + trend guards
SIGNALS_P3: List[Tuple] = [
    ('S21_union_vix_gt15',           sig_s21_union_high_vix,           5, 0.03, 50),
    ('S22_spy_lag_tsla_above_50ma',  sig_s22_spy_lag_tsla_uptrend,     5, 0.03, 50),
    ('S23_spy_lag_spy_not_crashing', sig_s23_spy_lag_spy_not_collapsing, 5, 0.03, 50),
    ('S24_high_vix_lag_10d_hold',    sig_s24_high_vix_lag_longer,     10, 0.05, 50),
    ('S25_weekly_no_bear_guard',     sig_s25_weekly_no_bear,            5, 0.05, 50),
    ('S26_spy_lag_tsla_recovering',  sig_s26_spy_lag_recovery,          5, 0.03, 50),
    ('S27_spy_lag_dual_rsi_sweet',   sig_s27_spy_lag_both_rsi,          5, 0.03, 50),
    ('S28_vix_extreme_rsi_oversold', sig_s28_vix_extreme_bounce,       10, 0.05, 50),
    ('S29_spy_lag_vix_15_35',        sig_s29_spy_lag_vix_regime,        5, 0.03, 50),
    ('S30_union_spy_uptrend',        sig_s30_union_spy_filter,          5, 0.04, 50),
]

# ── Phase 4 signals — robust composite + crash guards ────────────────────────

def sig_s31_s29_crash_guard(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S31: S29 + SPY not in freefall (not >8% below 30d high)."""
    if sig_s29_spy_lag_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    spy_30d_high = spy['high'].iloc[max(0, i - 30):i].max()
    if spy['close'].iloc[i] < spy_30d_high * 0.92:
        return 0
    return 1


def sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S32: Union of S29 (VIX-gated lag) + S25 (weekly no-bear) — best IS+OOS combo."""
    if sig_s29_spy_lag_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s25_weekly_no_bear(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return 0


def sig_s33_s29_spy_rsi_healthy(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S33: S29 + SPY RSI > 45 (SPY not deeply oversold when we enter)."""
    if sig_s29_spy_lag_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    s_rsi = rs.iloc[i]
    if pd.isna(s_rsi) or s_rsi < 45:
        return 0
    return 1


def sig_s34_spy_lag_vix_18_28(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S34: Narrower VIX band 18-28 (core moderate-stress zone)."""
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = vix['close'].iloc[i]
    if not (18 <= vix_now <= 28):
        return 0
    return 1


def sig_s35_s21_vix_cap30(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S35: Union (S04 or S09) with VIX 15-30 only (cap at 30, not 35)."""
    if vix['close'].iloc[i] < 15 or vix['close'].iloc[i] > 30:
        return 0
    return sig_s13_either_s04_or_s09(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s36_s29_tsla_not_downtrend(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S36: S29 + TSLA not in severe downtrend (TSLA above 20d low * 0.90)."""
    if sig_s29_spy_lag_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    tsla_20d_low = tsla['low'].iloc[max(0, i - 20):i].min()
    if tsla['close'].iloc[i] < tsla_20d_low * 0.92:
        return 0
    return 1


def sig_s37_union_s29_s25_s15(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S37: Triple union — S29 + S25 + S15 (high VIX lag). All three regimes covered."""
    if sig_s29_spy_lag_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s25_weekly_no_bear(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s15_spy_lag_high_vix(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return 0


def sig_s38_intersect_s29_s09(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S38: Both S29 (lag) AND S09 (weekly bounce) agree — high confidence."""
    if sig_s29_spy_lag_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if sig_s09_weekly_bounce(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1


def sig_s39_s29_vix_any(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S39: SPY lag with any VIX but split by regime — VIX<20 use tight, VIX>20 normal."""
    # Just test SPY lag with VIX > 20 (rising volatility = better odds)
    if sig_s04_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if vix['close'].iloc[i] < 20:
        return 0
    return 1


def sig_s40_best_composite(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S40: Best composite — S32 (S29+S25 union) with VIX < 35 overall cap."""
    if vix['close'].iloc[i] >= 35:
        return 0
    return sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w)


# Phase 4
SIGNALS_P4: List[Tuple] = [
    ('S31_s29_crash_guard',          sig_s31_s29_crash_guard,           5, 0.03, 50),
    ('S32_union_s29_s25',            sig_s32_union_s29_s25,             5, 0.04, 50),
    ('S33_s29_spy_rsi_healthy',      sig_s33_s29_spy_rsi_healthy,       5, 0.03, 50),
    ('S34_spy_lag_vix_18_28',        sig_s34_spy_lag_vix_18_28,         5, 0.03, 50),
    ('S35_union_vix_15_30',          sig_s35_s21_vix_cap30,             5, 0.03, 50),
    ('S36_s29_tsla_not_downtrend',   sig_s36_s29_tsla_not_downtrend,    5, 0.03, 50),
    ('S37_triple_union_s29_s25_s15', sig_s37_union_s29_s25_s15,         5, 0.04, 50),
    ('S38_intersect_s29_s09',        sig_s38_intersect_s29_s09,         5, 0.03, 50),
    ('S39_spy_lag_vix_gt20',         sig_s39_s29_vix_any,               5, 0.03, 50),
    ('S40_best_composite',           sig_s40_best_composite,            5, 0.04, 50),
]

# ── Phase 5 (daily) — Bear market filter + refinements on S32 ─────────────────
# Starting from S41. Goal: fix 2022 (-$306K) without hurting 9 profitable years.
# Key idea: prolonged bears (2022) have SPY below 200d MA for months. VIX was
# mostly 20-30 so the VIX cap alone didn't protect.

def sig_s41_s32_spy_ma200(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S41: S32 + SPY above 200d MA — hard bear market filter."""
    if i < 200:
        return 0
    spy_ma200 = spy['close'].iloc[i - 200:i].mean()
    if spy['close'].iloc[i] < spy_ma200:
        return 0
    return sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s42_s32_spy_ma100(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S42: S32 + SPY above 100d MA — medium-term trend filter (less restrictive)."""
    if i < 100:
        return 0
    spy_ma100 = spy['close'].iloc[i - 100:i].mean()
    if spy['close'].iloc[i] < spy_ma100:
        return 0
    return sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s43_s32_no_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S43: S32 + TSLA not gapping down >2% on entry day (avoid gap-down traps)."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 1
    gap = (tsla['open'].iloc[i] / tsla['close'].iloc[i - 1]) - 1.0
    if gap < -0.02:
        return 0
    return 1


def sig_s44_s41_hold8(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S44: S41 (S32 + MA200) — same signal, tested with hold=8d in registry."""
    return sig_s41_s32_spy_ma200(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s45_s32_tsla_above_50d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S45: S32 + TSLA itself above 50d MA (not in its own downtrend)."""
    if i < 50:
        return 0
    tsla_ma50 = tsla['close'].iloc[i - 50:i].mean()
    if tsla['close'].iloc[i] < tsla_ma50:
        return 0
    return sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s46_s41_tsla_above_50d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S46: S41 (SPY MA200) + TSLA above 50d MA — dual trend filter."""
    if sig_s41_s32_spy_ma200(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 0
    tsla_ma50 = tsla['close'].iloc[i - 50:i].mean()
    if tsla['close'].iloc[i] < tsla_ma50:
        return 0
    return 1


def sig_s47_s32_spy_ma200_no_gap(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S47: S41 + no gap-down filter — both bear + gap protection."""
    if sig_s41_s32_spy_ma200(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 1
    gap = (tsla['open'].iloc[i] / tsla['close'].iloc[i - 1]) - 1.0
    if gap < -0.02:
        return 0
    return 1


# ── Phase 6 (daily) — Proactive Break Prediction (S50+) ───────────────────────
# These signals try to predict a channel BREAK *before* it happens, rather than
# waiting for the bounce at the boundary. The goal: enter earlier, ride the
# full breakout move.
#
# Key insight from c9 energy analysis:
#   High energy at boundary → FAILED break → violent reversal (the bounce system trades this)
#   Successful breaks tend to build slowly: compression, RSI divergence, SPY confirmation
#
# Signal structure: still LONG only. Predicts upside break of upper channel line.
# Entry fires 1-5 bars BEFORE the price reaches the channel boundary.
# Exit: same stop/timeout/signal logic as bounce signals.
# Wider hold window (8-10d) to allow time for the break to develop.

def _channel_width_ratio(tsla: pd.DataFrame, i: int, w: int, lookback: int = 10) -> float:
    """Ratio of current channel width to channel width `lookback` bars ago.
    < 1.0 = compression (narrowing), > 1.0 = expansion. Returns 1.0 on failure."""
    ch_now  = _channel_at(tsla.iloc[max(0, i - w):i])
    ch_past = _channel_at(tsla.iloc[max(0, i - w - lookback):i - lookback])
    if ch_now is None or ch_past is None:
        return 1.0
    lo_now  = ch_now.lower_line[-1];  hi_now  = ch_now.upper_line[-1]
    lo_past = ch_past.lower_line[-1]; hi_past = ch_past.upper_line[-1]
    w_now  = hi_now  - lo_now
    w_past = hi_past - lo_past
    if w_past <= 0:
        return 1.0
    return w_now / w_past


def _channel_pos(price: float, ch) -> float:
    """Price position within channel: 0.0 = lower line, 1.0 = upper line."""
    if ch is None:
        return 0.5
    lo = ch.lower_line[-1]; hi = ch.upper_line[-1]
    w  = hi - lo
    if w <= 0:
        return 0.5
    return (price - lo) / w


def sig_s50_channel_compression_breakout(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S50: Channel width compressing + price upper half + SPY trending up.
    Bets that compression (coiling) will resolve as upside break."""
    if i < w + 15:
        return 0
    ch = _channel_at(tsla.iloc[i - w:i])
    if ch is None:
        return 0
    price = tsla['close'].iloc[i]
    pos   = _channel_pos(price, ch)
    # Must be in upper half of channel (not already at boundary)
    if pos < 0.45 or pos > 0.90:
        return 0
    # Channel must be compressing: current width < 85% of width 10 bars ago
    compression = _channel_width_ratio(tsla, i, w, lookback=10)
    if compression >= 0.88:
        return 0
    # SPY must be in uptrend (above 20d MA) — directional confirmation
    spy_ma20 = spy['close'].iloc[i - 20:i].mean()
    if spy['close'].iloc[i] < spy_ma20:
        return 0
    # TSLA RSI should show positive momentum
    t_rsi = rt.iloc[i]
    if pd.isna(t_rsi) or t_rsi < 48:
        return 0
    return 1


def sig_s51_multi_touch_upper(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S51: TSLA has touched the upper 20% of channel 2+ times in last 8 bars
    without breaking out → pressure accumulating, break imminent."""
    if i < w + 10:
        return 0
    ch = _channel_at(tsla.iloc[i - w:i])
    if ch is None:
        return 0
    price = tsla['close'].iloc[i]
    pos   = _channel_pos(price, ch)
    # Current price in upper 45% (approaching but not at boundary)
    if pos < 0.40 or pos > 0.88:
        return 0
    # Count prior touches in last 8 bars (upper 20% of channel)
    lo = ch.lower_line[-1]; hi = ch.upper_line[-1]; width = hi - lo
    if width <= 0:
        return 0
    touches = 0
    for j in range(i - 8, i):
        if j < 0:
            continue
        p_j = tsla['close'].iloc[j]
        if (p_j - lo) / width > 0.80:
            touches += 1
    if touches < 2:
        return 0
    # SPY not in downtrend
    spy_ma20 = spy['close'].iloc[i - 20:i].mean()
    if spy['close'].iloc[i] < spy_ma20 * 0.97:
        return 0
    return 1


def sig_s52_rsi_divergence_breakout(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S52: TSLA RSI making new 10-bar high while price is mid-to-upper channel
    (RSI divergence = hidden upside strength, break likely)."""
    if i < w + 15:
        return 0
    ch = _channel_at(tsla.iloc[i - w:i])
    if ch is None:
        return 0
    price = tsla['close'].iloc[i]
    pos   = _channel_pos(price, ch)
    # Mid-to-upper channel (0.40 - 0.85), not yet broken
    if pos < 0.40 or pos > 0.88:
        return 0
    # RSI must be making a new 10-bar high (momentum accelerating)
    t_rsi = rt.iloc[i]
    if pd.isna(t_rsi):
        return 0
    rsi_window = rt.iloc[i - 10:i]
    if t_rsi <= rsi_window.max():
        return 0
    # RSI > 55 (genuinely strong, not just bouncing from oversold)
    if t_rsi < 55:
        return 0
    # SPY above 50d MA (bull regime context)
    if i < 50:
        return 0
    spy_ma50 = spy['close'].iloc[i - 50:i].mean()
    if spy['close'].iloc[i] < spy_ma50:
        return 0
    return 1


def sig_s53_spy_accel_tsla_lag_break(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S53: Proactive SPY lag — SPY accelerating (3-day return > 1.5%) while
    TSLA is lagging AND still mid-channel. Predicts TSLA breakout catch-up."""
    if i < w + 5:
        return 0
    # SPY accelerating: 3-day return > 1.5%
    spy_3d_ret = (spy['close'].iloc[i] / spy['close'].iloc[i - 3]) - 1.0
    if spy_3d_ret < 0.015:
        return 0
    # TSLA lagging: TSLA 3-day return < SPY 3-day return
    tsla_3d_ret = (tsla['close'].iloc[i] / tsla['close'].iloc[i - 3]) - 1.0
    if tsla_3d_ret >= spy_3d_ret * 0.6:   # TSLA must be meaningfully behind
        return 0
    # TSLA not at channel top yet (still room to run)
    ch = _channel_at(tsla.iloc[i - w:i])
    if ch is not None:
        pos = _channel_pos(tsla['close'].iloc[i], ch)
        if pos > 0.85:  # already near top, break already happened
            return 0
    # VIX not panicking
    if vix['close'].iloc[i] > 35:
        return 0
    return 1


def sig_s54_compression_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S54: Combine S50 compression + S53 SPY acceleration — dual confirmation."""
    if sig_s50_channel_compression_breakout(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # SPY made new 10d high recently (within last 3 bars)
    spy_10d_high = spy['close'].iloc[i - 10:i].max()
    recent_spy_high = spy['close'].iloc[i - 3:i].max()
    if recent_spy_high < spy_10d_high * 0.99:
        return 0
    # TSLA lagging SPY over 5 days
    if i < 5:
        return 0
    spy_5d  = (spy['close'].iloc[i]  / spy['close'].iloc[i - 5])  - 1.0
    tsla_5d = (tsla['close'].iloc[i] / tsla['close'].iloc[i - 5]) - 1.0
    if tsla_5d >= spy_5d:
        return 0
    return 1


def sig_s55_breakout_union(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S55: Union of S50 + S53 — either compression OR SPY acceleration qualifies."""
    if sig_s50_channel_compression_breakout(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s53_spy_accel_tsla_lag_break(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return 0


def sig_s56_s51_ma200(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S56: S51 multi-touch + SPY above 200d MA — prevents buying distribution tops in bears."""
    if i < 200:
        return 0
    spy_ma200 = spy['close'].iloc[i - 200:i].mean()
    if spy['close'].iloc[i] < spy_ma200:
        return 0
    return sig_s51_multi_touch_upper(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s57_s51_intersect_s32(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S57: S51 AND S32 both agree — multi-touch breakout setup during proven lag regime."""
    if sig_s51_multi_touch_upper(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1


def sig_s58_s50_ma200(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S58: S50 compression + SPY above 200d MA — filters out bear compression traps."""
    if i < 200:
        return 0
    spy_ma200 = spy['close'].iloc[i - 200:i].mean()
    if spy['close'].iloc[i] < spy_ma200:
        return 0
    return sig_s50_channel_compression_breakout(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s59_breakout_or_bounce(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S59: Union of S32 (best bounce) + S56 (S51+MA200 break) — two complementary edges."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s56_s51_ma200(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return 0


def sig_s60_s51_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S60: S51 multi-touch + VIX 15-35 (same regime filter as S32's core)."""
    vix_now = vix['close'].iloc[i]
    if not (15 <= vix_now <= 35):
        return 0
    return sig_s51_multi_touch_upper(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s61_s51_ma200_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S61: S51 + SPY MA200 + VIX 15-35 — all three filters stacked."""
    if i < 200:
        return 0
    spy_ma200 = spy['close'].iloc[i - 200:i].mean()
    if spy['close'].iloc[i] < spy_ma200:
        return 0
    vix_now = vix['close'].iloc[i]
    if not (15 <= vix_now <= 35):
        return 0
    return sig_s51_multi_touch_upper(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s64_dynamic_bear_guard(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S64: S32 + dynamic bear guard — require SPY > 100d MA only when VIX > 25.
    Keeps all calm-market S32 trades; adds uptrend filter during stress."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = vix['close'].iloc[i]
    if vix_now > 25:
        if i < 100:
            return 0
        spy_ma100 = spy['close'].iloc[i - 100:i].mean()
        if spy['close'].iloc[i] < spy_ma100:
            return 0
    return 1


def sig_s65_s32_bullish_candle(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S65: S32 + TSLA closes above its open on signal day (bullish candle).
    Requires intraday momentum confirmation before entry."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if tsla['close'].iloc[i] <= tsla['open'].iloc[i]:
        return 0
    return 1


def sig_s66_s32_volume_surge(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S66: S32 + TSLA volume > 1.2x 10d average (unusual participation = conviction)."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 0
    avg_vol = tsla['volume'].iloc[i - 10:i].mean()
    if avg_vol <= 0 or tsla['volume'].iloc[i] < avg_vol * 1.2:
        return 0
    return 1


def sig_s67_s32_rsi_exit(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S67: S32 entry + RSI-based exit override.
    Returns 0 (exit signal) when RSI > 65 (overbought) even if S32 hasn't flipped.
    This lets us ride momentum but exit at exhaustion rather than waiting for full reversal."""
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi > 65:
        return 0   # overbought — exit or don't enter
    return sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s68_s62_dynamic_bear(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S68: S62 (S32+S51/VIX union) + dynamic bear guard (SPY MA100 when VIX > 25).
    Best combination signal with stress-regime bear protection."""
    if sig_s62_s32_plus_s51_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = vix['close'].iloc[i]
    if vix_now > 25:
        if i < 100:
            return 0
        spy_ma100 = spy['close'].iloc[i - 100:i].mean()
        if spy['close'].iloc[i] < spy_ma100:
            return 0
    return 1


def sig_s63_s62_ma200(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S63: S62 (S32 + S51/VIX union) + SPY above 200d MA — cuts bear market noise."""
    if i < 200:
        return 0
    spy_ma200 = spy['close'].iloc[i - 200:i].mean()
    if spy['close'].iloc[i] < spy_ma200:
        return 0
    return sig_s62_s32_plus_s51_union(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s62_s32_plus_s51_union(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S62: S32 union S60 (S51+VIX15-35) — adds break-prediction signals only in good regime."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s60_s51_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return 0


# ── Phase 7D — Unorthodox signals (S69-S76) ───────────────────────────────────
# Ideas from outside OHLCV: Hurst regime, OPEX calendar, permutation entropy,
# moon phase, efficiency ratio, VIX term structure.

def sig_s69_s32_hurst(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S69: S32 entry ONLY when 60-bar Hurst > 0.52 (trending regime).
    Hurst > 0.5 means price is more persistent (trending) today.
    SPY-TSLA lag works better when the broader trend is persistent.
    Hypothesis: trending regime increases probability lag follows through."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 65:
        return 1  # not enough data, don't filter
    prices = tsla['close'].iloc[i-60:i+1].values.astype(float)
    h = _hurst(prices)
    return 1 if h > 0.52 else 0


def sig_s70_s32_anti_hurst(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S70: S32 entry ONLY when Hurst < 0.50 (mean-reverting regime).
    Counter-hypothesis: the SPY-TSLA lag is a MEAN-REVERSION signal —
    TSLA that's lagged behind SPY mean-reverts upward. This should work
    better when the general regime is mean-reverting (H < 0.5)."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 65:
        return 1
    prices = tsla['close'].iloc[i-60:i+1].values.astype(float)
    h = _hurst(prices)
    return 1 if h < 0.50 else 0


def sig_s71_s32_post_opex(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S71: S32 entry ONLY in days 1-5 after monthly OPEX (3rd Friday).
    After OPEX, the options gamma pin releases and TSLA moves freely.
    Historical: TSLA has larger autonomous moves in the week after OPEX.
    Hypothesis: SPY-TSLA lag signals more likely to play out post-pin."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    to_next, from_last = _opex_proximity(dt)
    # Post-OPEX window: 1-7 days after expiry
    return 1 if 1 <= from_last <= 7 else 0


def sig_s72_s32_opex_avoid(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S72: S32 with OPEX pin week AVOIDED (days 0-5 before OPEX).
    Before OPEX, max-pain gravitational field pins TSLA near key strikes.
    Momentum signals during pin week have poor follow-through.
    Avoids the 5 days leading up to expiry."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    to_next, from_last = _opex_proximity(dt)
    # Skip the 5 days before OPEX (pin week)
    return 0 if to_next <= 5 else 1


def sig_s73_s32_low_entropy(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S73: S32 entry ONLY when permutation entropy < 0.85 (ordered dynamics).
    Low PE = price moving in a more predictable, ordered way.
    From information theory: low entropy system has more signal content.
    Hypothesis: ordered dynamics → lag signal more likely to follow through."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 15:
        return 1
    prices = tsla['close'].iloc[i-14:i+1].values.astype(float)
    pe = _perm_entropy(prices, order=3)
    return 1 if pe < 0.85 else 0


def sig_s74_s32_high_er(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S74: S32 entry ONLY when efficiency ratio > 0.25 (directional momentum).
    High ER = price moving efficiently in one direction.
    Kaufman ER > 0.25 means more than 25% of the price path is directional.
    Hypothesis: directional momentum increases lag follow-through probability."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 12:
        return 1
    prices = tsla['close'].iloc[i-10:i+1].values.astype(float)
    er = _efficiency_ratio(prices)
    return 1 if er > 0.25 else 0


def sig_s75_s32_new_moon(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S75: S32 entry ONLY near new moon (illumination < 25%).
    Based on Dichev & Janes (2001): stocks return 1.4% more in the 15 days
    around new moon vs full moon. Psychological risk appetite cycles.
    Hypothesis: new moon = risk-on phase = TSLA lag more likely to recover."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    phase = _moon_phase(dt)
    return 1 if phase < 25 else 0


def sig_s76_s32_avoid_full_moon(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S76: S32 with full moon AVOIDED (illumination > 75%).
    Complement to S75: avoid the risk-off phase of the lunar cycle.
    Dichev & Janes find underperformance in full moon window."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    phase = _moon_phase(dt)
    return 0 if phase > 75 else 1


def sig_s77_s32_hurst_opex(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S77: S32 + Hurst regime + post-OPEX (combined best ideas).
    Uses BOTH the regime filter (H>0.50) AND post-OPEX timing.
    Hypothesis: when regime is trending AND OPEX pin just released,
    TSLA has maximum free-movement energy for the lag to follow through."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    to_next, from_last = _opex_proximity(dt)
    in_opex_pin = to_next <= 5  # within 5 days of OPEX = pinned
    if in_opex_pin:
        return 0
    if i >= 65:
        prices = tsla['close'].iloc[i-60:i+1].values.astype(float)
        h = _hurst(prices)
        if h < 0.48:  # clear mean-reverting regime — weaker signal
            return 0
    return 1


def sig_s78_s32_vix_structure(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S78: S32 entry ONLY when VIX is in contango (calm regime, VIX < 30d MA).
    VIX contango (spot < 30d average) = current vol below recent norm = calm.
    VIX backwardation (spot > 30d average) = acute stress = choppy signals.
    Hypothesis: calm VIX structure = cleaner trend environment for lag signals."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 35:
        return 1
    vix_now  = vix['close'].iloc[i]
    vix_ma30 = vix['close'].iloc[i-30:i].mean()
    vix_struct = vix_now / max(float(vix_ma30), 1.0)
    return 1 if vix_struct < 1.05 else 0  # slight backwardation OK, >1.05 = stress


def sig_s79_s32_avoid_august(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S79: S32 entry ONLY in non-August months.
    Calendar analysis (2015-2024): August = worst month for TSLA 5d returns (-2.812%, p=0.005**).
    Statistically significant underperformance — skip all lag entries in August.
    Hypothesis: TSLA institutional selling into Aug options roll causes structural weakness."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    return 0 if dt.month == 8 else 1


def sig_s80_s32_waning_moon(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S80: S32 entry ONLY during waning moon phase (illumination >= 75%).
    Calendar analysis (2015-2024): Waning phase = best 5d return (+1.413%, p=0.000**).
    Waning moon (nearly full → new) is the risk-on phase of the lunar cycle.
    Moon phase r=+0.066 with 5d returns (p=0.000**) — strongest calendar correlation found."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    phase = _moon_phase(dt)
    return 1 if phase >= 75 else 0


def sig_s81_s32_opex_window(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S81: S32 entry ONLY in the 8-14 day window before OPEX.
    Calendar analysis (2015-2024): 8-14d before OPEX = best 5d returns (+1.271%, p=0.000**).
    Hypothesis: in this window options dealers net long gamma → stabilize moves → lag follows through.
    The 0-3d pin zone and post-OPEX are weaker; 8-14d is the sweet spot."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    to_next, _ = _opex_proximity(dt)
    return 1 if 8 <= to_next <= 14 else 0


def sig_s82_s32_pres_year_filter(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S82: S32 entry SKIP presidential Year 3 (mid-term year 2 of cycle).
    Calendar analysis (2015-2024): Year 3 = -1.255% avg 5d return (p=0.002**).
    Year 3 examples: 2019, 2023, 2027 (two years before election year).
    Hypothesis: policy uncertainty peak + mid-term fatigue = structural TSLA headwinds."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    yr = dt.year
    election_yrs = [2016, 2020, 2024, 2028]
    for ey in election_yrs:
        if yr == ey - 1:  # Year 3 of that cycle = one year before election
            return 0
    return 1


def sig_s83_s32_calendar_combo(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S83: S32 + avoid August + avoid Year3 + OPEX 8-14d window (calendar trifecta).
    Combines three independent statistically significant calendar effects:
    - Not August (worst month p=0.005**): -2.812% vs baseline 0.622%
    - Not presidential Year 3 (p=0.002**): -1.255% vs baseline
    - In 8-14d pre-OPEX window (p=0.000**): +1.271% vs 0.387% at pin zone
    Fewer trades but higher-quality timing stacking three edges."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    if dt.month == 8:
        return 0
    yr = dt.year
    election_yrs = [2016, 2020, 2024, 2028]
    for ey in election_yrs:
        if yr == ey - 1:
            return 0
    to_next, _ = _opex_proximity(dt)
    return 1 if 8 <= to_next <= 14 else 0


SIGNALS_P7D: List[Tuple] = [
    # (name, fn, max_hold_days, stop_pct, channel_window)
    ('S69_s32_hurst_trending',   sig_s69_s32_hurst,         5, 0.04, 50),
    ('S70_s32_hurst_reverting',  sig_s70_s32_anti_hurst,    5, 0.04, 50),
    ('S71_s32_post_opex',        sig_s71_s32_post_opex,     5, 0.04, 50),
    ('S72_s32_opex_avoid',       sig_s72_s32_opex_avoid,    5, 0.04, 50),
    ('S73_s32_low_entropy',      sig_s73_s32_low_entropy,   5, 0.04, 50),
    ('S74_s32_high_er',          sig_s74_s32_high_er,       5, 0.04, 50),
    ('S75_s32_new_moon',         sig_s75_s32_new_moon,      5, 0.04, 50),
    ('S76_s32_avoid_full_moon',  sig_s76_s32_avoid_full_moon, 5, 0.04, 50),
    ('S77_s32_hurst_opex',       sig_s77_s32_hurst_opex,    5, 0.04, 50),
    ('S78_s32_vix_structure',    sig_s78_s32_vix_structure, 5, 0.04, 50),
    # Phase 7E — Calendar effects (statistically validated, 2015-2024)
    ('S79_s32_avoid_august',     sig_s79_s32_avoid_august,  5, 0.04, 50),
    ('S80_s32_waning_moon',      sig_s80_s32_waning_moon,   5, 0.04, 50),
    ('S81_s32_opex_window',      sig_s81_s32_opex_window,   5, 0.04, 50),
    ('S82_s32_pres_year_filter', sig_s82_s32_pres_year_filter, 5, 0.04, 50),
    ('S83_s32_calendar_combo',   sig_s83_s32_calendar_combo,   5, 0.04, 50),
]

def sig_s84_s32_time_compression(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S84: S32 + time compression filter (ATR contraction = coiling spring).
    When ATR_5 < 0.75 × ATR_20, TSLA is in a tight consolidation — the spring is wound.
    Wyckoff accumulation: price drifts flat/down on declining volume before explosive move.
    Hypothesis: compressed volatility before a lag signal = bigger snap-back energy."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 25:
        return 1
    closes = tsla['close'].iloc[i-20:i+1].values.astype(float)
    highs  = tsla['high'].iloc[i-20:i+1].values.astype(float)
    lows   = tsla['low'].iloc[i-20:i+1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1])))
    atr_5  = tr[-5:].mean()
    atr_20 = tr.mean()
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s85_s32_volume_dry_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S85: S32 + volume dry-up (3-day declining volume = distribution exhaustion).
    When volume has been declining for 3+ days while TSLA lags, sellers are exhausting.
    Classic accumulation pattern: volume contracts into weakness = smart money absorbing.
    Hypothesis: low-volume lag signal → sellers done → snap-back more forceful."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    vols = tsla['volume'].iloc[i-3:i+1].values.astype(float)
    # Check if volume has been declining for 3 consecutive days
    if vols[0] > vols[1] > vols[2] > vols[3]:
        return 1
    # Alternative: current volume < 70% of 10-day average
    vol_10d = tsla['volume'].iloc[i-10:i].mean()
    return 1 if float(vols[-1]) < 0.70 * float(vol_10d) else 0


def sig_s86_s32_atr_expansion(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S86: S32 + ATR expansion (ATR_3 > 1.3 × ATR_20 = momentum building).
    When short-term ATR is expanding relative to longer-term, a directional move has started.
    Hypothesis: when TSLA starts moving forcefully (↑ ATR) AND lags SPY, the lag closes faster.
    Complement to S84 (compression): this catches the EARLY phase of ATR expansion."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 25:
        return 1
    closes = tsla['close'].iloc[i-20:i+1].values.astype(float)
    highs  = tsla['high'].iloc[i-20:i+1].values.astype(float)
    lows   = tsla['low'].iloc[i-20:i+1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1])))
    atr_3  = tr[-3:].mean()
    atr_20 = tr.mean()
    return 1 if atr_3 > 1.30 * atr_20 else 0


def sig_s87_s32_dow_mon_tue(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S87: S32 entry ONLY on Monday or Tuesday.
    c9 regime analysis found Mon/Tue are stronger for TSLA bounces than Wed-Thu.
    Calendar: Mon avg=+0.262% (p=0.131), Tue avg=+0.243% (p=0.138) — directionally positive.
    Hypothesis: end-of-week forced selling (margin calls etc.) resolves over weekend → Mon entry."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    return 1 if dt.weekday() in (0, 1) else 0  # 0=Mon, 1=Tue


def sig_s88_s32_hurst_spy_ma200(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S88: S32 + Hurst trending regime + SPY above 200d MA (dual regime guard).
    S69 showed Hurst filter doesn't hurt (≈S32). S41 showed MA200 helps in bear markets.
    Stacking both: only take the lag signal when BOTH regime conditions are favorable.
    Hypothesis: trending TSLA (H>0.50) in bull market (SPY>200d) = highest snap-back prob."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # SPY must be above 200-day MA (bull market filter from S41)
    if i >= 200:
        spy_close = spy['close'].iloc[i]
        spy_ma200 = spy['close'].iloc[i-200:i].mean()
        if float(spy_close) < float(spy_ma200):
            return 0
    # TSLA Hurst must be >= 0.50 (not mean-reverting)
    if i >= 65:
        prices = tsla['close'].iloc[i-60:i+1].values.astype(float)
        h = _hurst(prices)
        if h < 0.48:
            return 0
    return 1


def sig_s89_s32_multi_score(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S89: S32 scored filter — fire when 3+ of 5 favorable conditions met.
    Conditions (each worth 1 point):
      1. Hurst > 0.50 (trending regime)
      2. Not August (avoid worst month)
      3. Moon waning (phase >= 50) — risk-on lunar phase
      4. SPY above 200d MA (bull market)
      5. VIX in contango (VIX < 1.1 × 30d VIX avg)
    Hypothesis: soft-filter stacking weak signals → higher quality entries than any single filter."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    score = 0
    dt = tsla.index[i]

    # 1. Hurst trending
    if i >= 65:
        prices = tsla['close'].iloc[i-60:i+1].values.astype(float)
        if _hurst(prices) >= 0.50:
            score += 1
    else:
        score += 1  # default to favorable if not enough data

    # 2. Not August
    if dt.month != 8:
        score += 1

    # 3. Waning or full moon (phase >= 50)
    phase = _moon_phase(dt)
    if phase >= 50:
        score += 1

    # 4. SPY above 200d MA
    if i >= 200:
        if float(spy['close'].iloc[i]) > float(spy['close'].iloc[i-200:i].mean()):
            score += 1
    else:
        score += 1

    # 5. VIX structure (not backwardation)
    if i >= 30:
        vix_now = float(vix['close'].iloc[i])
        vix_avg = float(vix['close'].iloc[i-30:i].mean())
        if vix_now < 1.10 * vix_avg:
            score += 1
    else:
        score += 1

    return 1 if score >= 3 else 0


def sig_s90_s32_compression_bounce(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S90: S32 + Wyckoff compression-then-bounce (volume dry-up AND time compression).
    Stacks S84 + S85: fire only when BOTH ATR is contracting AND volume is drying up.
    This is the classic 'coiling spring' pattern — tighter than either alone.
    Hypothesis: double confirmation of consolidation = maximum snap-back energy stored."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 25:
        return 1
    closes = tsla['close'].iloc[i-20:i+1].values.astype(float)
    highs  = tsla['high'].iloc[i-20:i+1].values.astype(float)
    lows   = tsla['low'].iloc[i-20:i+1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1])))
    atr_5  = tr[-5:].mean()
    atr_20 = tr.mean()
    atr_compressed = atr_5 < 0.80 * atr_20  # slightly looser than S84
    # Volume dry-up: current vol < 75% of 10d avg
    vol_10d = float(tsla['volume'].iloc[i-10:i].mean()) if i >= 10 else 1
    vol_now = float(tsla['volume'].iloc[i])
    vol_dry = vol_now < 0.75 * vol_10d
    return 1 if (atr_compressed and vol_dry) else 0


def sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S91: S32 + ATR extreme (EITHER compressed OR expanding) — avoid the muddy middle.
    S84 (compressed) and S86 (expanding) both have 60%+ WR and PF>2.0.
    The "middle" ATR zone (0.75-1.30× the 20d avg) is lower quality.
    Hypothesis: S32 is highest quality at volatility extremes, not in the ambiguous middle."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 25:
        return 1
    closes = tsla['close'].iloc[i-20:i+1].values.astype(float)
    highs  = tsla['high'].iloc[i-20:i+1].values.astype(float)
    lows   = tsla['low'].iloc[i-20:i+1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1])))
    atr_5  = tr[-5:].mean()
    atr_3  = tr[-3:].mean()
    atr_20 = tr.mean()
    compressed = atr_5 < 0.75 * atr_20  # from S84
    expanding  = atr_3 > 1.30 * atr_20  # from S86
    return 1 if (compressed or expanding) else 0


def sig_s92_s32_compression_hurst(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S92: S32 + time compression + Hurst trending (two strongest features combined).
    S84 (compression): avg $26,638/trade, WR=61%, PF=2.57 — best single signal.
    Adding Hurst H>0.50: ensures the compression is in a trending regime.
    Hypothesis: compressed TSLA in a trending regime = maximum snap-back reliability."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 25:
        return 1
    closes = tsla['close'].iloc[i-20:i+1].values.astype(float)
    highs  = tsla['high'].iloc[i-20:i+1].values.astype(float)
    lows   = tsla['low'].iloc[i-20:i+1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1])))
    atr_5  = tr[-5:].mean()
    atr_20 = tr.mean()
    if atr_5 >= 0.80 * atr_20:  # slightly looser than S84's 0.75 to get more trades
        return 0
    if i >= 65:
        prices = tsla['close'].iloc[i-60:i+1].values.astype(float)
        if _hurst(prices) < 0.48:  # not in mean-reverting regime
            return 0
    return 1


def sig_s93_s32_expansion_bull(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S93: S32 + ATR expansion + SPY in bull market (expanding momentum + macro tailwind).
    S86 found ATR expansion helps. Adding SPY>200d MA ensures it's not bear market expansion.
    Bear market expansions = gap-downs and sell-offs. Bull market = positive breakouts.
    Hypothesis: expanding ATR in bull market = TSLA's lag is closing through genuine buying."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 25:
        return 1
    # SPY bull market filter
    if i >= 200:
        if float(spy['close'].iloc[i]) < float(spy['close'].iloc[i-200:i].mean()):
            return 0
    closes = tsla['close'].iloc[i-20:i+1].values.astype(float)
    highs  = tsla['high'].iloc[i-20:i+1].values.astype(float)
    lows   = tsla['low'].iloc[i-20:i+1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1])))
    atr_3  = tr[-3:].mean()
    atr_20 = tr.mean()
    return 1 if atr_3 > 1.25 * atr_20 else 0  # slightly looser than S86


def sig_s94_s32_buy_pressure(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S94: S32 + buy pressure proxy (Stochastic-style close position in day range).
    Buy pressure = (Close - Low) / (High - Low) — fraction of daily range closed near high.
    High buy pressure (>0.65) = buyers controlling the session despite TSLA lagging.
    Hypothesis: TSLA lagging BUT closing near day's high = accumulation underway → snap-back soon."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    # Average buy pressure over last 3 bars
    bp_total = 0.0
    count = 0
    for j in range(max(0, i-2), i+1):
        h = float(tsla['high'].iloc[j])
        l = float(tsla['low'].iloc[j])
        c = float(tsla['close'].iloc[j])
        if h > l:
            bp_total += (c - l) / (h - l)
            count += 1
    bp = bp_total / count if count > 0 else 0.5
    return 1 if bp >= 0.60 else 0


def sig_s95_s32_signed_volume(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S95: S32 + positive signed volume (Weis Wave proxy — more volume on up vs down days).
    Signed volume = volume × sign(Close - Open). Cumulate over 5 days.
    Positive Weis Wave = accumulation (more volume on up days than down days).
    Hypothesis: TSLA lags SPY but volume is ON THE UPSIDE → institutional accumulation → snap-back."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    signed_vol = 0.0
    for j in range(max(0, i-4), i+1):
        o = float(tsla['open'].iloc[j])
        c = float(tsla['close'].iloc[j])
        v = float(tsla['volume'].iloc[j])
        signed_vol += v * (1 if c >= o else -1)
    return 1 if signed_vol > 0 else 0


def sig_s96_s91_spy_ma200(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S96: S91 (ATR extreme) + SPY above 200-day MA (macro bull filter).
    S91's only big losing year was 2016 (-$85K). S32's 2022 loss is already solved by S91.
    Adding SPY>200d MA should filter the remaining bear-market ATR traps.
    Hypothesis: ATR extremes in bull markets = directional → snap-backs reliable.
    ATR extremes in bear markets = gap-downs and regime shifts → unreliable lag signals."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 200:
        if float(spy['close'].iloc[i]) < float(spy['close'].iloc[i-200:i].mean()):
            return 0
    return 1


def sig_s97_s91_hold8(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S97: S91 (ATR extreme) with 8-day hold time (for scheduling only — same signal fn).
    ATR compression setups (springs) sometimes need more time to unfold than 5 days.
    Testing if hold=8d captures more of the move without adding significant decay.
    Note: hold period is set in the SIGNALS tuple, not the signal function itself."""
    return sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s98_s91_tighter_stop(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S98: S91 (ATR extreme) signal — tighter stop variant (3% stop, set in tuple).
    High PF=2.70 means we can afford a tighter stop — if a trade goes against us, exit faster.
    In compression setups: if the spring doesn't fire immediately, it's likely a failed setup.
    Note: 3% stop is set in the SIGNALS tuple; this fn is identical to S91."""
    return sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s99_s91_no_stop(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S99: S91 (ATR extreme) with no stop loss — hold to signal exit or 5-day timeout.
    At 64% WR and PF=2.70, the wins are very large relative to losses.
    Hypothesis: removing the stop lets big compression snap-backs run fully.
    Note: --no-stop is set in the SIGNALS tuple via stop_pct=None."""
    return sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w)


SIGNALS_P7F: List[Tuple] = [
    # Phase 7F — Physics-based and regime signals
    ('S84_s32_time_compress',    sig_s84_s32_time_compression, 5, 0.04, 50),
    ('S85_s32_volume_dry_up',    sig_s85_s32_volume_dry_up,    5, 0.04, 50),
    ('S86_s32_atr_expansion',    sig_s86_s32_atr_expansion,    5, 0.04, 50),
    ('S87_s32_dow_mon_tue',      sig_s87_s32_dow_mon_tue,      5, 0.04, 50),
    ('S88_s32_hurst_spy_ma',     sig_s88_s32_hurst_spy_ma200,  5, 0.04, 50),
    ('S89_s32_multi_score',      sig_s89_s32_multi_score,      5, 0.04, 50),
    ('S90_s32_compression_bnce', sig_s90_s32_compression_bounce, 5, 0.04, 50),
    # Phase 7G — S91 extensions and variants
    ('S91_s32_atr_extreme',      sig_s91_s32_atr_extreme,      5, 0.04, 50),
    ('S92_s32_compress_hurst',   sig_s92_s32_compression_hurst, 5, 0.04, 50),
    ('S93_s32_expansion_bull',   sig_s93_s32_expansion_bull,   5, 0.04, 50),
    ('S94_s32_buy_pressure',     sig_s94_s32_buy_pressure,     5, 0.04, 50),
    ('S95_s32_signed_volume',    sig_s95_s32_signed_volume,    5, 0.04, 50),
    ('S96_s91_spy_ma200',        sig_s96_s91_spy_ma200,        5, 0.04, 50),
    ('S97_s91_hold8d',           sig_s97_s91_hold8,            8, 0.04, 50),
    ('S98_s91_stop3pct',         sig_s98_s91_tighter_stop,     5, 0.03, 50),
    ('S99_s91_no_stop',          sig_s99_s91_no_stop,         10, 0.20, 50),
]


def sig_s100_s99_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S100: S99 (no stop) + SPY short-term momentum guard.
    S99's worst trades occurred when SPY was in a sharp short-term decline.
    Guard: SPY 5-day return > -3% (not in a panic selloff).
    S91's losers: 2016-02 (energy collapse + Fed hike fear), 2018-09/10 (rising rates), 2019-01/03.
    Hypothesis: when SPY is dropping fast, even ATR-extreme TSLA setups fail."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 5:
        spy_5d = float(spy['close'].iloc[i]) / float(spy['close'].iloc[i-5]) - 1
        if spy_5d < -0.03:  # SPY dropped >3% in 5 days = panic zone
            return 0
    return 1


def sig_s101_s91_looser_atr(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S101: S32 + looser ATR extremes (0.82× compressed or 1.20× expanding).
    S91 used tight thresholds (0.75× or 1.30×). Loosening gives more trades.
    Trade-off: more coverage but possibly lower per-trade quality.
    Test: does loosening from 110→150 trades maintain the PF>2 quality?"""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 25:
        return 1
    closes = tsla['close'].iloc[i-20:i+1].values.astype(float)
    highs  = tsla['high'].iloc[i-20:i+1].values.astype(float)
    lows   = tsla['low'].iloc[i-20:i+1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1])))
    atr_5  = tr[-5:].mean()
    atr_3  = tr[-3:].mean()
    atr_20 = tr.mean()
    compressed = atr_5 < 0.82 * atr_20  # from S84's 0.75
    expanding  = atr_3 > 1.20 * atr_20  # from S86's 1.30
    return 1 if (compressed or expanding) else 0


def sig_s102_s32_atr_score(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S102: S32 + dynamic ATR score — fire at ANY ATR extreme AND score other conditions.
    Use 0.82/1.20 thresholds (S101) AND require score >= 2 of:
      1. Hurst >= 0.48 (not strongly mean-reverting)
      2. Not August
      3. SPY above 50d MA (not in short-term downtrend)
    Hypothesis: broaden ATR condition while soft-filtering with quality signals."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 25:
        return 1
    closes = tsla['close'].iloc[i-20:i+1].values.astype(float)
    highs  = tsla['high'].iloc[i-20:i+1].values.astype(float)
    lows   = tsla['low'].iloc[i-20:i+1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1])))
    atr_5  = tr[-5:].mean()
    atr_3  = tr[-3:].mean()
    atr_20 = tr.mean()
    if not (atr_5 < 0.82 * atr_20 or atr_3 > 1.20 * atr_20):
        return 0
    score = 0
    dt = tsla.index[i]
    if i >= 65:
        prices = tsla['close'].iloc[i-60:i+1].values.astype(float)
        if _hurst(prices) >= 0.48:
            score += 1
    else:
        score += 1
    if dt.month != 8:
        score += 1
    if i >= 50:
        if float(spy['close'].iloc[i]) >= float(spy['close'].iloc[i-50:i].mean()):
            score += 1
    else:
        score += 1
    return 1 if score >= 2 else 0


def sig_s103_s32_rsi_pullback(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S103: S32 + RSI pullback filter (TSLA RSI in 30-55 zone before snap-back).
    RSI 30-55 = genuine weakness/consolidation, not panic (RSI<20) or still strong (RSI>55).
    When TSLA lags SPY in this RSI zone, the lag is likely a controlled pullback, not a breakdown.
    Hypothesis: controlled RSI weakness + ATR extreme = highest snap-back probability."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 15:
        return 1
    closes = tsla['close'].iloc[i-14:i+1].values.astype(float)
    gains = np.maximum(np.diff(closes), 0)
    losses = np.maximum(-np.diff(closes), 0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss < 1e-10:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - 100 / (1 + rs)
    return 1 if 25 <= rsi <= 60 else 0


def _get_tsla_earnings_dates():
    """TSLA approximate earnings dates (4th Wednesday of Jan/Apr/Jul/Oct, ±1 week).
    Used to avoid entering trades close to earnings where IV expansion distorts price.
    Pre-earnings: IV swells → prices anchored near strikes → lags persist longer.
    Post-earnings: IV crush → gap moves → unreliable lag signal.
    """
    # Approximate TSLA earnings dates 2015-2025
    return [
        pd.Timestamp('2015-01-28'), pd.Timestamp('2015-04-22'), pd.Timestamp('2015-07-22'), pd.Timestamp('2015-10-07'),
        pd.Timestamp('2016-02-10'), pd.Timestamp('2016-05-04'), pd.Timestamp('2016-08-03'), pd.Timestamp('2016-10-26'),
        pd.Timestamp('2017-02-22'), pd.Timestamp('2017-05-03'), pd.Timestamp('2017-08-02'), pd.Timestamp('2017-11-01'),
        pd.Timestamp('2018-02-07'), pd.Timestamp('2018-05-02'), pd.Timestamp('2018-08-01'), pd.Timestamp('2018-10-24'),
        pd.Timestamp('2019-02-20'), pd.Timestamp('2019-04-24'), pd.Timestamp('2019-07-24'), pd.Timestamp('2019-10-23'),
        pd.Timestamp('2020-01-29'), pd.Timestamp('2020-04-29'), pd.Timestamp('2020-07-22'), pd.Timestamp('2020-10-21'),
        pd.Timestamp('2021-01-27'), pd.Timestamp('2021-04-26'), pd.Timestamp('2021-07-26'), pd.Timestamp('2021-10-20'),
        pd.Timestamp('2022-01-26'), pd.Timestamp('2022-04-20'), pd.Timestamp('2022-07-20'), pd.Timestamp('2022-10-19'),
        pd.Timestamp('2023-01-25'), pd.Timestamp('2023-04-19'), pd.Timestamp('2023-07-19'), pd.Timestamp('2023-10-18'),
        pd.Timestamp('2024-01-24'), pd.Timestamp('2024-04-23'), pd.Timestamp('2024-07-23'), pd.Timestamp('2024-10-23'),
        pd.Timestamp('2025-01-29'), pd.Timestamp('2025-04-22'), pd.Timestamp('2025-07-23'),
    ]


_TSLA_EARNINGS = _get_tsla_earnings_dates()


def _days_to_earnings(dt) -> int:
    """Days to next TSLA earnings (or from last if past)."""
    d = pd.Timestamp(dt).normalize()
    future = [abs((d - e).days) for e in _TSLA_EARNINGS if e >= d]
    past   = [abs((d - e).days) for e in _TSLA_EARNINGS if e < d]
    to_next   = min(future) if future else 999
    from_last = min(past)   if past   else 999
    return min(to_next, from_last)  # days from nearest earnings


def sig_s104_s91_no_earnings(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S104: S91 (ATR extreme) + NOT within 7 days of TSLA earnings.
    Pre-earnings: IV expansion anchors price near max-pain strike → lags persist longer.
    Post-earnings: gap and IV crush → undefined directional momentum.
    The 7-day exclusion avoids the 'dead zone' where fundamental news dominates technicals.
    Hypothesis: removing earnings-adjacent S91 trades should reduce variance and improve WR."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    if _days_to_earnings(dt) <= 7:
        return 0
    return 1


def sig_s105_s91_tsla_bull(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S105: S91 + TSLA itself above 200d MA (TSLA is in its own bull market).
    S96 uses SPY>200d MA (macro bull). S105 uses TSLA>200d MA (stock-specific bull).
    Idea: even if SPY is in a bull market, TSLA can be in its own bear (2022).
    Adding TSLA-specific bull check should filter TSLA idiosyncratic bear markets."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 200:
        tsla_close = float(tsla['close'].iloc[i])
        tsla_ma200 = float(tsla['close'].iloc[i-200:i].mean())
        if tsla_close < tsla_ma200:
            return 0
    return 1


def sig_s106_s32_best_months(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S106: S32 in June, January, November only (statistically best 5d return months).
    Calendar analysis: Jun +2.736% (p=0.000**), Jan +1.956% (p=0.000**), Nov +1.487% (p=0.002**).
    The S32 lag signal in a month with strong positive seasonality = dual tailwind.
    Hypothesis: SPY-TSLA lag convergence is faster when broader TSLA seasonality is positive."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    return 1 if dt.month in (1, 6, 11) else 0


def sig_s107_s91_vix_elevated(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S107: S91 + elevated VIX (VIX >= 18) — high-vol regime snap-backs are bigger.
    c9 regime analysis: high VIX (>30) = $3,393/trade (1.66x boost). Mid VIX (20-30) = 1.35x.
    When ATR is extreme AND VIX is elevated, the snap-back energy is at maximum.
    Hypothesis: VIX elevation amplifies ATR-extreme snap-backs → even higher avg P&L."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    vix_now = float(vix['close'].iloc[i])
    return 1 if vix_now >= 18 else 0


def sig_s108_s32_best_regime(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S108: S32 + ALL three regime conditions (ultimate quality filter).
    Requirements:
      1. ATR at extreme (compressed OR expanding) — S91 condition
      2. SPY above 200d MA (macro bull market)
      3. VIX >= 18 (some vol — clean vol, not panic)
    Hypothesis: stacking ATR-extreme + bull market + elevated vol = maximum snap-back."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 200:
        if float(spy['close'].iloc[i]) < float(spy['close'].iloc[i-200:i].mean()):
            return 0
    vix_now = float(vix['close'].iloc[i]) if i >= 0 else 20.0
    return 1 if vix_now >= 18 else 0


SIGNALS_P7H: List[Tuple] = [
    # Phase 7H — S91/S99 refinements and novel combinations
    ('S100_s99_spy_momentum',    sig_s100_s99_spy_momentum,    10, 0.20, 50),
    ('S101_s32_looser_atr',      sig_s101_s91_looser_atr,       5, 0.04, 50),
    ('S102_s32_atr_score',       sig_s102_s32_atr_score,        5, 0.04, 50),
    ('S103_s91_rsi_pullback',    sig_s103_s32_rsi_pullback,     10, 0.20, 50),
    # Phase 7I — Earnings exclusion, sector, seasonality, regime
    ('S104_s91_no_earnings',     sig_s104_s91_no_earnings,       10, 0.20, 50),
    ('S105_s91_tsla_bull',       sig_s105_s91_tsla_bull,          5, 0.04, 50),
    ('S106_s32_best_months',     sig_s106_s32_best_months,        5, 0.04, 50),
    ('S107_s91_vix_elevated',    sig_s107_s91_vix_elevated,      10, 0.20, 50),
    ('S108_s32_best_regime',     sig_s108_s32_best_regime,       10, 0.20, 50),
]


SIGNALS_P6D: List[Tuple] = [
    # (name, fn, max_hold_days, stop_pct, channel_window)
    # Wider hold (8-10d) to give breaks time to develop
    ('S50_compression_breakout',     sig_s50_channel_compression_breakout, 10, 0.05, 50),
    ('S51_multi_touch_upper',        sig_s51_multi_touch_upper,            10, 0.05, 50),
    ('S52_rsi_divergence_break',     sig_s52_rsi_divergence_breakout,      10, 0.05, 50),
    ('S53_spy_accel_tsla_lag',       sig_s53_spy_accel_tsla_lag_break,      7, 0.04, 50),
    ('S54_compression_spy_lag',      sig_s54_compression_spy_lag,          10, 0.05, 50),
    ('S55_breakout_union',           sig_s55_breakout_union,               10, 0.05, 50),
    # Filtered variants
    ('S56_s51_ma200',                sig_s56_s51_ma200,                    10, 0.05, 50),
    ('S57_s51_intersect_s32',        sig_s57_s51_intersect_s32,             5, 0.04, 50),
    ('S58_s50_ma200',                sig_s58_s50_ma200,                    10, 0.05, 50),
    ('S59_bounce_or_break',          sig_s59_breakout_or_bounce,            7, 0.04, 50),
    ('S60_s51_vix_regime',           sig_s60_s51_vix_regime,               10, 0.05, 50),
    ('S61_s51_ma200_vix',            sig_s61_s51_ma200_vix,                10, 0.05, 50),
    ('S62_s32_plus_s51_vix',         sig_s62_s32_plus_s51_union,            5, 0.04, 50),
    ('S63_s62_ma200',                sig_s63_s62_ma200,                     5, 0.04, 50),
    ('S64_s32_dynamic_bear_guard',   sig_s64_dynamic_bear_guard,            5, 0.04, 50),
    ('S65_s32_bullish_candle',       sig_s65_s32_bullish_candle,            5, 0.04, 50),
    ('S66_s32_volume_surge',         sig_s66_s32_volume_surge,              5, 0.04, 50),
    ('S67_s32_rsi_exit',             sig_s67_s32_rsi_exit,                  5, 0.04, 50),
    ('S68_s62_dynamic_bear',         sig_s68_s62_dynamic_bear,              5, 0.04, 50),
]

SIGNALS_P5D: List[Tuple] = [
    # (name, fn, max_hold_days, stop_pct, channel_window)
    ('S41_s32_spy_ma200',            sig_s41_s32_spy_ma200,            5, 0.04, 50),
    ('S42_s32_spy_ma100',            sig_s42_s32_spy_ma100,            5, 0.04, 50),
    ('S43_s32_no_gap_down',          sig_s43_s32_no_gap_down,          5, 0.04, 50),
    ('S44_s41_hold8d',               sig_s44_s41_hold8,                8, 0.04, 50),
    ('S45_s32_tsla_above_50d',       sig_s45_s32_tsla_above_50d,       5, 0.04, 50),
    ('S46_s41_tsla_above_50d',       sig_s46_s41_tsla_above_50d,       5, 0.04, 50),
    ('S47_s32_ma200_no_gap',         sig_s47_s32_spy_ma200_no_gap,     5, 0.04, 50),
]

def sig_s109_s91_atr_inflection(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S109: S32 + ATR turning from compressed to expanding (inflection point).
    More selective than S91: requires BOTH prior compression AND NOW expanding.
    The spring not just coiled, but RELEASING: ATR_3 > ATR_5 (turning up from compression).
    Hypothesis: the inflection from quiet to active catches the exact snap-back moment."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 25:
        return 1
    closes = tsla['close'].iloc[i-20:i+1].values.astype(float)
    highs  = tsla['high'].iloc[i-20:i+1].values.astype(float)
    lows   = tsla['low'].iloc[i-20:i+1].values.astype(float)
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - closes[:-1]),
                    np.abs(lows[1:]  - closes[:-1])))
    atr_3  = tr[-3:].mean()
    atr_5  = tr[-5:].mean()
    atr_10 = tr[-10:].mean()
    atr_20 = tr.mean()
    # Spring releasing: was compressed (ATR_10 < 0.85×ATR_20) AND now expanding (ATR_3 > ATR_5)
    was_compressed = atr_10 < 0.85 * atr_20
    now_expanding  = atr_3 > atr_5 * 1.10  # ATR_3 at least 10% above ATR_5
    return 1 if (was_compressed and now_expanding) else 0


def sig_s110_s32_earnings_catalyst(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S110: S91 NEAR earnings (within 3-7 days) — earnings as forced catalyst.
    Earnings force the SPY-TSLA lag to close (no more 'ignoring' the divergence).
    S104 excluded earnings; S110 targets them specifically.
    S99 analysis showed earnings-adjacent trades average $36K+/trade (above overall avg).
    Hypothesis: ATR-extreme lag + imminent earnings = strongest snap-back catalyst."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    days = _days_to_earnings(dt)
    return 1 if 3 <= days <= 10 else 0


def sig_s111_s99_short_hold(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S111: S91 signal, 3-day max hold (no stop) — capture snap-back core only.
    S99 shows avg hold = 2.5d and most big winners resolve in 2-4 days.
    Truncating to 3 days reduces exposure to tail risk while keeping core move.
    Note: hold=3, stop=20% (effectively none) set in SIGNALS tuple."""
    return sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w)


SIGNALS_P7J: List[Tuple] = [
    # Phase 7J — Advanced refinements
    ('S109_s91_atr_inflection',   sig_s109_s91_atr_inflection,    10, 0.20, 50),
    ('S110_s32_earnings_catalyst', sig_s110_s32_earnings_catalyst, 10, 0.20, 50),
    ('S111_s91_hold3d',           sig_s111_s99_short_hold,          3, 0.20, 50),
]


# ── Phase 7K — VIX filter + fast exit, deep lag, volume, micro-timing ────────

def sig_s112_s107_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S112: S107 (VIX>=18 + ATR extreme) with 3-day hold.
    S107 WR=75%, PF=4.26 — but S111 showed 3d hold lifts results vs 10d.
    Hypothesis: combining the VIX quality filter with faster exit is optimal."""
    return sig_s107_s91_vix_elevated(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s113_s108_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S113: S108 (SPY>200d + VIX>=18 + ATR extreme) with 3-day hold.
    S108 WR=76%, PF=5.60 — the tightest quality filter. Test fast exit on it.
    Hypothesis: best-regime signal + snap-back exit = maximum Sharpe."""
    return sig_s108_s32_best_regime(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s114_s91_gap_entry(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S114: S91 + today's open gaps DOWN vs prev close (>1% gap).
    A gap-down on signal day = forced liquidation at open = better entry price.
    Hypothesis: entering after a gap-down = discounted entry into snap-back."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 1
    today_open = float(tsla['open'].iloc[i])
    prev_close  = float(tsla['close'].iloc[i-1])
    return 1 if today_open < prev_close * 0.990 else 0


def sig_s115_s91_deep_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S115: S91 + TSLA has lagged SPY by >3% over 5 days (meaningful divergence).
    S32 fires on any positive lag; S115 requires a quantified meaningful lag.
    Hypothesis: larger divergence = larger snap-back amplitude."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    tsla_5d = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i-5]) - 1
    spy_5d  = float(spy['close'].iloc[i])  / float(spy['close'].iloc[i-5])  - 1
    lag = spy_5d - tsla_5d  # positive = TSLA lagging
    return 1 if lag > 0.03 else 0


def sig_s116_s91_spy_strong(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S116: S91 + SPY 5d return > 2% (strong market pulling TSLA).
    When SPY has surged, the lag tension on TSLA is greatest.
    Hypothesis: high SPY momentum = stronger snap-back force on lagging TSLA."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    spy_5d = float(spy['close'].iloc[i]) / float(spy['close'].iloc[i-5]) - 1
    return 1 if spy_5d > 0.02 else 0


def sig_s117_s91_hold2d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S117: S91 with 2-day max hold — capture only the first impulse.
    S111 (3d hold) improved over S99 (10d hold). Test if 2d hold is even better.
    Hypothesis: the snap-back core resolves in 48 hours."""
    return sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s118_s91_rsi_oversold(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S118: S91 + TSLA RSI < 40 (momentum oversold at ATR extreme).
    Double oversold: ATR-extreme (price not moving) AND RSI declining.
    Hypothesis: RSI oversold + ATR snap = maximum mean-reversion setup."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 14:
        return 1
    closes = tsla['close'].iloc[i-14:i+1].values.astype(float)
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = gains.mean(); avg_l = losses.mean()
    rsi    = 100.0 if avg_l < 1e-10 else 100 - 100 / (1 + avg_g / avg_l)
    return 1 if rsi < 40 else 0


def sig_s119_s91_vol_spike(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S119: S91 + today's volume >= 1.5x 20-day avg (capitulation volume).
    High volume on ATR-extreme day = institutional conviction or forced seller.
    Hypothesis: volume surge on snap setup = stronger confirmation of reversal."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    vol_now = float(tsla['volume'].iloc[i])
    vol_avg = float(tsla['volume'].iloc[i-20:i].mean())
    return 1 if vol_now >= 1.5 * vol_avg else 0


def sig_s120_s91_below_20ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S120: S91 + TSLA close below its 20d MA (near-term bearish = deeper pullback).
    Buying the lag when TSLA is also below its 20d MA = double-oversold setup.
    Hypothesis: ATR-extreme + below 20d MA = stronger mean-reversion force."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    tsla_c  = float(tsla['close'].iloc[i])
    ma20    = float(tsla['close'].iloc[i-20:i].mean())
    return 1 if tsla_c < ma20 else 0


def sig_s121_s111_spy_bull(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S121: S111 (3d champion) + SPY > 200d MA (macro bull filter).
    Remove all bear-market trades from the champion signal.
    Hypothesis: champion snap-backs are cleaner in bull markets."""
    if sig_s111_s99_short_hold(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 200:
        if float(spy['close'].iloc[i]) < float(spy['close'].iloc[i-200:i].mean()):
            return 0
    return 1


def sig_s122_s111_vix_bull(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S122: S111 (3d hold) + VIX>=18 + SPY>200d MA (all three).
    Stack VIX elevation + macro bull onto the fast-hold champion.
    Hypothesis: S111 trades in a bull market with elevated vol are the cream."""
    if sig_s111_s99_short_hold(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 200:
        if float(spy['close'].iloc[i]) < float(spy['close'].iloc[i-200:i].mean()):
            return 0
    vix_now = float(vix['close'].iloc[i]) if i >= 0 else 20.0
    return 1 if vix_now >= 18 else 0


SIGNALS_P7K: List[Tuple] = [
    # Phase 7K — VIX+fast-exit combos, deep lag, micro-timing
    ('S112_s107_hold3d',   sig_s112_s107_hold3d,   3, 0.20, 50),
    ('S113_s108_hold3d',   sig_s113_s108_hold3d,   3, 0.20, 50),
    ('S114_s91_gap_entry', sig_s114_s91_gap_entry, 10, 0.20, 50),
    ('S115_s91_deep_lag',  sig_s115_s91_deep_lag,  10, 0.20, 50),
    ('S116_s91_spy_strong', sig_s116_s91_spy_strong, 10, 0.20, 50),
    ('S117_s91_hold2d',    sig_s117_s91_hold2d,     2, 0.20, 50),
    ('S118_s91_rsi_os',    sig_s118_s91_rsi_oversold, 10, 0.20, 50),
    ('S119_s91_vol_spike', sig_s119_s91_vol_spike,  10, 0.20, 50),
    ('S120_s91_below20ma', sig_s120_s91_below_20ma, 10, 0.20, 50),
    ('S121_s111_spy_bull', sig_s121_s111_spy_bull,   3, 0.20, 50),
    ('S122_s111_vix_bull', sig_s122_s111_vix_bull,   3, 0.20, 50),
]


# ── Phase 7L — Decomposed ATR + unorthodox signals ───────────────────────────

def _atr_components(tsla, i):
    """Return (atr_3, atr_5, atr_10, atr_20) or None if i < 20."""
    if i < 20:
        return None
    closes = tsla['close'].iloc[i-20:i+1].values.astype(float)
    highs  = tsla['high'].iloc[i-20:i+1].values.astype(float)
    lows   = tsla['low'].iloc[i-20:i+1].values.astype(float)
    tr = np.maximum(highs[1:]-lows[1:],
         np.maximum(np.abs(highs[1:]-closes[:-1]),
                    np.abs(lows[1:]-closes[:-1])))
    return tr[-3:].mean(), tr[-5:].mean(), tr[-10:].mean(), tr.mean()


def sig_s123_s91_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S123: S91 — COMPRESSED component only (ATR_5 < 0.75×ATR_20, NOT expanding).
    Decompose S91: isolate the 'coiling spring' trades from the 'explosive' trades.
    Hypothesis: compression drives MORE alpha than expansion (spring analogy)."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    atr_3, atr_5, _, atr_20 = c
    compressed = atr_5 < 0.75 * atr_20
    expanding  = atr_3 > 1.30 * atr_20
    return 1 if (compressed and not expanding) else 0


def sig_s124_s91_expanding_only(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S124: S91 — EXPANDING component only (ATR_3 > 1.30×ATR_20, NOT compressed).
    Decompose S91: isolate the 'momentum building' trades from the 'coiling' trades.
    If expansion >>> compression, we want the moving stock, not the quiet one."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    atr_3, atr_5, _, atr_20 = c
    compressed = atr_5 < 0.75 * atr_20
    expanding  = atr_3 > 1.30 * atr_20
    return 1 if (expanding and not compressed) else 0


def sig_s125_s107_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S125: S107 (VIX>=18) but COMPRESSED-ONLY component.
    When VIX is elevated and TSLA is quietly coiling = maximum tension.
    Market fear + TSLA silence = spring about to explode."""
    if sig_s107_s91_vix_elevated(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s126_s91_open_gap_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S126: S91 + today's price ACTION bullish (close > open, bullish candle day).
    The signal fires at close; entry is next bar open. A bullish candle on signal day
    means intraday buyers confirmed the lag closure. Hypothesis: candle direction
    on signal day predicts next bar direction."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    today_open  = float(tsla['open'].iloc[i])
    today_close = float(tsla['close'].iloc[i])
    return 1 if today_close > today_open * 1.005 else 0  # close > open by >0.5%


def sig_s127_s91_double_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S127: S91 + TSLA lagging its sector XLY (consumer discretionary).
    If XLY (sector ETF) has outperformed TSLA by >2% in 5d, sector AND market both
    diverged from TSLA simultaneously. Double lag = stronger mean-reversion pressure.
    Note: XLY not fetched separately — use SPY×1.2 as rough sector proxy."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    # Deep lag: TSLA vs SPY >4% over 5d (more extreme than S115's 3%)
    tsla_5d = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i-5]) - 1
    spy_5d  = float(spy['close'].iloc[i])  / float(spy['close'].iloc[i-5])  - 1
    lag_5d  = spy_5d - tsla_5d
    # AND TSLA vs SPY >2% over 10d (persistent, not just 1-week noise)
    tsla_10d = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i-10]) - 1
    spy_10d  = float(spy['close'].iloc[i])  / float(spy['close'].iloc[i-10]) - 1
    lag_10d  = spy_10d - tsla_10d
    return 1 if (lag_5d > 0.04 and lag_10d > 0.02) else 0


def sig_s128_s32_after_failure(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S128: S32 but only the 2nd signal after a failed S32 trade.
    After a S32 trade fails (exit at stop), the next S32 signal often has STRONGER
    follow-through (the divergence wasn't resolved, so it built up further).
    Hypothesis: failed → retry pattern improves subsequent trade quality.
    Implementation: track last exit reason via relative price position."""
    # Simplified: only fire if TSLA is still below 5-day ago level (lag persisting)
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    # TSLA has lagged for multiple bars (10d return still negative = lag hasn't resolved)
    tsla_10d = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i-10]) - 1
    spy_10d  = float(spy['close'].iloc[i])  / float(spy['close'].iloc[i-10]) - 1
    # TSLA underperformed SPY over 10d AND S32 fires again = persistent lag
    return 1 if (spy_10d - tsla_10d) > 0.02 else 0


def sig_s129_s91_high_kurtosis(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S129: S91 + recent TSLA returns show high kurtosis (fat tails building).
    High kurtosis in recent daily returns = price making occasional big moves,
    mostly small moves. This is the statistical signature of a coiling market.
    Hypothesis: high kurtosis predicts the next big move (ATR snap)."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    closes = tsla['close'].iloc[i-20:i+1].values.astype(float)
    rets   = np.diff(closes) / closes[:-1]
    if len(rets) < 10:
        return 1
    mu  = rets.mean()
    std = rets.std()
    if std < 1e-10:
        return 1
    kurt = np.mean(((rets - mu) / std) ** 4) - 3.0  # excess kurtosis
    return 1 if kurt > 0.5 else 0  # fat tails = big moves coming


def sig_s130_s111_vix20(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S130: S111 (3d champion) + VIX >= 20 (tighter VIX filter than S122's >=18).
    Test if even higher VIX threshold = even better results.
    S107 uses >=18, S122 uses >=18 + bull. Does >=20 standalone beat all?"""
    if sig_s111_s99_short_hold(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i]) if i >= 0 else 20.0
    return 1 if vix_now >= 20 else 0


def sig_s131_s91_range_contraction(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S131: S91 + NR7 pattern (narrowest range of last 7 bars).
    NR7 = classic Toby Crabel pattern: price range contracts to a 7-bar minimum.
    The body of the candle (H-L) is smallest in 7 bars = compression peak.
    Hypothesis: NR7 marks the EXACT compression peak = best snap-back entry."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 7:
        return 1
    today_range = float(tsla['high'].iloc[i]) - float(tsla['low'].iloc[i])
    past_ranges = [float(tsla['high'].iloc[i-j]) - float(tsla['low'].iloc[i-j])
                   for j in range(1, 7)]
    return 1 if today_range <= min(past_ranges) else 0


def sig_s132_s91_vix_spike_recovery(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S132: S91 + VIX was elevated 5d ago but has since dropped (VIX spike recovery).
    When fear spikes (VIX up) and then recedes, risk assets snap back hard.
    TSLA often lags this VIX-recovery rally — S91 on VIX-recovery = amplified.
    Hypothesis: recovering-from-fear environment is optimal for lag closures."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now   = float(vix['close'].iloc[i])
    vix_5d    = float(vix['close'].iloc[i-5])
    vix_10d   = float(vix['close'].iloc[i-10])
    # VIX peaked 5-10d ago AND has since come down (recovery pattern)
    spike_then = vix_5d > 20 or vix_10d > 20  # was elevated
    calm_now   = vix_now < vix_5d * 0.90       # has dropped >=10%
    return 1 if (spike_then and calm_now) else 0


SIGNALS_P7L: List[Tuple] = [
    # Phase 7L — Decomposed ATR + unorthodox
    ('S123_s91_compressed',       sig_s123_s91_compressed_only,   10, 0.20, 50),
    ('S124_s91_expanding',        sig_s124_s91_expanding_only,    10, 0.20, 50),
    ('S125_s107_compressed',      sig_s125_s107_compressed_only,  10, 0.20, 50),
    ('S126_s91_bull_candle',      sig_s126_s91_open_gap_up,       10, 0.20, 50),
    ('S127_s91_double_lag',       sig_s127_s91_double_lag,        10, 0.20, 50),
    ('S128_s32_persistent_lag',   sig_s128_s32_after_failure,      5, 0.04, 50),
    ('S129_s91_kurtosis',         sig_s129_s91_high_kurtosis,     10, 0.20, 50),
    ('S130_s111_vix20',           sig_s130_s111_vix20,             3, 0.20, 50),
    ('S131_s91_nr7',              sig_s131_s91_range_contraction, 10, 0.20, 50),
    ('S132_s91_vix_recovery',     sig_s132_s91_vix_spike_recovery, 10, 0.20, 50),
]


# ── Phase 7M — NR-family + VIX recovery extensions + ultra-quality ───────────

def _nr_n(tsla, i, n):
    """True if today's range (H-L) is the narrowest of last n bars."""
    if i < n:
        return True
    today_range = float(tsla['high'].iloc[i]) - float(tsla['low'].iloc[i])
    for j in range(1, n):
        past_r = float(tsla['high'].iloc[i-j]) - float(tsla['low'].iloc[i-j])
        if today_range > past_r:
            return False
    return True


def sig_s133_s32_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S133: S32 baseline + NR7 (WITHOUT requiring ATR extreme).
    S131 requires ATR extreme + NR7. S133 asks: does NR7 add value to plain S32?
    NR7 alone might identify compression better than ATR_5/ATR_20 ratio."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s134_s91_nr4(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S134: S91 + NR4 (4-bar narrowest range — rarer, more extreme compression).
    NR4 is even more selective than NR7: only 1 in ~16 bars qualify.
    Hypothesis: NR4 marks the PEAK of compression — maximum spring tension."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 4) else 0


def sig_s135_s91_nr7_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S135: S91 + NR7 + VIX>=18 — triple compression signal.
    NR7 (price range minimum) + ATR-extreme (ATR minimum or max) + elevated VIX.
    Hypothesis: all three compression/vol indicators agreeing = most extreme setup."""
    if sig_s131_s91_range_contraction(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i]) if i >= 0 else 20.0
    return 1 if vix_now >= 18 else 0


def sig_s136_s132_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S136: VIX recovery (S132) + COMPRESSED ATR — double signal.
    VIX was high and is coming down, AND TSLA is quietly coiling.
    The market de-fearing + TSLA spring = amplified snap."""
    if sig_s132_s91_vix_spike_recovery(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s137_s125_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S137: S125 (VIX>=18 + compressed) with 3d hold.
    S125 WR=81%, PF=6.41, avg=$47K. Does capping hold at 3d improve it?"""
    return sig_s125_s107_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s138_s131_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S138: S131 (NR7 + ATR extreme) with 3d hold.
    S131: 8/8 years profitable, WR=75%, PF=5.01. Fast exit variant."""
    return sig_s131_s91_range_contraction(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s139_s132_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S139: S132 (VIX recovery) with 3d hold.
    S132: WR=75%, PF=8.53, avg=$54,894 — best per-trade. Fast exit variant."""
    return sig_s132_s91_vix_spike_recovery(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s140_s91_vix_term(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S140: S91 + VIX term structure (VIX dropping from recent peak = backwardation→contango).
    Measure VIX trend: if VIX today < VIX_5d AND VIX_5d > 20, market just left fear mode.
    This is a softer version of S132 (only needs VIX to be falling from >=20, not 10% drop)."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i-5])
    return 1 if (vix_5d > 20 and vix_now < vix_5d) else 0


def sig_s141_s112_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S141: S112 (S107+3d hold) restricted to COMPRESSED-only ATR.
    S112 WR=76%, PF=4.99. Does removing expanding-ATR trades improve it?
    Hypothesis: in VIX elevated + compressed state, snap is more reliable."""
    if sig_s112_s107_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s142_s91_vix_crush(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S142: S91 + VIX has been falling for 3 consecutive days (momentum declining).
    Sustained VIX decline = fear leaving market = risk-on momentum.
    TSLA lags in this environment and then snaps when fear has fully left."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    v0 = float(vix['close'].iloc[i])
    v1 = float(vix['close'].iloc[i-1])
    v2 = float(vix['close'].iloc[i-2])
    v3 = float(vix['close'].iloc[i-3])
    return 1 if (v0 < v1 < v2 < v3) else 0  # 3 consecutive VIX down days


SIGNALS_P7M: List[Tuple] = [
    # Phase 7M — NR family + VIX recovery extensions
    ('S133_s32_nr7',           sig_s133_s32_nr7,           5, 0.04, 50),
    ('S134_s91_nr4',           sig_s134_s91_nr4,          10, 0.20, 50),
    ('S135_s91_nr7_vix',       sig_s135_s91_nr7_vix,      10, 0.20, 50),
    ('S136_s132_compressed',   sig_s136_s132_compressed,  10, 0.20, 50),
    ('S137_s125_hold3d',       sig_s137_s125_hold3d,       3, 0.20, 50),
    ('S138_s131_hold3d',       sig_s138_s131_hold3d,       3, 0.20, 50),
    ('S139_s132_hold3d',       sig_s139_s132_hold3d,       3, 0.20, 50),
    ('S140_s91_vix_falling',   sig_s140_s91_vix_term,     10, 0.20, 50),
    ('S141_s112_compressed',   sig_s141_s112_compressed,   3, 0.20, 50),
    ('S142_s91_vix_crush',     sig_s142_s91_vix_crush,    10, 0.20, 50),
]


# ── Phase 7N — Ultra-quality combos + VIX recovery depth + timing ─────────────

def sig_s143_s136_spy_bull(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S143: S136 (VIX recovery + compressed) + SPY > 200d MA.
    Add macro bull filter to the best signal (S136 PF=16.81).
    Hypothesis: VIX recovery + ATR compression + bull market = perfect trifecta."""
    if sig_s136_s132_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 200:
        if float(spy['close'].iloc[i]) < float(spy['close'].iloc[i-200:i].mean()):
            return 0
    return 1


def sig_s144_vix_recovery_strong(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S144: S91 + stronger VIX recovery (VIX 5d ago > 22, now down 15%).
    Tighter version of S132: requires deeper VIX spike and sharper recovery.
    Hypothesis: larger VIX spike → larger fear event → larger snap when resolved."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i-5])
    vix_10d = float(vix['close'].iloc[i-10])
    spike_then = vix_5d > 22 or vix_10d > 22
    calm_now   = vix_now < vix_5d * 0.85  # 15% drop required
    return 1 if (spike_then and calm_now) else 0


def sig_s145_s91_spy_run(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S145: S91 + SPY up 3+ consecutive days (SPY on a streak, TSLA quiet).
    When SPY strings together consecutive positive closes but TSLA hasn't moved,
    the divergence is building day-by-day. The snap is forced by momentum.
    Hypothesis: SPY streak + ATR extreme = momentum-forced lag closure."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    # SPY up 3 consecutive days
    for j in range(3):
        if float(spy['close'].iloc[i-j]) <= float(spy['close'].iloc[i-j-1]):
            return 0
    return 1


def sig_s146_s91_tsla_down10d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S146: S91 + TSLA 10d return < 0 while SPY 10d > 1%.
    TSLA has actually declined over 10 days while SPY has risen.
    This is stronger divergence than S32 (which only looks at 3-5d lag).
    Hypothesis: 10d negative TSLA return with rising SPY = maximum pent-up lag."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    tsla_10d = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i-10]) - 1
    spy_10d  = float(spy['close'].iloc[i])  / float(spy['close'].iloc[i-10])  - 1
    return 1 if (tsla_10d < 0 and spy_10d > 0.01) else 0


def sig_s147_s91_monday(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S147: S91 on MONDAY only (day of week = 0).
    Weekend holds let the SPY-TSLA gap widen undisturbed — largest divergence
    builds over weekends. Entry Monday = first shot at lag closure of the week.
    Hypothesis: Mon ATR-extreme signal = 2 days of unclosed gap pressure."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if tsla.index[i].dayofweek == 0 else 0


def sig_s148_s91_month_start(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S148: S91 on 1st-5th trading day of month (month-start rebalancing flow).
    Large institutions rebalance portfolios at month-start → buy SPY, delay TSLA.
    If S91 fires in this window, institutional flow is the CATALYST for snap.
    Hypothesis: month-start rebalancing amplifies the lag closure force."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    # Count trading days so far this month
    year, month = dt.year, dt.month
    trading_days_in_month = tsla.index[(tsla.index.year == year) & (tsla.index.month == month)]
    day_rank = list(trading_days_in_month).index(dt) + 1 if dt in trading_days_in_month else 99
    return 1 if day_rank <= 5 else 0


def sig_s149_s91_nr7_vix_recovery(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S149: S91 + NR7 + VIX recovery — three independent signals agreeing.
    NR7: price compressed (Crabel). VIX recovery: fear receding. S91: ATR extreme.
    Hypothesis: three independent compression/recovery signals = maximum snap."""
    if sig_s131_s91_range_contraction(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i-5])
    vix_10d = float(vix['close'].iloc[i-10])
    spike_then = vix_5d > 18 or vix_10d > 18
    calm_now   = vix_now < vix_5d * 0.92
    return 1 if (spike_then and calm_now) else 0


def sig_s150_s91_rsi_range(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S150: S91 + TSLA RSI 30-50 (oversold but not extreme — stable oversold).
    RSI < 30 = too extreme / capitulation (S118 uses < 40).
    RSI 30-50 = steady downtrend / lag zone = systematic underperformance.
    Hypothesis: moderate RSI oversold + ATR extreme = most persistent lag setup."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 14:
        return 1
    closes = tsla['close'].iloc[i-14:i+1].values.astype(float)
    deltas = np.diff(closes)
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = gains.mean(); avg_l = losses.mean()
    rsi    = 100.0 if avg_l < 1e-10 else 100 - 100 / (1 + avg_g / avg_l)
    return 1 if 30 <= rsi <= 50 else 0


def sig_s151_s136_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S151: S136 (VIX recovery + compressed) + NR7 — ultra-rare, ultra-quality.
    Requires VIX spike recovery AND ATR compressed AND price range at 7-bar minimum.
    Three independent signals all pointing at maximum compression + fear recovery.
    Hypothesis: this fires only in the most perfect setup — near 100% accuracy."""
    if sig_s136_s132_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s152_s91_vix15(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S152: S91 + VIX >= 15 (lower threshold — captures more 'slightly elevated' vol).
    S107 uses >=18, S130 uses >=20. Test >=15 = base volatility guard.
    Hypothesis: any elevated vol above >=15 adds value to ATR-extreme."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i]) if i >= 0 else 16.0
    return 1 if vix_now >= 15 else 0


SIGNALS_P7N: List[Tuple] = [
    # Phase 7N — Ultra-quality combos + VIX extensions
    ('S143_s136_spy_bull',        sig_s143_s136_spy_bull,       10, 0.20, 50),
    ('S144_vix_recovery_strong',  sig_s144_vix_recovery_strong, 10, 0.20, 50),
    ('S145_s91_spy_run',          sig_s145_s91_spy_run,         10, 0.20, 50),
    ('S146_s91_tsla_down10d',     sig_s146_s91_tsla_down10d,    10, 0.20, 50),
    ('S147_s91_monday',           sig_s147_s91_monday,          10, 0.20, 50),
    ('S148_s91_month_start',      sig_s148_s91_month_start,     10, 0.20, 50),
    ('S149_s91_nr7_vix_rec',      sig_s149_s91_nr7_vix_recovery, 10, 0.20, 50),
    ('S150_s91_rsi_moderate',     sig_s150_s91_rsi_range,       10, 0.20, 50),
    ('S151_s136_nr7',             sig_s151_s136_nr7,            10, 0.20, 50),
    ('S152_s91_vix15',            sig_s152_s91_vix15,           10, 0.20, 50),
]


# ── Phase 7P — StatArb, ultra-combo, VIX sweep, NR extensions ─────────────────

def sig_s153_s152_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S153: S152 (S91 + VIX>=15) with 3-day hold.
    S152 is best total P&L signal ($3.125M). Test fast exit on it.
    Hypothesis: VIX>=15 captures more trades; 3d hold cuts tail-risk exposure."""
    return sig_s152_s91_vix15(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s154_s91_vix25(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S154: S91 + VIX >= 25 (panic zone threshold).
    VIX hierarchy: >=15 (S152), >=18 (S107), >=20 (S130), >=25 (S154), >=30.
    At VIX>=25, market is in near-panic. TSLA lags become extreme → biggest snaps.
    Hypothesis: VIX>=25 trades are the highest quality within the VIX filter family."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i]) if i >= 0 else 16.0
    return 1 if vix_now >= 25 else 0


def sig_s155_s152_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S155: S152 (S91 + VIX>=15) + NR7 — best P&L base + NR7 quality gate.
    S152 = $3.125M (85 trades). Adding NR7 selects the tightest-range subset.
    Hypothesis: NR7 within VIX>=15 environment = compression peak in recovery."""
    if sig_s152_s91_vix15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s156_s149_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S156: S149 (ATR extreme + NR7 + VIX recovery) with 3d hold.
    S149: WR=82%, PF=13.76, avg=$60K. Test fast exit on ultra-quality signal."""
    return sig_s149_s91_nr7_vix_recovery(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s157_statarb_residual(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S157: Statistical arbitrage — TSLA residual from rolling 20d OLS on SPY.
    Fit TSLA = alpha + beta*SPY over last 20 days.
    If TSLA is >2 std below predicted = extreme negative residual = snap-back.
    This is a proper z-score of the TSLA-SPY spread, more rigorous than S32's
    simple return comparison."""
    if i < 22:
        return 1
    # Rolling 20d OLS: TSLA_ret ~ SPY_ret
    tsla_ret = tsla['close'].iloc[i-20:i].pct_change().dropna().values
    spy_ret  = spy['close'].iloc[i-20:i].pct_change().dropna().values
    n = min(len(tsla_ret), len(spy_ret))
    if n < 15:
        return 1
    tsla_ret = tsla_ret[-n:]; spy_ret = spy_ret[-n:]
    # OLS beta
    cov  = np.cov(spy_ret, tsla_ret)
    if cov[0, 0] < 1e-12:
        return 1
    beta = cov[0, 1] / cov[0, 0]
    alpha = tsla_ret.mean() - beta * spy_ret.mean()
    # Residuals
    predicted = alpha + beta * spy_ret
    residuals = tsla_ret - predicted
    resid_std = residuals.std()
    if resid_std < 1e-10:
        return 1
    # Today's residual (use last 3d avg vs model)
    spy_3d   = float(spy['close'].iloc[i])   / float(spy['close'].iloc[i-3])   - 1
    tsla_3d  = float(tsla['close'].iloc[i])  / float(tsla['close'].iloc[i-3])  - 1
    today_resid = tsla_3d - (alpha * 3 + beta * spy_3d)
    z_score = today_resid / resid_std
    # Fire if z-score below -1.5 (TSLA significantly below predicted)
    return 1 if z_score < -1.5 else 0


def sig_s158_s91_statarb(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S158: S91 (ATR extreme) + statistical arbitrage z-score < -1.5.
    Combine the quantified ATR-extreme condition with the proper stat-arb residual.
    Both must agree: price is coiled AND statistically too far from predicted."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return sig_s157_statarb_residual(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s159_s151_spy_bull(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S159: S151 (VIX rec + compressed + NR7) + SPY > 200d MA.
    Ultra-quality S151 (PF=18.58) in bull market only.
    Hypothesis: 4/5 years already profitable; adding bull filter → 100% year coverage."""
    if sig_s151_s136_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 200:
        if float(spy['close'].iloc[i]) < float(spy['close'].iloc[i-200:i].mean()):
            return 0
    return 1


def sig_s160_s91_tsla_range_z(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S160: S91 + TSLA 5d range / 50d avg range < 0.5 (range Z-score compression).
    Normalize today's 5d range relative to 50d average range.
    Quantifies compression more precisely than ATR_5/ATR_20.
    Hypothesis: range z-score < 0.5 = top 25th percentile of compression."""
    if sig_s32_union_s29_s25(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 55:
        return 1
    highs  = tsla['high'].iloc[i-50:i+1].values.astype(float)
    lows   = tsla['low'].iloc[i-50:i+1].values.astype(float)
    daily_ranges = highs - lows
    range_5d  = daily_ranges[-5:].mean()
    range_50d = daily_ranges.mean()
    if range_50d < 1e-8:
        return 1
    range_ratio = range_5d / range_50d
    return 1 if range_ratio < 0.6 else 0


def sig_s161_s152_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S161: S152 (VIX>=15 + ATR extreme) + VIX was higher 5d ago (recovery).
    Add VIX recovery pattern to the high-P&L S152 signal.
    Hypothesis: VIX>=15 now AND VIX was higher 5d ago = recovery in progress."""
    if sig_s152_s91_vix15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i-5])
    return 1 if vix_now < vix_5d else 0  # VIX is falling (recovery)


def sig_s162_s91_vix30(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S162: S91 + VIX >= 30 (extreme fear threshold).
    Above VIX 30 = market crash conditions. TSLA lags become massive → biggest snaps.
    Very rare but potentially highest per-trade quality in the VIX sweep."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i]) if i >= 0 else 16.0
    return 1 if vix_now >= 30 else 0


SIGNALS_P7P: List[Tuple] = [
    # Phase 7P — StatArb + ultra-combo + VIX sweep
    ('S153_s152_hold3d',       sig_s153_s152_hold3d,        3, 0.20, 50),
    ('S154_s91_vix25',         sig_s154_s91_vix25,         10, 0.20, 50),
    ('S155_s152_nr7',          sig_s155_s152_nr7,          10, 0.20, 50),
    ('S156_s149_hold3d',       sig_s156_s149_hold3d,        3, 0.20, 50),
    ('S157_statarb_zscore',    sig_s157_statarb_residual,   5, 0.04, 50),
    ('S158_s91_statarb',       sig_s158_s91_statarb,       10, 0.20, 50),
    ('S159_s151_spy_bull',     sig_s159_s151_spy_bull,     10, 0.20, 50),
    ('S160_s91_range_z',       sig_s160_s91_tsla_range_z,  10, 0.20, 50),
    ('S161_s152_vix_rec',      sig_s161_s152_vix_rec,      10, 0.20, 50),
    ('S162_s91_vix30',         sig_s162_s91_vix30,         10, 0.20, 50),
]


# ── Phase 7Q — Beta-adjusted lag + persistence + best-signal combos ───────────

def sig_s163_beta_adjusted_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S163: Beta-adjusted SPY-TSLA divergence (proper beta calculation).
    Compute TSLA's rolling 60d beta vs SPY. If TSLA 3d return < beta × SPY 3d,
    it's underperforming EVEN relative to its own systematic risk.
    More rigorous than S32: accounts for TSLA's natural amplification factor."""
    if i < 62:
        return 1
    # Rolling 60d beta
    tsla_d = tsla['close'].iloc[i-60:i].pct_change().dropna().values
    spy_d  = spy['close'].iloc[i-60:i].pct_change().dropna().values
    n = min(len(tsla_d), len(spy_d))
    if n < 40:
        return 1
    tsla_d = tsla_d[-n:]; spy_d = spy_d[-n:]
    spy_var = spy_d.var()
    if spy_var < 1e-12:
        return 1
    beta = np.cov(spy_d, tsla_d)[0, 1] / spy_var
    # 3d actual vs beta-expected
    spy_3d  = float(spy['close'].iloc[i])  / float(spy['close'].iloc[i-3])  - 1
    tsla_3d = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i-3]) - 1
    expected = beta * spy_3d
    lag = expected - tsla_3d  # positive = TSLA below expected
    return 1 if lag > 0.02 else 0  # beta-adjusted lag > 2%


def sig_s164_s91_statarb_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S164: S91 + StatArb z-score + VIX>=18.
    Stack ATR-extreme + statistical residual + elevated vol.
    S158 (S91+StatArb) was 9/10 years. Adding VIX should improve quality."""
    if sig_s158_s91_statarb(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i]) if i >= 0 else 16.0
    return 1 if vix_now >= 18 else 0


def sig_s165_s153_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S165: S153 (best P&L signal: S91+VIX>=15+3d hold) + NR7 quality gate.
    S153 = $3.31M. Adding NR7 selects the tightest compression subset.
    Hypothesis: NR7 within VIX>=15 environment catches the exact peak."""
    if sig_s153_s152_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s166_s153_spy_bull(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S166: S153 (VIX>=15+3d hold) + SPY > 200d MA (macro bull filter).
    Remove bear-market trades from the best P&L signal.
    Hypothesis: 8/9 years → 9/9 years by removing 2022 bear market."""
    if sig_s153_s152_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 200:
        if float(spy['close'].iloc[i]) < float(spy['close'].iloc[i-200:i].mean()):
            return 0
    return 1


def sig_s167_s91_consecutive_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S167: S91 + S32 has fired 3+ consecutive days (persistent lag building).
    When SPY-TSLA divergence fires repeatedly, the lag is GROWING not resolving.
    A 3-consecutive-day signal = more pent-up divergence = stronger snap.
    Hypothesis: persistent multi-day lag setup → larger snap-back amplitude."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    # Check if S32 also fired on previous 2 bars
    def s32_at(j):
        return sig_s32_union_s29_s25(j, tsla, spy, vix, tw, sw, rt, rs, w)
    return 1 if (s32_at(i-1) == 1 and s32_at(i-2) == 1) else 0


def sig_s168_s91_spy_20d_high(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S168: S91 + SPY at or near its 20d high (SPY strong) while TSLA lags.
    When SPY is making new near-term highs but TSLA hasn't responded,
    the divergence is at its maximum strength point.
    Hypothesis: TSLA lag at SPY strength = highest probability snap."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    spy_now    = float(spy['close'].iloc[i])
    spy_20d_hi = float(spy['high'].iloc[i-20:i+1].max())
    # SPY within 2% of its 20d high
    return 1 if spy_now >= spy_20d_hi * 0.98 else 0


def sig_s169_s112_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S169: S112 (best volume-quality: S107+3d hold) + NR7.
    S112: $2.79M, PF=4.99, WR=76%, 67 trades. Adding NR7 → higher quality subset.
    Hypothesis: NR7 within VIX>=18 + ATR-extreme = triple validation."""
    if sig_s112_s107_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s170_s113_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S170: S113 (best quality: S108+3d hold, WR=82%) + NR7.
    S113: PF=7.69, WR=82%, 44 trades. Adding NR7 → even more selective.
    Hypothesis: SPY>200d + VIX>=18 + ATR-extreme + NR7 = maximum quality."""
    if sig_s113_s108_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s171_s152_spy_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S171: S152 (VIX>=15) + SPY > 200d MA + NR7 — triple filter on best P&L.
    Stacking macro bull + compression pattern onto the high-frequency signal."""
    if sig_s152_s91_vix15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 200:
        if float(spy['close'].iloc[i]) < float(spy['close'].iloc[i-200:i].mean()):
            return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s172_s91_vix_just_20(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S172: S91 + VIX just dropped below 20 (crossing from fear to normal).
    The moment VIX crosses below 20 = fear exiting market = risk-on.
    Yesterday VIX >= 20, today VIX < 20 → fear-to-calm transition.
    Hypothesis: VIX normalization crossover = optimal lag-closure catalyst."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 1
    vix_now  = float(vix['close'].iloc[i])
    vix_prev = float(vix['close'].iloc[i-1])
    return 1 if (vix_prev >= 20 and vix_now < 20) else 0


def sig_s173_s91_beta_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S173: S91 + beta-adjusted lag > 1.5% (combine ATR with beta divergence).
    S163 (beta-adjusted lag alone): tests the divergence metric.
    S173 combines it with ATR-extreme for dual confirmation."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return sig_s163_beta_adjusted_lag(i, tsla, spy, vix, tw, sw, rt, rs, w)


SIGNALS_P7Q: List[Tuple] = [
    # Phase 7Q — Beta lag + persistence + combos
    ('S163_beta_lag',          sig_s163_beta_adjusted_lag,  5, 0.04, 50),
    ('S164_s91_statarb_vix',   sig_s164_s91_statarb_vix,  10, 0.20, 50),
    ('S165_s153_nr7',          sig_s165_s153_nr7,           3, 0.20, 50),
    ('S166_s153_spy_bull',     sig_s166_s153_spy_bull,      3, 0.20, 50),
    ('S167_s91_consec_lag',    sig_s167_s91_consecutive_lag, 10, 0.20, 50),
    ('S168_s91_spy_strong',    sig_s168_s91_spy_20d_high,  10, 0.20, 50),
    ('S169_s112_nr7',          sig_s169_s112_nr7,           3, 0.20, 50),
    ('S170_s113_nr7',          sig_s170_s113_nr7,           3, 0.20, 50),
    ('S171_s152_spy_nr7',      sig_s171_s152_spy_nr7,       3, 0.20, 50),
    ('S172_s91_vix_cross20',   sig_s172_s91_vix_just_20,  10, 0.20, 50),
    ('S173_s91_beta_lag',      sig_s173_s91_beta_lag,      10, 0.20, 50),
]


# ── Phase 7R — Bollinger, return z-score, vol ratio, MACD ────────────────────

def sig_s174_s91_lower_bb(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S174: S91 + TSLA close near or below its lower Bollinger Band (20d, 2σ).
    Lower BB = classic mean-reversion entry: price statistically oversold.
    Combines BB oversold condition with ATR-extreme for dual confirmation.
    Hypothesis: BB-touch + ATR-extreme = maximum mean-reversion setup."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    closes = tsla['close'].iloc[i-19:i+1].values.astype(float)
    mu  = closes.mean()
    std = closes.std()
    bb_lower = mu - 2 * std
    tsla_c   = float(tsla['close'].iloc[i])
    return 1 if tsla_c <= bb_lower * 1.02 else 0  # within 2% of lower BB


def sig_s175_s91_return_zscore(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S175: S91 + TSLA 3d return z-score below -1.5 (statistically extreme drop).
    Normalize TSLA's 3d return against its 60d distribution of 3d returns.
    Z < -1.5 means this 3d decline is in the bottom 7% of historical 3d moves.
    Hypothesis: statistical extreme drop + ATR extreme = strongest snap-back."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 65:
        return 1
    # Rolling 3d returns over last 60 days
    ret_3d_series = []
    for j in range(i-60, i-2):
        r = float(tsla['close'].iloc[j+3]) / float(tsla['close'].iloc[j]) - 1
        ret_3d_series.append(r)
    if len(ret_3d_series) < 30:
        return 1
    arr  = np.array(ret_3d_series)
    mu   = arr.mean(); std = arr.std()
    if std < 1e-10:
        return 1
    today_3d = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i-3]) - 1
    z = (today_3d - mu) / std
    return 1 if z < -1.5 else 0


def sig_s176_s91_tsla_calm_vs_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S176: S91 + TSLA realized vol (10d) < VIX (TSLA calmer than market).
    VIX reflects expected SPY vol. When TSLA realized vol < VIX, the market
    is MORE fearful than TSLA actually is — TSLA has been suppressed.
    Hypothesis: market fear > TSLA actual vol = TSLA is being held down artificially."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    rets = tsla['close'].iloc[i-10:i+1].pct_change().dropna().values
    if len(rets) < 5:
        return 1
    tsla_rvol_ann = rets.std() * np.sqrt(252) * 100  # annualized %
    vix_level = float(vix['close'].iloc[i])
    return 1 if tsla_rvol_ann < vix_level else 0


def sig_s177_s91_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S177: S91 + TSLA MACD histogram turning up (momentum shift to bullish).
    MACD(12,26,9): if the histogram was negative but is now less negative (turning),
    short-term momentum is inflecting upward while ATR is at extreme.
    Hypothesis: MACD turn + ATR extreme = momentum + mean-reversion double signal."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 27:
        return 1
    closes = tsla['close'].iloc[i-30:i+1].values.astype(float)
    # EMA helper
    def ema(data, n):
        e = data[0]
        k = 2 / (n + 1)
        for x in data[1:]:
            e = x * k + e * (1 - k)
        return e
    ema12_now  = ema(closes[-13:], 12)
    ema26_now  = ema(closes[-27:], 26)
    ema12_prev = ema(closes[-14:-1], 12)
    ema26_prev = ema(closes[-28:-1], 26)
    macd_now  = ema12_now  - ema26_now
    macd_prev = ema12_prev - ema26_prev
    # MACD was negative but is improving (histogram turning up)
    return 1 if (macd_prev < 0 and macd_now > macd_prev) else 0


def sig_s178_s91_consec_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S178: S91 + TSLA has had 3+ consecutive down closes (exhaustion pattern).
    After 3 consecutive down days, selling exhaustion often sets in.
    Combined with ATR-extreme (low vol = structured selling, not panic).
    Hypothesis: quiet 3-day decline + ATR extreme = exhaustion snap-back."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    for j in range(3):
        if float(tsla['close'].iloc[i-j]) >= float(tsla['close'].iloc[i-j-1]):
            return 0
    return 1


def sig_s179_s152_nr7_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S179: S152 (VIX>=15+ATR extreme) + NR7 + VIX recovery — best combo.
    Ultimate stacking: high P&L base (S152) + quality filter (NR7) + timing (VIX rec).
    Hypothesis: all three conditions = maximum compression + recovery setup."""
    if sig_s152_s91_vix15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _nr_n(tsla, i, 7):
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i-5])
    return 1 if vix_now < vix_5d * 0.95 else 0  # VIX dropped >=5% in 5d


def sig_s180_s153_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S180: S153 (best P&L: VIX>=15+3d hold) + VIX recovery.
    Add the recovery timing to the highest-P&L signal.
    Hypothesis: best-frequency signal + recovery timing = large P&L + quality."""
    if sig_s153_s152_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i-5])
    return 1 if vix_now < vix_5d * 0.95 else 0


SIGNALS_P7R: List[Tuple] = [
    # Phase 7R — Bollinger, return z-score, vol ratio, momentum
    ('S174_s91_lower_bb',       sig_s174_s91_lower_bb,       10, 0.20, 50),
    ('S175_s91_return_zscore',  sig_s175_s91_return_zscore,  10, 0.20, 50),
    ('S176_s91_tsla_calm',      sig_s176_s91_tsla_calm_vs_vix, 10, 0.20, 50),
    ('S177_s91_macd_turn',      sig_s177_s91_macd_turning,   10, 0.20, 50),
    ('S178_s91_consec_down',    sig_s178_s91_consec_down,    10, 0.20, 50),
    ('S179_s152_nr7_vix_rec',   sig_s179_s152_nr7_vix_rec,  10, 0.20, 50),
    ('S180_s153_vix_rec',       sig_s180_s153_vix_rec,        3, 0.20, 50),
]


# ── Phase 7S — January Effect (First 5 / First 3 trading days of year) ────────
# Theory: if SPY's first 5 (or 3) trading days of the year close UP vs prior
# year-end, the full year tends to be bullish. Filter existing signals by this.
# Implemented strictly with NO look-ahead: signal is inactive until after day 5
# (or day 3) of the new year has closed.

def _jan_effect_n(spy: pd.DataFrame, bar_date, n_days: int) -> bool:
    """True if SPY closed UP over first n_days trading days of bar_date's year.
    Returns False if we haven't yet passed day n yet (strict no look-ahead).
    Returns True if data is unavailable (conservative default = allow trades)."""
    year = bar_date.year
    year_bars = spy[spy.index.year == year]
    if len(year_bars) < n_days:
        return True  # Too early in year or no data — default allow
    day_n_date = year_bars.index[n_days - 1]
    if bar_date < day_n_date:
        return False  # Haven't reached day N yet — suspend trading
    # Compare day-N close to prior year's last close
    prev_bars = spy[spy.index.year == year - 1]
    if len(prev_bars) == 0:
        return True
    prev_close = float(prev_bars['close'].iloc[-1])
    day_n_close = float(year_bars['close'].iloc[n_days - 1])
    return day_n_close > prev_close


def sig_s181_jan5d_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S181: S91 (ATR extreme) only in bull years per SPY first-5-day rule.
    January Effect: year predicted bullish if days 1-5 close UP vs Dec 31.
    Hypothesis: regime filter eliminates losing years entirely."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _jan_effect_n(spy, tsla.index[i], 5) else 0


def sig_s182_jan3d_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S182: S91 only in bull years per SPY first-3-day rule (tighter).
    3-day variant: quicker signal, resumes trading sooner in year.
    Hypothesis: shorter lookback = less predictive but more trades."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _jan_effect_n(spy, tsla.index[i], 3) else 0


def sig_s183_jan5d_s91_bear(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S183: S91 ONLY in BEAR years (SPY first 5 days DOWN) — inverse filter.
    Test: does the signal still work even in bearish-opening years?
    Or does it *only* work in bull years? This proves/disproves the theory."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    year = tsla.index[i].year
    year_bars = spy[spy.index.year == year]
    if len(year_bars) < 5:
        return 0  # Not yet past day 5
    day5_date = year_bars.index[4]
    if tsla.index[i] < day5_date:
        return 0  # Before day 5
    prev_bars = spy[spy.index.year == year - 1]
    if len(prev_bars) == 0:
        return 0
    prev_close = float(prev_bars['close'].iloc[-1])
    day5_close = float(year_bars['close'].iloc[4])
    return 1 if day5_close <= prev_close else 0  # Bear year


def sig_s184_jan5d_s113(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S184: S113 (SPY>200d+VIX>=18+ATR, 3d hold) + Jan-5d bull-year filter.
    Apply best quality signal (S113: PF=7.69, WR=82%) only in bull-opening years.
    Hypothesis: already high-quality signal becomes nearly perfect with year filter."""
    if sig_s113_s108_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _jan_effect_n(spy, tsla.index[i], 5) else 0


def sig_s185_jan5d_s136(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S185: S136 (VIX recovery + compressed, PF=16.81) + Jan-5d bull-year filter.
    Apply highest-PF signal only in bull-opening years.
    Hypothesis: if the filter eliminates bear years, PF could approach perfection."""
    if sig_s136_s132_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _jan_effect_n(spy, tsla.index[i], 5) else 0


SIGNALS_P7S: List[Tuple] = [
    # Phase 7S — January Effect year filter (first 5 and 3 trading days)
    ('S181_jan5d_s91',          sig_s181_jan5d_s91,          10, 0.20, 50),
    ('S182_jan3d_s91',          sig_s182_jan3d_s91,          10, 0.20, 50),
    ('S183_jan5d_s91_bear',     sig_s183_jan5d_s91_bear,     10, 0.20, 50),
    ('S184_jan5d_s113',         sig_s184_jan5d_s113,          3, 0.20, 50),
    ('S185_jan5d_s136',         sig_s185_jan5d_s136,          2, 0.20, 50),
]


# ── Phase 7T — Novel compound signals: 52wk proximity, volume climax, persistent compression ─
# Exploring three new dimensions:
# 1. Price level context: 52-week low proximity (maximum pessimism zone)
# 2. Volume context: selling climax (yesterday high volume = exhaustion signal)
# 3. Temporal ATR: persistent compression (ATR compressed for 3+ consecutive days)
# 4. Relative weakness: TSLA lagging SPY by >20% over 30d (catch-up trade)

def _52wk_low_proximity(tsla: pd.DataFrame, i: int, pct: float) -> bool:
    """True if TSLA close is within pct% of its 52-week low."""
    lookback = min(i, 252)
    if lookback < 20:
        return False
    low_52wk = float(tsla['low'].iloc[i - lookback:i].min())
    close = float(tsla['close'].iloc[i])
    return close <= low_52wk * (1 + pct)


def sig_s186_52wk_low_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S186: S91 (ATR extreme) + TSLA within 25% of its 52-week low.
    Maximum pessimism zone: stock near yearly low AND ATR compressed = quiet capitulation.
    Hypothesis: deepest price levels + low volatility = institutional accumulation zone."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _52wk_low_proximity(tsla, i, 0.25) else 0


def sig_s187_52wk_low_s112(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S187: S112 (VIX>=18 + ATR extreme, 3d hold) + within 30% of 52-week low.
    Higher-quality base signal (S112 WR=76%) at maximum pessimism levels.
    Hypothesis: elevated fear + quiet dip near yearly low = best buy setup."""
    if sig_s112_s107_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _52wk_low_proximity(tsla, i, 0.30) else 0


def sig_s188_volume_climax_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S188: S91 + prior-day volume was 2x+ the 20-day average (selling climax).
    Selling climax theory: extreme volume on down day = exhaustion of sellers.
    Combined with compressed ATR today = sellers gave up, quiet consolidation follows."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 21 or 'volume' not in tsla.columns:
        return 1
    vol_yesterday = float(tsla['volume'].iloc[i - 1])
    vol_20d_avg = float(tsla['volume'].iloc[i - 21:i - 1].mean())
    if vol_20d_avg < 1:
        return 1
    return 1 if vol_yesterday >= 2.0 * vol_20d_avg else 0


def sig_s189_persistent_compression(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S189: S91 + ATR has been compressed for 3+ consecutive trading days.
    Persistent compression = the coiling gets tighter, snap-back becomes more violent.
    Single-day ATR extreme is good; multi-day confirmation is stronger."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # Check last 3 days all had compressed ATR (atr_5 < 0.80 * atr_20)
    for lag in range(1, 3):
        c = _atr_components(tsla, i - lag)
        if c is None:
            return 1
        _, atr_5, _, atr_20 = c
        if atr_5 >= 0.80 * atr_20:
            return 0  # Not persistently compressed
    return 1


def sig_s190_tsla_lags_spy_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S190: S91 + TSLA lagging SPY by >15% over last 30 days (mean-reversion catch-up).
    When SPY is up X% but TSLA is flat/down, TSLA is unusually weak relative to market.
    ATR extreme + relative weakness = higher-beta stock coiled for catch-up rally."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 30:
        return 1
    tsla_ret_30d = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - 30]) - 1
    spy_ret_30d  = float(spy['close'].iloc[i])  / float(spy['close'].iloc[i - 30])  - 1
    # SPY up at least 3% AND TSLA lagging by 15%+ (e.g., SPY +8%, TSLA -7%)
    return 1 if (spy_ret_30d > 0.03 and tsla_ret_30d < spy_ret_30d - 0.15) else 0


def sig_s191_persistent_comp_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S191: Persistent compression (S189) + VIX>=18.
    ATR coiling for 3+ days WITH elevated VIX = market scared but TSLA going quiet.
    The divergence between market fear and TSLA calm = prime snap-back setup."""
    if sig_s189_persistent_compression(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i]) if i >= 0 else 16.0
    return 1 if vix_now >= 18 else 0


def sig_s192_52wk_low_s136(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S192: S136 (VIX recovery + compressed ATR, PF=16.81) + near 52-week low.
    Best-PF signal at maximum pessimism levels.
    Hypothesis: VIX recovery + compressed + near yearly low = triple confirmation."""
    if sig_s136_s132_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _52wk_low_proximity(tsla, i, 0.30) else 0


SIGNALS_P7T: List[Tuple] = [
    # Phase 7T — 52wk proximity, volume climax, persistent compression, relative weakness
    ('S186_52wk_low_s91',       sig_s186_52wk_low_s91,       10, 0.20, 50),
    ('S187_52wk_low_s112',      sig_s187_52wk_low_s112,       3, 0.20, 50),
    ('S188_volume_climax_s91',  sig_s188_volume_climax_s91,  10, 0.20, 50),
    ('S189_persist_compress',   sig_s189_persistent_compression, 10, 0.20, 50),
    ('S190_tsla_lags_spy',      sig_s190_tsla_lags_spy_s91,  10, 0.20, 50),
    ('S191_persist_comp_vix',   sig_s191_persistent_comp_vix, 10, 0.20, 50),
    ('S192_52wk_s136',          sig_s192_52wk_low_s136,       2, 0.20, 50),
]


# ── Phase 7U — Extensions of winning Phase 7T signals + seasonality ──────────
# Extending the best performers: volume climax (S188), persistent compression (S189/S191).
# Also testing monthly seasonality — TSLA historically strong Q4+Jan.

def sig_s193_volume_climax_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S193: Volume climax (S188) + VIX>=18 — quality upgrade.
    Selling climax + compressed ATR + elevated VIX = maximum panic exhaustion setup.
    S188 was 6/6 years; adding VIX filter should raise WR while keeping year consistency."""
    if sig_s188_volume_climax_s91(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i]) if i >= 0 else 16.0
    return 1 if vix_now >= 18 else 0


def sig_s194_volume_climax_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S194: Volume climax (S188) + NR7 — rarest, highest-quality variant.
    Yesterday's extreme volume + today's narrowest range of 7 bars = perfect setup.
    Volume exhaustion then price compression = coil after climax."""
    if sig_s188_volume_climax_s91(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s195_persist_comp_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S195: Persistent compression (S189) + VIX recovery — dual compression types.
    ATR has been compressed 3+ days AND VIX has been cooling (spike then calm).
    Hypothesis: price quiet + fear leaving = double snap-back pressure."""
    if sig_s189_persistent_compression(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    spike_then = vix_5d > 20 or vix_10d > 20
    calm_now   = vix_now < vix_5d * 0.90
    return 1 if (spike_then and calm_now) else 0


def sig_s196_persist_comp_spy200(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S196: Persistent compression (S189) + SPY above 200d MA — bull market filter.
    Only take persistent-compression setups when the broad market is in uptrend.
    Removes the 2019 losing year (market correction) from the S189 results."""
    if sig_s189_persistent_compression(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_200d  = float(spy['close'].iloc[i - 200:i].mean())
    return 1 if spy_close > spy_200d else 0


# Monthly seasonality helper
_TSLA_BULL_MONTHS = {10, 11, 12, 1}   # Oct, Nov, Dec, Jan historically strong


def sig_s197_seasonality_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S197: S91 only in TSLA's historically strong months (Oct/Nov/Dec/Jan).
    Q4 deliveries + year-end tax-loss buying reversal + New Year optimism.
    Hypothesis: signal quality highest when seasonal tailwind is active."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    month = tsla.index[i].month
    return 1 if month in _TSLA_BULL_MONTHS else 0


def sig_s198_seasonality_s113(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S198: S113 (best quality: VIX>=18+SPY>200d+ATR, 3d hold) in bull months only.
    Apply the highest-WR signal (82%) to seasonally strong months.
    Hypothesis: dual tailwind → near-perfect win rate."""
    if sig_s113_s108_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    month = tsla.index[i].month
    return 1 if month in _TSLA_BULL_MONTHS else 0


def sig_s199_s191_3d_hold(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S199: S191 (persistent compression + VIX>=18) with 3-day hold.
    S191 was WR=82%, PF=4.38. Test if 3d hold improves further (same as S113 benefit).
    Hold until signal reversal OR 3 days, whichever first."""
    return sig_s191_persistent_comp_vix(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s200_persist_comp_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S200: Persistent compression + NR7 — signal #200!
    3+ days of compressed ATR culminating in the NARROWEST day of the last 7.
    Hypothesis: multi-day coiling peak = maximum snap-back potential.
    The Toby Crabel NR7 on already-compressed stock = rare but powerful."""
    if sig_s189_persistent_compression(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


SIGNALS_P7U: List[Tuple] = [
    # Phase 7U — Volume climax extensions, persistent compression variants, seasonality
    ('S193_vol_climax_vix',     sig_s193_volume_climax_vix,  10, 0.20, 50),
    ('S194_vol_climax_nr7',     sig_s194_volume_climax_nr7,  10, 0.20, 50),
    ('S195_persist_vix_rec',    sig_s195_persist_comp_vix_rec, 10, 0.20, 50),
    ('S196_persist_spy200',     sig_s196_persist_comp_spy200, 10, 0.20, 50),
    ('S197_season_s91',         sig_s197_seasonality_s91,    10, 0.20, 50),
    ('S198_season_s113',        sig_s198_seasonality_s113,    3, 0.20, 50),
    ('S199_s191_hold3d',        sig_s199_s191_3d_hold,       10, 0.20, 50),
    ('S200_persist_nr7',        sig_s200_persist_comp_nr7,   10, 0.20, 50),
]


# ── Phase 7V — Extending S195 + Doji + RSI divergence + relaxed persistence ──
# S195 (WR=88%, PF=28.79) is the best signal found. Extend it:
# 1. Add NR7 filter to S195 (triple compound)
# 2. Relax persistence threshold (2 days instead of 3 = more trades)
# 3. Raise VIX threshold for S195 (>=20 vs >=18)
# 4. Doji candle + ATR extreme (small body = market indecision)
# 5. RSI bullish divergence + ATR extreme (price lower, RSI higher)
# 6. S195 variant: use 5-day VIX recovery window instead of 5-10d
# 7. S113 + seasonality (best quality in best months)

def sig_s201_s195_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S201: S195 (persistent compress + VIX recovery) + NR7 — 4-way compound.
    Persistent ATR coiling + VIX fear exiting + narrowest range = ultimate setup.
    Hypothesis: all four conditions = maximum snap-back probability."""
    if sig_s195_persist_comp_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s202_relax_persist_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S202: 2-day persistent compression + VIX recovery (relaxed S195).
    S195 requires 3 days; this requires only 2 → more trades, slightly lower bar.
    Hypothesis: even 2-day coiling before recovery = strong signal."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # Check last 2 days had compressed ATR (relaxed from 3 in S189)
    for lag in range(1, 2):
        c = _atr_components(tsla, i - lag)
        if c is None:
            return 1
        _, atr_5, _, atr_20 = c
        if atr_5 >= 0.80 * atr_20:
            return 0
    # VIX recovery
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    spike_then = vix_5d > 20 or vix_10d > 20
    calm_now   = vix_now < vix_5d * 0.90
    return 1 if (spike_then and calm_now) else 0


def sig_s203_doji_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S203: S91 (ATR extreme) + Doji candle (body < 0.3% of price).
    Doji = open and close are nearly equal = market indecision after a move.
    ATR compressed + Doji = even the volatility within the day dried up.
    Hypothesis: maximum indecision in compressed environment → directional break."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    open_p  = float(tsla['open'].iloc[i])
    close_p = float(tsla['close'].iloc[i])
    if close_p < 1:
        return 1
    body_pct = abs(close_p - open_p) / close_p
    return 1 if body_pct < 0.003 else 0  # Body < 0.3% of price = Doji


def sig_s204_rsi_diverge_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S204: S91 + RSI bullish divergence (price lower than 10d ago but RSI higher).
    Bullish divergence: price making lower lows while RSI makes higher lows.
    Combined with ATR extreme = quiet accumulation phase with hidden momentum.
    Hypothesis: compressed + divergence = smart money buying into weakness."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 24:
        return 1
    # Calculate RSI(14) now and 10 bars ago
    def _rsi(prices):
        changes = [prices[j] - prices[j-1] for j in range(1, len(prices))]
        gains = [max(c, 0) for c in changes]
        losses = [max(-c, 0) for c in changes]
        avg_gain = sum(gains) / len(gains)
        avg_loss = sum(losses) / len(losses)
        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)
    closes_now  = [float(tsla['close'].iloc[i - 14 + j]) for j in range(15)]
    closes_prev = [float(tsla['close'].iloc[i - 24 + j]) for j in range(15)]
    rsi_now  = _rsi(closes_now)
    rsi_prev = _rsi(closes_prev)
    price_now  = float(tsla['close'].iloc[i])
    price_prev = float(tsla['close'].iloc[i - 10])
    # Divergence: price lower but RSI higher = bullish
    return 1 if (price_now < price_prev and rsi_now > rsi_prev) else 0


def sig_s205_s195_vix20(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S205: S195 (persistent compress + VIX recovery) requiring VIX>=20 at spike.
    S195 requires VIX was >20 in the spike. This makes it more explicit.
    This is nearly identical to S195 but clarifies the VIX threshold is >=20."""
    if sig_s189_persistent_compression(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    spike_then = vix_5d >= 22 or vix_10d >= 22  # Stricter: VIX>=22 (vs >20)
    calm_now   = vix_now < vix_5d * 0.88        # Stricter: dropped >=12% (vs 10%)
    return 1 if (spike_then and calm_now) else 0


def sig_s206_consec_down_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S206: 3+ consecutive down days (S178 exhaustion) + VIX recovery.
    Sellers exhausted (3 down days) AND fear abating (VIX cooling) = reversal.
    Different from S195: uses price exhaustion instead of ATR compression.
    Hypothesis: behavioral exhaustion + macro fear dropping = buy signal."""
    if i < 10:
        return 1
    # 3 consecutive down closes (exhaustion)
    for j in range(3):
        if float(tsla['close'].iloc[i - j]) >= float(tsla['close'].iloc[i - j - 1]):
            return 0
    # VIX recovery
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    spike_then = vix_5d > 20 or vix_10d > 20
    calm_now   = vix_now < vix_5d * 0.90
    return 1 if (spike_then and calm_now) else 0


def sig_s207_s113_season(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S207: S113 (best quality: SPY>200d+VIX>=18+ATR, 3d hold) + bull months.
    S113 WR=82%, PF=7.69. Apply in TSLA's historically strong months (Oct-Jan).
    Hypothesis: best-in-class signal + seasonal tailwind = near-perfect results."""
    if sig_s113_s108_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    month = tsla.index[i].month
    return 1 if month in _TSLA_BULL_MONTHS else 0


SIGNALS_P7V: List[Tuple] = [
    # Phase 7V — S195 extensions, Doji, RSI divergence, relaxed persistence
    ('S201_s195_nr7',           sig_s201_s195_nr7,           10, 0.20, 50),
    ('S202_relax_persist_vxr',  sig_s202_relax_persist_vix_rec, 10, 0.20, 50),
    ('S203_doji_s91',           sig_s203_doji_s91,           10, 0.20, 50),
    ('S204_rsi_div_s91',        sig_s204_rsi_diverge_s91,    10, 0.20, 50),
    ('S205_s195_vix22',         sig_s205_s195_vix20,          10, 0.20, 50),
    ('S206_consec_down_vxr',    sig_s206_consec_down_vix_rec, 10, 0.20, 50),
    ('S207_s113_season',        sig_s207_s113_season,          3, 0.20, 50),
]


# ── Phase 7W — S206 extensions + Hammer candle + gap reversal + multi-TF ─────
# Extending the S206 (consecutive down + VIX recovery) signal family.
# Also: classical candlestick reversal patterns that haven't been tested.

def sig_s208_s206_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S208: S206 (3 consec down + VIX recovery) + NR7.
    After 3 down days the range is also narrowest of the week = maximum exhaustion.
    Hypothesis: S206 was WR=80%, PF=12.49. Adding NR7 should push WR even higher."""
    if sig_s206_consec_down_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s209_s206_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S209: S206 (3 consec down + VIX recovery) + compressed ATR.
    3 consecutive down days with quiet volatility (not a panic crash) + VIX cooling.
    Hypothesis: calm orderly selling exhaustion + VIX recovery = cleaner reversal."""
    if sig_s206_consec_down_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.80 * atr_20 else 0


def sig_s210_s206_spy200(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S210: S206 + SPY above 200d MA (bull market filter).
    Only trade consecutive-down-days exhaustion when overall market is in uptrend.
    Hypothesis: removes bear market instances where bounces are weaker."""
    if sig_s206_consec_down_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_200d  = float(spy['close'].iloc[i - 200:i].mean())
    return 1 if spy_close > spy_200d else 0


def sig_s211_hammer_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S211: S91 (ATR extreme) + Hammer candle (long lower wick, small body at top).
    Hammer: lower wick >= 2x body AND body in upper 1/3 of range.
    Hypothesis: long wick shows rejection of lower prices + ATR compressed = reversal."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    op = float(tsla['open'].iloc[i])
    cl = float(tsla['close'].iloc[i])
    hi = float(tsla['high'].iloc[i])
    lo = float(tsla['low'].iloc[i])
    total_range = hi - lo
    if total_range < 0.001:
        return 0
    body = abs(cl - op)
    lower_wick = min(op, cl) - lo
    upper_wick = hi - max(op, cl)
    # Hammer: lower wick >= 2x body, upper wick small, bullish close
    if body < 0.0001:
        return 0
    return 1 if (lower_wick >= 2 * body and upper_wick <= 0.5 * body and cl > op) else 0


def sig_s212_gap_down_recovery(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S212: TSLA gapped down >1.5% at open (fear) but closed UP vs prior day.
    Gap down then recovery = intraday reversal = buyers overwhelmed initial sellers.
    Combined with VIX elevated = macro fear was the driver, already correcting."""
    if i < 5:
        return 1
    today_open  = float(tsla['open'].iloc[i])
    prev_close  = float(tsla['close'].iloc[i - 1])
    today_close = float(tsla['close'].iloc[i])
    gap_down_pct = (today_open - prev_close) / prev_close
    recovery = today_close > prev_close
    if not (gap_down_pct < -0.015 and recovery):
        return 0
    vix_now = float(vix['close'].iloc[i])
    return 1 if vix_now >= 18 else 0


def sig_s213_4_consec_down_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S213: 4 consecutive down closes (extreme exhaustion) + VIX recovery.
    Stricter version of S206 (3 days → 4 days) = rarer but higher-conviction.
    Hypothesis: 4-day orderly decline + fear cooling = even stronger reversal."""
    if i < 14:
        return 1
    for j in range(4):
        if float(tsla['close'].iloc[i - j]) >= float(tsla['close'].iloc[i - j - 1]):
            return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    spike_then = vix_5d > 20 or vix_10d > 20
    calm_now   = vix_now < vix_5d * 0.90
    return 1 if (spike_then and calm_now) else 0


def sig_s214_weekly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S214: Multi-TF alignment — weekly TSLA near lower channel + daily ATR extreme.
    When weekly chart is also at support AND daily is quiet = dual-timeframe setup.
    Uses the weekly bars (tw) to check channel position."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if tw is None or len(tw) < 25:
        return 1
    daily_date = tsla.index[i]
    weekly_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if weekly_idx < 20:
        return 1
    ch = _channel_at(tw.iloc[weekly_idx - 20:weekly_idx])
    if ch is None:
        return 1
    wk_close = float(tw['close'].iloc[weekly_idx])
    return 1 if _near_lower(wk_close, ch, 0.30) else 0


SIGNALS_P7W: List[Tuple] = [
    # Phase 7W — S206 extensions + hammer candle + gap recovery + multi-TF
    ('S208_s206_nr7',           sig_s208_s206_nr7,           10, 0.20, 50),
    ('S209_s206_compressed',    sig_s209_s206_compressed,    10, 0.20, 50),
    ('S210_s206_spy200',        sig_s210_s206_spy200,        10, 0.20, 50),
    ('S211_hammer_s91',         sig_s211_hammer_s91,         10, 0.20, 50),
    ('S212_gap_down_recovery',  sig_s212_gap_down_recovery,  10, 0.20, 50),
    ('S213_4down_vix_rec',      sig_s213_4_consec_down_vix,  10, 0.20, 50),
    ('S214_weekly_low_daily',   sig_s214_weekly_low_daily_s91, 10, 0.20, 50),
]


# ── Phase 7X — S214 extensions (multi-TF breakthrough) + S210 enhancements ───
# S214 (weekly near lower channel + daily ATR extreme) hit $2.16M, 9/10 years.
# This is the best consistency of any new signal. Extend it aggressively.

def sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S215: S214 (multi-TF) + VIX>=18.
    Multi-TF weekly support + daily ATR extreme + elevated VIX = triple confirmation.
    S214 was 9/10 years; VIX filter should remove the one losing year (2019).
    Hypothesis: VIX>=18 during weekly support visit = institutional accumulation."""
    if sig_s214_weekly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i])
    return 1 if vix_now >= 18 else 0


def sig_s216_s214_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S216: S214 (multi-TF) + VIX recovery (spike-then-calm).
    Weekly support + daily ATR extreme + VIX cooling from elevated levels.
    Hypothesis: the one losing year (2019 -$94K) had no VIX spike → this removes it."""
    if sig_s214_weekly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    spike_then = vix_5d > 20 or vix_10d > 20
    calm_now   = vix_now < vix_5d * 0.90
    return 1 if (spike_then and calm_now) else 0


def sig_s217_s214_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S217: S214 (multi-TF) + NR7.
    Weekly support + daily ATR extreme + narrowest range of last 7 days.
    Hypothesis: weekly and daily both at compression peak = maximum coil."""
    if sig_s214_weekly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s218_s214_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S218: S214 (multi-TF) with 3-day hold — test if holding longer improves.
    S214 avg hold was 2.5d; try 3d max hold with same signal.
    Hypothesis: weekly support visits take a few days to develop fully."""
    return sig_s214_weekly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s219_s210_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S219: S210 (3-consec-down + VIX recovery + SPY>200d) + NR7.
    S210 was 5/5 years, WR=83%, PF=27.88. Adding NR7 = rareer but higher-conviction.
    Hypothesis: exhaustion + VIX calm + bull market + narrowest range = perfect setup."""
    if sig_s210_s206_spy200(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s220_s214_spy200(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S220: S214 (multi-TF) + SPY>200d MA.
    Multi-TF weekly support + daily ATR extreme + broad market in uptrend.
    Should remove 2019 miss (market was in correction) while keeping bull years."""
    if sig_s214_weekly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_200d  = float(spy['close'].iloc[i - 200:i].mean())
    return 1 if spy_close > spy_200d else 0


def sig_s221_s214_vix18_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S221: S215 (S214+VIX>=18) with 3-day hold.
    Multi-TF + elevated VIX + 3d hold = test of optimal hold period.
    Hypothesis: fear-driven setups need a day or two to resolve = longer hold better."""
    return sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w)


SIGNALS_P7X: List[Tuple] = [
    # Phase 7X — S214 multi-TF extensions + S210 enhancements
    ('S215_s214_vix18',         sig_s215_s214_vix18,         10, 0.20, 50),
    ('S216_s214_vix_rec',       sig_s216_s214_vix_rec,       10, 0.20, 50),
    ('S217_s214_nr7',           sig_s217_s214_nr7,           10, 0.20, 50),
    ('S218_s214_hold3d',        sig_s218_s214_hold3d,         3, 0.20, 50),
    ('S219_s210_nr7',           sig_s219_s210_nr7,           10, 0.20, 50),
    ('S220_s214_spy200',        sig_s220_s214_spy200,        10, 0.20, 50),
    ('S221_s215_hold3d',        sig_s221_s214_vix18_hold3d,   3, 0.20, 50),
]


# ── Phase 7Y — Oversold-vs-MA, end-of-quarter, month-end, S215 compressed ────
# Exploring: deeply oversold vs MA + compressed, end-of-quarter effects,
# month-end/month-start patterns, and S215 compressed-only variant.

def sig_s222_below_50ma_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S222: S91 (ATR extreme) + TSLA >15% below its 50-day MA.
    Deep MA undershoot = extreme negative sentiment vs trend.
    ATR compressed at this level = institutional accumulation, not panic.
    Hypothesis: deeply oversold vs trend + quiet = best value entry."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    close = float(tsla['close'].iloc[i])
    ma50 = float(tsla['close'].iloc[i - 50:i].mean())
    return 1 if close < ma50 * 0.85 else 0  # >15% below 50d MA


def sig_s223_below_50ma_s215(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S223: S215 (multi-TF + VIX>=18) + TSLA >10% below 50d MA.
    Best consistency signal (8/8 years) applied when stock is deep below trend.
    Hypothesis: weekly support + daily ATR + fear + deeply undervalued = ideal."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    close = float(tsla['close'].iloc[i])
    ma50 = float(tsla['close'].iloc[i - 50:i].mean())
    return 1 if close < ma50 * 0.90 else 0  # >10% below 50d MA


def sig_s224_s215_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S224: S215 (multi-TF + VIX>=18) with ATR compressed-only filter.
    S91 fires on both compressed AND expanding ATR. This keeps only compressed.
    Hypothesis: compressed ATR at weekly support = better than expanding."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0  # Compressed only


def sig_s225_qtr_end_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S225: S91 (ATR extreme) in last 5 trading days of a quarter.
    Quarter-end: index rebalancing creates temporary dislocations.
    TSLA often gets oversold/undersold by forced rebalancing flows.
    Hypothesis: compression during quarter-end = post-rebalancing snap-back."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    month = tsla.index[i].month
    day = tsla.index[i].day
    # Last 5 trading days of Mar, Jun, Sep, Dec (estimated via >22nd of month)
    is_qtr_end_month = month in {3, 6, 9, 12}
    return 1 if (is_qtr_end_month and day >= 22) else 0


def sig_s226_month_end_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S226: S91 + last 3 trading days of any month (month-end effect).
    Month-end rebalancing by funds creates similar dislocations as quarter-end.
    Hypothesis: signal during rebalancing window = faster resolution."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    month = tsla.index[i].month
    # Find last 3 trading days of current month
    all_month = tsla[tsla.index.month == month]
    this_year_month = tsla[(tsla.index.month == month) & (tsla.index.year == tsla.index[i].year)]
    if len(this_year_month) < 3:
        return 1
    # If current date is among last 3 trading days
    last_3 = this_year_month.index[-3:]
    return 1 if tsla.index[i] in last_3 else 0


def sig_s227_s215_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S227: S215 (multi-TF + VIX>=18) + NR7.
    The best all-around signal with the NR7 Crabel compression peak filter.
    Hypothesis: S215 8/8 years → add NR7 → approach 100% WR with good trade count."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr_n(tsla, i, 7) else 0


def sig_s228_s215_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S228: S215 (multi-TF + VIX>=18) + VIX recovery (spike then calm).
    S215 requires VIX>=18 now. This adds: VIX was elevated 5-10d ago AND is now cooling.
    Hypothesis: S215 identifies setup, VIX recovery adds timing precision."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    spike_then = vix_5d > 20 or vix_10d > 20
    calm_now   = vix_now < vix_5d * 0.92  # 8% drop = VIX cooling
    return 1 if (spike_then and calm_now) else 0


SIGNALS_P7Y: List[Tuple] = [
    # Phase 7Y — MA oversold, quarter-end, month-end, S215 variants
    ('S222_below50ma_s91',      sig_s222_below_50ma_s91,     10, 0.20, 50),
    ('S223_below50ma_s215',     sig_s223_below_50ma_s215,    10, 0.20, 50),
    ('S224_s215_compress_only', sig_s224_s215_compressed_only, 10, 0.20, 50),
    ('S225_qtr_end_s91',        sig_s225_qtr_end_s91,        10, 0.20, 50),
    ('S226_month_end_s91',      sig_s226_month_end_s91,      10, 0.20, 50),
    ('S227_s215_nr7',           sig_s227_s215_nr7,           10, 0.20, 50),
    ('S228_s215_vix_rec',       sig_s228_s215_vix_rec,       10, 0.20, 50),
]


# ── Phase 7Z — Final batch: S215 with week-number/day filters, S224 combos ───
# Last exploration phase: remaining ideas not yet tested.
# Focus on S215-family improvements and a few structural market patterns.

def sig_s229_s215_week1_month(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S229: S215 (multi-TF + VIX>=18) only in first 2 weeks of any month.
    Thesis: fund allocations typically deploy in first half of month.
    If S215 fires early in month, new money may be entering simultaneously.
    Hypothesis: removes late-month noise (end-of-month rebalancing selling)."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    day = tsla.index[i].day
    return 1 if day <= 15 else 0


def sig_s230_s224_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S230: S224 (multi-TF + compressed-only, WR=93%, PF=334) with 3-day hold.
    S224 has 15 trades, 8/8 years, WR=93%. Test if holding longer changes results.
    Hypothesis: same setup quality, slightly longer mean-reversion completion."""
    return sig_s224_s215_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s231_s228_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S231: S228 (multi-TF + VIX recovery, WR=93%, PF=14.85) with 3-day hold.
    S228 was 15 trades, 6/7 years, WR=93%. Test hold period variation.
    Hypothesis: VIX recovery setups take slightly longer to fully develop."""
    return sig_s228_s215_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s232_s215_sma20cross(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S232: S215 (multi-TF + VIX>=18) + TSLA below 20d SMA (momentum bearish).
    When TSLA is in the weekly channel support zone AND below 20d MA = deeper dip.
    Hypothesis: deeper into correction + multi-TF support = stronger reversal."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    close = float(tsla['close'].iloc[i])
    sma20 = float(tsla['close'].iloc[i - 20:i].mean())
    return 1 if close < sma20 else 0


def sig_s233_s218_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S233: S218 (multi-TF + 3d hold, best P&L $2.18M) + VIX>=18 filter.
    S218 is 9/10 years, $2.18M but WR=71%. Adding VIX>=18 should raise WR.
    Hypothesis: removing low-VIX trades from S218 → approach S215's 8/8 record."""
    if sig_s218_s214_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i])
    return 1 if vix_now >= 18 else 0


def sig_s234_s215_vix25(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S234: S215 (multi-TF) with higher VIX threshold (VIX>=25 vs VIX>=18).
    Hypothesis: even higher fear = even stronger reversal from weekly support.
    Trade-off: fewer trades but potentially higher WR and avg/trade."""
    if sig_s214_weekly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i])
    return 1 if vix_now >= 25 else 0


def sig_s235_s224_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S235: S224 (multi-TF+compressed) + VIX recovery — ultimate combo.
    S224 is WR=93%, PF=334. Add VIX recovery timing → should be near-perfect.
    Hypothesis: weekly support + compressed ATR + VIX cooling = perfect setup."""
    if sig_s224_s215_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    spike_then = vix_5d > 20 or vix_10d > 20
    calm_now   = vix_now < vix_5d * 0.90
    return 1 if (spike_then and calm_now) else 0


SIGNALS_P7Z: List[Tuple] = [
    # Phase 7Z — Final batch: S215/S224/S228 variations, hold period tests
    ('S229_s215_week1_month',   sig_s229_s215_week1_month,   10, 0.20, 50),
    ('S230_s224_hold3d',        sig_s230_s224_hold3d,         3, 0.20, 50),
    ('S231_s228_hold3d',        sig_s231_s228_hold3d,         3, 0.20, 50),
    ('S232_s215_below20sma',    sig_s232_s215_sma20cross,    10, 0.20, 50),
    ('S233_s218_vix18',         sig_s233_s218_vix18,          3, 0.20, 50),
    ('S234_s215_vix25',         sig_s234_s215_vix25,         10, 0.20, 50),
    ('S235_s224_vix_rec',       sig_s235_s224_vix_rec,       10, 0.20, 50),
]


SIGNALS = SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y + SIGNALS_P7Z


# ── Phase 8A — Multi-TF extensions: RSI, inside day, monthly TF, consec-down ──
# Deepen the S214/S215/S224 family with new analytical dimensions:
#   • RSI-based oversold filters → statistical oversold at multi-TF support
#   • Inside day pattern → price consolidating before reversal
#   • Monthly channel → 3rd timeframe alignment (monthly+weekly+daily)
#   • Consecutive down days → capitulation sequence into weekly support
#   • SPY below 50MA → broad market weakness amplifying mean-reversion

def _bb_lower(tsla, i, period: int = 20, std_mult: float = 2.0) -> Optional[float]:
    """Return lower Bollinger Band (period-day 2σ). None if insufficient data."""
    if i < period:
        return None
    closes = tsla['close'].iloc[i - period:i + 1].values.astype(float)
    mean = closes.mean()
    std  = float(np.std(closes, ddof=1))
    return mean - std_mult * std


def _stoch_k(tsla, i, period: int = 14) -> Optional[float]:
    """Return %K stochastic (0-100). None if insufficient data or flat range."""
    if i < period:
        return None
    lows   = tsla['low'].iloc[i - period:i + 1].values.astype(float)
    highs  = tsla['high'].iloc[i - period:i + 1].values.astype(float)
    lowest, highest = lows.min(), highs.max()
    if highest - lowest < 1e-6:
        return None
    return (float(tsla['close'].iloc[i]) - lowest) / (highest - lowest) * 100.0


def _monthly_channel_near_lower(tsla, i, n_months: int = 12, frac: float = 0.30) -> bool:
    """True if current price is near lower boundary of n_months monthly channel.
    Resamples daily slice to monthly, excludes incomplete current month.
    Returns True (pass-through) when insufficient data."""
    if i < 60:
        return True  # not enough history
    daily_slice = tsla.iloc[:i + 1]
    try:
        monthly = daily_slice.resample('ME').agg(
            open=('open', 'first'), high=('high', 'max'),
            low=('low', 'min'), close=('close', 'last'),
            volume=('volume', 'sum')
        ).dropna(subset=['close'])
        # Drop the incomplete current month (its end date is in the future)
        if len(monthly) < 2:
            return True
        monthly = monthly.iloc[:-1]   # remove current incomplete month
        if len(monthly) < n_months:
            return True
        ch = _channel_at(monthly.iloc[-n_months:])
        if ch is None:
            return True
        cur_close = float(tsla['close'].iloc[i])
        return _near_lower(cur_close, ch, frac)
    except Exception:
        return True


# S236 — S215 + RSI < 35 (daily oversold confirmation)
def sig_s236_s215_rsi35(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S236: S215 (multi-TF + VIX>=18) + TSLA RSI < 35.
    Adds a statistical oversold filter to the 8/8yr signal.
    Hypothesis: weekly support + elevated fear + oversold RSI = peak panic entry."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi = rt.iloc[i]
    if pd.isna(rsi):
        return 1
    return 1 if rsi < 35 else 0


# S237 — S224 + RSI < 30 (best purity + deep oversold)
def sig_s237_s224_rsi30(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S237: S224 (WR=93%, PF=334, multi-TF+compressed) + TSLA RSI < 30.
    S224 is the highest-purity signal. Deep RSI adds maximum oversold confirmation.
    Hypothesis: coiling at weekly support while deeply oversold = spring about to uncoil."""
    if sig_s224_s215_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi = rt.iloc[i]
    if pd.isna(rsi):
        return 1
    return 1 if rsi < 30 else 0


# S238 — S215 + inside day (consolidation candle)
def sig_s238_s215_inside_day(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S238: S215 + today is an inside day (high<=prev high AND low>=prev low).
    Inside day = market pausing/consolidating at weekly support.
    Hypothesis: volatility contraction at support = institutional accumulation in progress."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 1
    hi, lo     = float(tsla['high'].iloc[i]),     float(tsla['low'].iloc[i])
    prev_hi, prev_lo = float(tsla['high'].iloc[i-1]), float(tsla['low'].iloc[i-1])
    return 1 if (hi <= prev_hi and lo >= prev_lo) else 0


# S239 — S224 + inside day
def sig_s239_s224_inside_day(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S239: S224 (best purity, WR=93%) + inside day.
    Double compression: ATR coiling + inside candle at monthly+weekly support.
    Hypothesis: two independent compression signals converging = maximum tension before snap."""
    if sig_s224_s215_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 1
    hi, lo     = float(tsla['high'].iloc[i]),     float(tsla['low'].iloc[i])
    prev_hi, prev_lo = float(tsla['high'].iloc[i-1]), float(tsla['low'].iloc[i-1])
    return 1 if (hi <= prev_hi and lo >= prev_lo) else 0


# S240 — S215 + 3 consecutive down closes (S206 pattern in multi-TF context)
def sig_s240_s215_consec_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S240: S215 + 3+ consecutive down closes before entry.
    Merges the 'capitulation sequence' pattern (S206) with the multi-TF setup.
    Hypothesis: 3-day decline INTO weekly support = textbook exhaustion reversal."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    for j in range(3):
        if float(tsla['close'].iloc[i - j]) >= float(tsla['close'].iloc[i - j - 1]):
            return 0
    return 1


# S241 — S224 + 3 consecutive down closes
def sig_s241_s224_consec_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S241: S224 (best purity) + 3+ consecutive down closes.
    S224 already filters to compressed-only at weekly support. Adding sequential
    selling exhaustion on top of maximum compression = extreme setup convergence."""
    if sig_s224_s215_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    for j in range(3):
        if float(tsla['close'].iloc[i - j]) >= float(tsla['close'].iloc[i - j - 1]):
            return 0
    return 1


# S242 — S215 + SPY below 50d SMA (broad market weakness)
def sig_s242_s215_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S242: S215 + SPY currently below its 50-day SMA.
    When the broad market is in a downtrend AND TSLA hits weekly support = double weakness.
    Hypothesis: macro tailwind for mean-reversion when SPY also under pressure (more fear)."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_sma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close < spy_sma50 else 0


# S243 — S215 + Bollinger lower band touch
def sig_s243_s215_bb_lower(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S243: S215 + TSLA close <= lower Bollinger Band (20d, 2σ).
    Statistical extreme: at or below 2-sigma band at weekly support.
    Hypothesis: statistical + structural support confluence = high-probability snap-back."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    bb_lo = _bb_lower(tsla, i, 20, 2.0)
    if bb_lo is None:
        return 1
    return 1 if float(tsla['close'].iloc[i]) <= bb_lo else 0


# S244 — S215 + 5d hold (testing optimal hold period)
def sig_s244_s215_hold5d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S244: S215 with 5-day hold (between 3d S221/S233 and 10d S215).
    S221/S233 (3d hold) = $1.90M 8/8yr. S215 (10d hold) = $1.94M 8/8yr.
    Testing 5d hold: should give cleaner exits than 10d but more capture than 3d."""
    return sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w)


# S245 — Monthly channel near lower + daily S91 (3rd timeframe)
def sig_s245_monthly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S245: Monthly channel near lower boundary + daily ATR extreme (S91).
    New timeframe dimension: checks TSLA is near lower monthly channel (12-month lookback).
    Hypothesis: monthly support + daily extreme = major structural bounce zone."""
    if sig_s91_s32_atr_extreme(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _monthly_channel_near_lower(tsla, i, n_months=12, frac=0.30) else 0


# S246 — Monthly channel near lower + S215 (3-TF: monthly+weekly+daily)
def sig_s246_monthly_low_s215(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S246: Monthly near lower channel + S215 (3-timeframe: monthly+weekly+daily+VIX).
    If weekly filter already works, adding monthly alignment should create gold standard.
    Hypothesis: monthly support zone + weekly support zone + daily ATR extreme = ultra-rare
    major bottom signal."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _monthly_channel_near_lower(tsla, i, n_months=12, frac=0.30) else 0


# S247 — Stochastic < 20 + S215
def sig_s247_s215_stoch20(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S247: S215 + Stochastic %K < 20 (oversold on 14-bar stochastic).
    Stochastic measures position of close within recent high-low range.
    Hypothesis: multi-TF support + stochastic oversold = momentum near exhaustion."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    stoch = _stoch_k(tsla, i, 14)
    if stoch is None:
        return 1
    return 1 if stoch < 20 else 0


# S248 — S224 + VIX >= 22 (elevated fear threshold)
def sig_s248_s224_vix22(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S248: S224 (WR=93%, PF=334) with VIX >= 22 instead of S215's VIX>=18.
    S224 inherits VIX>=18 from S215. Test slightly higher fear bar.
    Hypothesis: VIX>=22 = meaningful fear vs VIX>=18. Should maintain high WR with fewer trades."""
    if sig_s224_s215_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i])
    return 1 if vix_now >= 22 else 0


# S249 — S235 + inside day (100% WR base + consolidation)
def sig_s249_s235_inside_day(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S249: S235 (100% WR: S224+VIX_recovery) + inside day.
    S235 = multi-TF + compressed + VIX recovery (6/6yr 100% WR).
    Adding inside day should keep only highest-quality setups."""
    if sig_s235_s224_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 1
    hi, lo     = float(tsla['high'].iloc[i]),     float(tsla['low'].iloc[i])
    prev_hi, prev_lo = float(tsla['high'].iloc[i-1]), float(tsla['low'].iloc[i-1])
    return 1 if (hi <= prev_hi and lo >= prev_lo) else 0


# S250 — S215 + gap down open (capitulation open at weekly support)
def sig_s250_s215_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S250: S215 + today opened below yesterday's close (gap down).
    Gap down at weekly support = panic opening, immediate value zone.
    Hypothesis: institutional demand absorbs the gap at major support = strong reversal."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 1
    open_today   = float(tsla['open'].iloc[i])
    close_prev   = float(tsla['close'].iloc[i - 1])
    return 1 if open_today < close_prev else 0


SIGNALS_P8A: List[Tuple] = [
    # Phase 8A — Multi-TF extensions: RSI, inside day, monthly TF, consecutive down
    ('S236_s215_rsi35',         sig_s236_s215_rsi35,         10, 0.20, 50),
    ('S237_s224_rsi30',         sig_s237_s224_rsi30,         10, 0.20, 50),
    ('S238_s215_inside_day',    sig_s238_s215_inside_day,    10, 0.20, 50),
    ('S239_s224_inside_day',    sig_s239_s224_inside_day,    10, 0.20, 50),
    ('S240_s215_consec_down',   sig_s240_s215_consec_down,   10, 0.20, 50),
    ('S241_s224_consec_down',   sig_s241_s224_consec_down,   10, 0.20, 50),
    ('S242_s215_spy_weak',      sig_s242_s215_spy_weak,      10, 0.20, 50),
    ('S243_s215_bb_lower',      sig_s243_s215_bb_lower,      10, 0.20, 50),
    ('S244_s215_hold5d',        sig_s244_s215_hold5d,         5, 0.20, 50),
    ('S245_monthly_low_s91',    sig_s245_monthly_low_daily_s91, 10, 0.20, 50),
    ('S246_monthly_low_s215',   sig_s246_monthly_low_s215,   10, 0.20, 50),
    ('S247_s215_stoch20',       sig_s247_s215_stoch20,       10, 0.20, 50),
    ('S248_s224_vix22',         sig_s248_s224_vix22,         10, 0.20, 50),
    ('S249_s235_inside_day',    sig_s249_s235_inside_day,    10, 0.20, 50),
    ('S250_s215_gap_down',      sig_s250_s215_gap_down,      10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A)


# ── Phase 8B — Extend top Phase 8A signals: S245/S242/S248/S250 families ──────
# S245 (monthly near low + S91) loses 2016/2017 (low-VIX bull years) → add VIX filter.
# S248 (S224+VIX>=22) is a new high-purity base → combine with VIX recovery.
# S242 (S215+SPY<50MA) is $1.14M 5/6yr → test VIX recovery and compression filters.
# S250 (S215+gap down) is 6/8yr → test VIX recovery and compression sub-filters.

def sig_s251_s245_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S251: S245 (monthly near lower + daily S91) + VIX >= 18.
    S245 loses 2016/2017 (low-VIX bull market). VIX>=18 gate removes low-quality setups.
    Hypothesis: monthly support + elevated fear = major regime bottom."""
    if sig_s245_monthly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if float(vix['close'].iloc[i]) >= 18 else 0


def sig_s252_s245_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S252: S245 (monthly near lower + daily S91) + VIX recovery pattern.
    Monthly support + fear subsiding = the VIX recovery catalyst (S132 family).
    Hypothesis: monthly bottom + VIX cooling = most powerful macro reversal timing."""
    if sig_s245_monthly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s253_s245_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S253: S245 (monthly near lower + S91) + compressed ATR only.
    Monthly support + coiling compression = multi-TF spring setup.
    Similar to S224 (weekly+compressed) but with monthly TF instead of weekly."""
    if sig_s245_monthly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s254_s245_hold3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S254: S245 (monthly near lower + S91) with 3-day hold.
    Monthly support bounces may complete faster. Test shorter hold period."""
    return sig_s245_monthly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s255_s248_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S255: S248 (S224+VIX>=22) + VIX recovery.
    S248 is WR=83%, PF=131, 3/3yr. Adding VIX recovery to the high-VIX filter.
    Hypothesis: compressed at weekly support + VIX>=22 AND then VIX cooling = peak fear exit."""
    if sig_s248_s224_vix22(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s256_s242_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S256: S242 (S215+SPY<50MA, $1.14M, 5/6yr) + VIX recovery.
    SPY below 50MA + weekly TSLA support + VIX cooling = classic macro reversal.
    Hypothesis: broad market weakness + VIX timing = highest-quality S242 subset."""
    if sig_s242_s215_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s257_s242_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S257: S242 (S215+SPY<50MA) + compressed ATR only.
    Remove expanding-ATR setups from the SPY-weak filter.
    Hypothesis: SPY weakness + weekly support + coiling compression = maximum alpha."""
    if sig_s242_s215_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s258_s250_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S258: S250 (S215+gap down) + VIX recovery.
    Gap down at weekly support + VIX cooling = capitulation open + macro tailwind.
    Hypothesis: institutional selling exhausts (gap) just as macro fear subsides."""
    if sig_s250_s215_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s259_s250_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S259: S250 (S215+gap down) + compressed ATR only.
    Gap down + compression at weekly support = stock quietly sold off to support.
    Hypothesis: no explosive panic (compressed) + gap = controlled distribution ending."""
    if sig_s250_s215_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s260_s245_weekly_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S260: Monthly near lower channel + S214 (weekly near lower + daily S91).
    3-TF alignment without VIX requirement: monthly+weekly+daily all near lower boundary.
    Compare to S246 (same but also VIX>=18). Does VIX gate add or subtract value here?"""
    if sig_s214_weekly_low_daily_s91(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _monthly_channel_near_lower(tsla, i, n_months=12, frac=0.30) else 0


SIGNALS_P8B: List[Tuple] = [
    # Phase 8B — Monthly TF extensions + S242/S248/S250 family deeper filters
    ('S251_s245_vix18',         sig_s251_s245_vix18,         10, 0.20, 50),
    ('S252_s245_vix_rec',       sig_s252_s245_vix_rec,       10, 0.20, 50),
    ('S253_s245_compressed',    sig_s253_s245_compressed,    10, 0.20, 50),
    ('S254_s245_hold3d',        sig_s254_s245_hold3d,         3, 0.20, 50),
    ('S255_s248_vix_rec',       sig_s255_s248_vix_rec,       10, 0.20, 50),
    ('S256_s242_vix_rec',       sig_s256_s242_vix_rec,       10, 0.20, 50),
    ('S257_s242_compressed',    sig_s257_s242_compressed,    10, 0.20, 50),
    ('S258_s250_vix_rec',       sig_s258_s250_vix_rec,       10, 0.20, 50),
    ('S259_s250_compressed',    sig_s259_s250_compressed,    10, 0.20, 50),
    ('S260_s245_weekly_low',    sig_s260_s245_weekly_low,    10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B)


# ── Phase 8C — Extend Phase 8B winners + 5-day momentum dimension ─────────────
# Deep combinatorial exploration of new signals found in 8A/8B:
#   S258 (100% WR gap down+VIX rec) + sub-filters
#   S257/S260/S253 with new combinations
#   5-day momentum decline: sharp short-term selloff into weekly support

def sig_s261_s258_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S261: S258 (100% WR: gap down+VIX recovery at weekly support) + compressed ATR.
    4-way filter: gap down + VIX recovery + compressed ATR + weekly low.
    Hypothesis: all 4 conditions = once-a-year institutional capitulation close."""
    if sig_s258_s250_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s262_s257_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S262: S257 (SPY<50MA+compressed, 6/6yr WR=89%) + VIX recovery timing.
    Adding the optimal VIX timing signal to the 6/6 year perfect record.
    Hypothesis: SPY weakness + compression + VIX cooling = multi-factor confluence."""
    if sig_s257_s242_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s263_s260_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S263: S260 (3-TF: monthly+weekly+daily near lower channels) + VIX recovery.
    Adding VIX timing to the 3-TF structural alignment signal.
    Hypothesis: 3 timeframes all at lower boundary + fear cooling = rare regime bottom."""
    if sig_s260_s245_weekly_low(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s264_s260_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S264: S260 (3-TF monthly+weekly+daily) + compressed ATR only.
    3-TF structural support + coiling compression = maximum tension before snap.
    Hypothesis: all three TFs point to support + daily compression = textbook setup."""
    if sig_s260_s245_weekly_low(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s265_s215_5d_down8pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S265: S215 (multi-TF 8/8yr) + TSLA down 8%+ over last 5 trading days.
    Sharp short-term selloff INTO the weekly support level = capitulation velocity.
    Hypothesis: rapid decline to support (not gradual) = sellers exhausted at key level."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now  = float(tsla['close'].iloc[i])
    close_5d   = float(tsla['close'].iloc[i - 5])
    chg_5d = (close_now - close_5d) / close_5d
    return 1 if chg_5d <= -0.08 else 0


def sig_s266_s215_5d_down5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S266: S215 + TSLA down 5%+ over last 5 days (looser threshold than S265).
    More trades with weaker selectivity. Trade-off: frequency vs purity.
    Hypothesis: even 5% 5-day decline at weekly support shows directional conviction."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now  = float(tsla['close'].iloc[i])
    close_5d   = float(tsla['close'].iloc[i - 5])
    chg_5d = (close_now - close_5d) / close_5d
    return 1 if chg_5d <= -0.05 else 0


def sig_s267_s224_5d_down5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S267: S224 (best purity WR=93%) + TSLA down 5%+ over last 5 days.
    The 5-day momentum decline on top of compressed ATR at weekly support.
    Hypothesis: compressed ATR but falling price = steady institutional selling ending."""
    if sig_s224_s215_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now  = float(tsla['close'].iloc[i])
    close_5d   = float(tsla['close'].iloc[i - 5])
    chg_5d = (close_now - close_5d) / close_5d
    return 1 if chg_5d <= -0.05 else 0


def sig_s268_s242_consec_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S268: S242 (S215+SPY<50MA, $1.14M) + 3 consecutive down closes.
    Broad market weakness + weekly TSLA support + sequential TSLA selling = stacked.
    Hypothesis: individual stock capitulates while market already weakened = max fear."""
    if sig_s242_s215_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    for j in range(3):
        if float(tsla['close'].iloc[i - j]) >= float(tsla['close'].iloc[i - j - 1]):
            return 0
    return 1


def sig_s269_s253_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S269: S253 (monthly near lower + compressed ATR) + VIX >= 18.
    Adding fear context to the monthly support + compression signal.
    Hypothesis: monthly support + compression + elevated market fear = high-quality."""
    if sig_s253_s245_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if float(vix['close'].iloc[i]) >= 18 else 0


def sig_s270_s252_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S270: S252 (monthly near lower + VIX recovery) + compressed ATR.
    Triple: monthly structural support + VIX recovery timing + daily compression.
    Hypothesis: monthly bottom + fear cooling + coiling = monthly edition of S235."""
    if sig_s252_s245_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


SIGNALS_P8C: List[Tuple] = [
    # Phase 8C — 4-way combinatorial filters + 5-day momentum dimension
    ('S261_s258_compressed',    sig_s261_s258_compressed,    10, 0.20, 50),
    ('S262_s257_vix_rec',       sig_s262_s257_vix_rec,       10, 0.20, 50),
    ('S263_s260_vix_rec',       sig_s263_s260_vix_rec,       10, 0.20, 50),
    ('S264_s260_compressed',    sig_s264_s260_compressed,    10, 0.20, 50),
    ('S265_s215_5d_down8pct',   sig_s265_s215_5d_down8pct,  10, 0.20, 50),
    ('S266_s215_5d_down5pct',   sig_s266_s215_5d_down5pct,  10, 0.20, 50),
    ('S267_s224_5d_down5pct',   sig_s267_s224_5d_down5pct,  10, 0.20, 50),
    ('S268_s242_consec_down',   sig_s268_s242_consec_down,  10, 0.20, 50),
    ('S269_s253_vix18',         sig_s269_s253_vix18,         10, 0.20, 50),
    ('S270_s252_compressed',    sig_s270_s252_compressed,    10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C)


# ── Phase 8D — 5-day momentum family + extend 8C top signals ──────────────────
# S262 (100% WR: SPY<50MA+compressed+VIX rec) is our new highest-quality base.
# S265/S266 (5-day momentum decline) is a new dimension — extend aggressively.
# S269 (monthly+compressed+VIX18, 5/5yr) is the best monthly signal — extend further.

def sig_s271_s265_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S271: S265 (S215+8% 5-day decline, 7/8yr WR=86%) + VIX recovery.
    Sharp 8% decline into weekly support AND VIX cooling simultaneously.
    Hypothesis: multi-day selling pressure exhausts exactly when macro fear peaks."""
    if sig_s265_s215_5d_down8pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s272_s265_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S272: S265 (S215+8% 5-day decline) + SPY below 50d SMA.
    Sharp TSLA decline at weekly support AND broad market weakness.
    Hypothesis: individual + macro weakness converge → cleanest mean-reversion entry."""
    if sig_s265_s215_5d_down8pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_sma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close < spy_sma50 else 0


def sig_s273_s265_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S273: S265 (S215+8% 5-day decline) + compressed ATR.
    ATR compressed = stock declining QUIETLY (not explosively) over 5 days.
    Hypothesis: quiet 8% drift down to support + compression = controlled selling ending."""
    if sig_s265_s215_5d_down8pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s274_s267_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S274: S267 (S224+5d decline, 100% WR) + VIX recovery.
    S267 = compressed + weekly support + 5% 5-day decline. Now add VIX timing.
    Hypothesis: all 4 elements = ultra-rare. Should be 100% WR or near-perfect."""
    if sig_s267_s224_5d_down5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s275_s263_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S275: S263 (3-TF+VIX recovery, avg=$101K) + compressed ATR.
    3-TF structural alignment + VIX recovery timing + daily compression.
    Hypothesis: 4-way signal convergence = absolute highest-quality monthly setup."""
    if sig_s263_s260_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s276_s269_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S276: S269 (monthly+compressed+VIX18, 5/5yr WR=92%) + VIX recovery timing.
    S269 is perfect year rate with 12 trades. Adding VIX recovery timing refines entry.
    Hypothesis: monthly structural + compression + fear cooling → strongest monthly setup."""
    if sig_s269_s253_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s277_s264_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S277: S264 (3-TF+compressed, 7/9yr WR=79%) + VIX >= 18.
    3-TF structural compression + elevated fear context.
    Hypothesis: VIX gate removes low-quality low-fear setups from the 3-TF pattern."""
    if sig_s264_s260_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if float(vix['close'].iloc[i]) >= 18 else 0


def sig_s278_s266_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S278: S266 (S215+5% 5-day decline, 7/8yr) + VIX recovery.
    Combining 5-day momentum (broader threshold) with VIX cooling timing.
    Hypothesis: 5% steady decline + VIX subsiding = market participants re-entering."""
    if sig_s266_s215_5d_down5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s279_s260_5d_down5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S279: S260 (3-TF: monthly+weekly+daily channels) + 5% 5-day decline.
    All three timeframe channels near lower boundary AND recent sharp decline.
    Hypothesis: 3 structural supports + velocity of decline = textbook capitulation."""
    if sig_s260_s245_weekly_low(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now = float(tsla['close'].iloc[i])
    close_5d  = float(tsla['close'].iloc[i - 5])
    chg_5d = (close_now - close_5d) / close_5d
    return 1 if chg_5d <= -0.05 else 0


def sig_s280_s262_vix22(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S280: S262 (100% WR: SPY<50MA+compressed+VIX_rec) + VIX>=22.
    S262 has 5 trades 100% WR — all in high-fear periods anyway.
    Test: does higher VIX threshold (22 vs 18) maintain or improve the signal?"""
    if sig_s262_s257_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if float(vix['close'].iloc[i]) >= 22 else 0


SIGNALS_P8D: List[Tuple] = [
    # Phase 8D — 5-day momentum extensions + 4-way combinatorial filters
    ('S271_s265_vix_rec',       sig_s271_s265_vix_rec,       10, 0.20, 50),
    ('S272_s265_spy_weak',      sig_s272_s265_spy_weak,      10, 0.20, 50),
    ('S273_s265_compressed',    sig_s273_s265_compressed,    10, 0.20, 50),
    ('S274_s267_vix_rec',       sig_s274_s267_vix_rec,       10, 0.20, 50),
    ('S275_s263_compressed',    sig_s275_s263_compressed,    10, 0.20, 50),
    ('S276_s269_vix_rec',       sig_s276_s269_vix_rec,       10, 0.20, 50),
    ('S277_s264_vix18',         sig_s277_s264_vix18,         10, 0.20, 50),
    ('S278_s266_vix_rec',       sig_s278_s266_vix_rec,       10, 0.20, 50),
    ('S279_s260_5d_down5pct',   sig_s279_s260_5d_down5pct,  10, 0.20, 50),
    ('S280_s262_vix22',         sig_s280_s262_vix22,         10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D)


# ── Phase 8E — S277 family + 10d momentum + volume dimensions ─────────────────
# S277 (3-TF+compressed+VIX18, 5/5yr WR=90%) is our best-coverage signal — extend it.
# New dimension: 10-day momentum (down 12%+ over 2 weeks = serious correction).
# New dimension: volume expansion at entry (2× avg → institutional accumulation signal).

def sig_s281_s277_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S281: S277 (3-TF+compressed+VIX18, 5/5yr) + VIX recovery timing.
    Adding the best timing signal to the best structure signal.
    Hypothesis: 3-TF structural + compression + VIX gate + VIX cooling = ultra-rare."""
    if sig_s277_s264_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s282_s277_5d_down5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S282: S277 (3-TF+compressed+VIX18) + 5% 5-day decline.
    Adds momentum direction confirmation: stock has been falling into the support zone.
    Hypothesis: directional decline + 3-TF structural support = entries with momentum."""
    if sig_s277_s264_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now = float(tsla['close'].iloc[i])
    close_5d  = float(tsla['close'].iloc[i - 5])
    return 1 if (close_now - close_5d) / close_5d <= -0.05 else 0


def sig_s283_s277_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S283: S277 (3-TF+compressed+VIX18) + SPY below 50d SMA.
    Adding broad market context: 3-TF structural support during macro downturn.
    Hypothesis: all structural TFs point down + SPY weak = maximum risk-off reversal."""
    if sig_s277_s264_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_sma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close < spy_sma50 else 0


def sig_s284_s215_10d_down12pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S284: S215 (multi-TF 8/8yr) + TSLA down 12%+ over last 10 trading days.
    2-week significant correction into weekly support zone.
    Hypothesis: sustained 2-week decline to weekly support = sellers running out."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    close_now = float(tsla['close'].iloc[i])
    close_10d = float(tsla['close'].iloc[i - 10])
    return 1 if (close_now - close_10d) / close_10d <= -0.12 else 0


def sig_s285_s215_vol2x(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S285: S215 (multi-TF 8/8yr) + today's volume >= 2× 20-day average.
    High volume at weekly support = potential institutional accumulation day.
    Hypothesis: volume surge at support = smart money entering, not just panic selling."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    vol_today = float(tsla['volume'].iloc[i])
    vol_avg   = float(tsla['volume'].iloc[i - 20:i].mean())
    if vol_avg <= 0:
        return 1
    return 1 if vol_today >= 2.0 * vol_avg else 0


def sig_s286_s215_vol15x(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S286: S215 + volume >= 1.5× 20-day average (looser threshold than S285).
    More trades vs S285's 2× threshold. Trade-off: frequency vs quality."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    vol_today = float(tsla['volume'].iloc[i])
    vol_avg   = float(tsla['volume'].iloc[i - 20:i].mean())
    if vol_avg <= 0:
        return 1
    return 1 if vol_today >= 1.5 * vol_avg else 0


def sig_s287_s224_vol2x(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S287: S224 (WR=93%, PF=334) + today's volume >= 2× 20-day average.
    Best purity signal with institutional accumulation volume.
    Hypothesis: compressed at weekly support + high volume = final accumulation day."""
    if sig_s224_s215_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    vol_today = float(tsla['volume'].iloc[i])
    vol_avg   = float(tsla['volume'].iloc[i - 20:i].mean())
    if vol_avg <= 0:
        return 1
    return 1 if vol_today >= 2.0 * vol_avg else 0


def sig_s288_s257_5d_down5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S288: S257 (SPY<50MA+compressed, 6/6yr WR=89%) + 5% 5-day decline.
    Adding momentum to the 6/6 perfect coverage signal.
    Hypothesis: macro weak + compressed at support + declining stock = peak capitulation."""
    if sig_s257_s242_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now = float(tsla['close'].iloc[i])
    close_5d  = float(tsla['close'].iloc[i - 5])
    return 1 if (close_now - close_5d) / close_5d <= -0.05 else 0


def sig_s289_s265_vol15x(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S289: S265 (S215+8% 5-day decline) + volume >= 1.5× average.
    Sharp price decline + above-average volume = institutional position-taking.
    Hypothesis: volume confirms the decline is a buying opportunity, not continuation."""
    if sig_s265_s215_5d_down8pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    vol_today = float(tsla['volume'].iloc[i])
    vol_avg   = float(tsla['volume'].iloc[i - 20:i].mean())
    if vol_avg <= 0:
        return 1
    return 1 if vol_today >= 1.5 * vol_avg else 0


def sig_s290_s215_rsi30(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S290: S215 (multi-TF 8/8yr) + RSI < 30 (deeply oversold).
    Stricter than S236 (RSI<35). S215 + deep oversold = maximum fear at weekly support.
    Hypothesis: standard oversold (30) vs panic (35) — does deeper oversold help?"""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi = rt.iloc[i]
    if pd.isna(rsi):
        return 1
    return 1 if rsi < 30 else 0


SIGNALS_P8E: List[Tuple] = [
    # Phase 8E — S277 extensions + 10d momentum + volume dimensions
    ('S281_s277_vix_rec',       sig_s281_s277_vix_rec,       10, 0.20, 50),
    ('S282_s277_5d_down5pct',   sig_s282_s277_5d_down5pct,  10, 0.20, 50),
    ('S283_s277_spy_weak',      sig_s283_s277_spy_weak,      10, 0.20, 50),
    ('S284_s215_10d_down12pct', sig_s284_s215_10d_down12pct, 10, 0.20, 50),
    ('S285_s215_vol2x',         sig_s285_s215_vol2x,         10, 0.20, 50),
    ('S286_s215_vol15x',        sig_s286_s215_vol15x,        10, 0.20, 50),
    ('S287_s224_vol2x',         sig_s287_s224_vol2x,         10, 0.20, 50),
    ('S288_s257_5d_down5pct',   sig_s288_s257_5d_down5pct,  10, 0.20, 50),
    ('S289_s265_vol15x',        sig_s289_s265_vol15x,        10, 0.20, 50),
    ('S290_s215_rsi30',         sig_s290_s215_rsi30,         10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E)


# ── Phase 8F — Fresh dimensions: 200MA, OPEX, 52wk proximity, VIX-rise ────────
# Unexplored structural context: long-term trend (200d MA), calendar patterns
# (OPEX week), 52-week channel position, VIX building before cooling.

def sig_s291_s215_below200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S291: S215 (multi-TF 8/8yr) + TSLA currently below 200-day SMA.
    Tests if mean-reversion from weekly support works better in long-term downtrends.
    Hypothesis: weekly support bounce in bear-trend = counter-trend with oversold bounce."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close < ma200 else 0


def sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S292: S215 + TSLA above 200-day SMA (long-term uptrend).
    Tests if weekly support bounces work better when stock is in long-term uptrend.
    Hypothesis: dip to weekly support in bull-trend = higher-quality entry."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s293_s215_opex_week(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S293: S215 + within OPEX week (option expiration ±5 calendar days).
    OPEX creates pinning and volatility effects. Test if multi-TF signal fires well here.
    Hypothesis: market makers pin price near max pain → support bounces reinforced."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    to_next, from_last = _opex_proximity(dt)
    in_opex_week = (to_next <= 5) or (from_last <= 5)
    return 1 if in_opex_week else 0


def sig_s294_s215_52wk_low20pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S294: S215 + TSLA within 20% of its 52-week low.
    Price near annual low = maximum long-term pessimism. Weekly support at yearly bottom.
    Hypothesis: 52wk low proximity + weekly structural support = major long-term bottom."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 252:
        return 1
    close = float(tsla['close'].iloc[i])
    lo_52 = float(tsla['low'].iloc[i - 252:i].min())
    if lo_52 <= 0:
        return 1
    return 1 if close <= lo_52 * 1.20 else 0


def sig_s295_s215_52wk_low30pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S295: S215 + TSLA within 30% of its 52-week low (looser than S294).
    More trades, weaker selectivity but more frequency of 52-week vicinity signals."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 252:
        return 1
    close = float(tsla['close'].iloc[i])
    lo_52 = float(tsla['low'].iloc[i - 252:i].min())
    if lo_52 <= 0:
        return 1
    return 1 if close <= lo_52 * 1.30 else 0


def sig_s296_s215_vix_rising3d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S296: S215 + VIX has risen 3 consecutive days (fear building, not yet peaking).
    VIX rising into the weekly support setup = fear still building, not yet reverting.
    Hypothesis: fear at peak is better entry than fear already cooling (complements VIX rec)."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    for j in range(3):
        if float(vix['close'].iloc[i - j]) <= float(vix['close'].iloc[i - j - 1]):
            return 0
    return 1


def sig_s297_s277_below200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S297: S277 (3-TF+compressed+VIX18, 5/5yr) + TSLA below 200d MA.
    3-TF structural support during long-term downtrend.
    Hypothesis: monthly+weekly+daily lower channels + bear trend = maximum structural fear."""
    if sig_s277_s264_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close < ma200 else 0


def sig_s298_s277_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S298: S277 (3-TF+compressed+VIX18) + TSLA above 200d MA.
    3-TF support bounce in long-term uptrend = quality dip-buy setup.
    Hypothesis: fear-elevated compression at 3-TF support in uptrend = cleanest entry."""
    if sig_s277_s264_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s299_s265_below200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S299: S265 (S215+8% 5-day decline) + TSLA below 200d MA.
    Sharp decline at weekly support AND in a long-term downtrend.
    Hypothesis: bouncing from weekly support in a bear market = strongest counter-trend."""
    if sig_s265_s215_5d_down8pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close < ma200 else 0


def sig_s300_s265_52wk_low30pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S300: S265 (S215+8% 5-day decline) + within 30% of 52-week low.
    Sharp decline AT annual low proximity + weekly support = double bottom potential.
    Hypothesis: fastest way down + near annual low at weekly support = maximum pessimism."""
    if sig_s265_s215_5d_down8pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 252:
        return 1
    close = float(tsla['close'].iloc[i])
    lo_52 = float(tsla['low'].iloc[i - 252:i].min())
    if lo_52 <= 0:
        return 1
    return 1 if close <= lo_52 * 1.30 else 0


SIGNALS_P8F: List[Tuple] = [
    # Phase 8F — 200MA context, OPEX, 52wk low proximity, VIX-rising
    ('S291_s215_below200ma',    sig_s291_s215_below200ma,    10, 0.20, 50),
    ('S292_s215_above200ma',    sig_s292_s215_above200ma,    10, 0.20, 50),
    ('S293_s215_opex_week',     sig_s293_s215_opex_week,     10, 0.20, 50),
    ('S294_s215_52wk_low20pct', sig_s294_s215_52wk_low20pct, 10, 0.20, 50),
    ('S295_s215_52wk_low30pct', sig_s295_s215_52wk_low30pct, 10, 0.20, 50),
    ('S296_s215_vix_rising3d',  sig_s296_s215_vix_rising3d,  10, 0.20, 50),
    ('S297_s277_below200ma',    sig_s297_s277_below200ma,    10, 0.20, 50),
    ('S298_s277_above200ma',    sig_s298_s277_above200ma,    10, 0.20, 50),
    ('S299_s265_below200ma',    sig_s299_s265_below200ma,    10, 0.20, 50),
    ('S300_s265_52wk_low30pct', sig_s300_s265_52wk_low30pct, 10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F)


# ── Phase 8G — 200MA breakthrough: extend S292 + OPEX combos ──────────────────
# S292 (S215+above200MA) = WR=89%, PF=23.15, $1.32M, 6/6yr — BEST balanced signal.
# Key finding: bounces from weekly support in a long-term UPTREND = dramatically better.
# Extend this dimension: above200MA × every major sub-filter.

def sig_s301_s292_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S301: S292 (S215+above200MA, WR=89%) + VIX recovery timing.
    Best balanced signal + best timing signal = premium entry quality.
    Hypothesis: uptrend + weekly support + VIX cooling = cleanest swing entry."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s302_s292_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S302: S292 (S215+above200MA) + compressed ATR only.
    Uptrend + weekly lower channel + coiling ATR = spring setup in bull trend.
    Hypothesis: compression in an uptrend at weekly support = highest-quality bounce."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s303_s292_5d_down5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S303: S292 (S215+above200MA) + 5% 5-day decline.
    Long-term uptrend + short-term 5% decline to weekly support = dip-buy.
    Hypothesis: uptrend intact + weekly support touched after 5% dip = textbook entry."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now = float(tsla['close'].iloc[i])
    close_5d  = float(tsla['close'].iloc[i - 5])
    return 1 if (close_now - close_5d) / close_5d <= -0.05 else 0


def sig_s304_s292_opex_week(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S304: S292 (S215+above200MA) + OPEX week.
    Uptrend + weekly support + OPEX pinning effect simultaneously.
    Hypothesis: 3 independent forces create extremely high-probability bounce."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    to_next, from_last = _opex_proximity(dt)
    return 1 if (to_next <= 5 or from_last <= 5) else 0


def sig_s305_s292_consec_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S305: S292 (S215+above200MA) + 3+ consecutive down closes.
    Uptrend dip: 3 days of consecutive selling into weekly support in a bull trend.
    Hypothesis: short-term panic in a long-term bull = strongest reversal signal."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    for j in range(3):
        if float(tsla['close'].iloc[i - j]) >= float(tsla['close'].iloc[i - j - 1]):
            return 0
    return 1


def sig_s306_s224_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S306: S224 (WR=93%, PF=334, best purity) + TSLA above 200d MA.
    Best purity signal + uptrend context.
    Hypothesis: compressed + weekly support + VIX18 + uptrend = all-star combo."""
    if sig_s224_s215_compressed_only(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s307_s235_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S307: S235 (100% WR: S224+VIX_recovery) + TSLA above 200d MA.
    S235 already achieved 100% WR (6/6yr). Adding uptrend context.
    Hypothesis: very rare. Should maintain or improve WR when in uptrend."""
    if sig_s235_s224_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s308_s293_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S308: S293 (S215+OPEX week, 8/8yr) + TSLA above 200d MA.
    OPEX week signal filtered to uptrend-only.
    Hypothesis: OPEX pinning + uptrend = the best OPEX bounces are in bull markets."""
    if sig_s293_s215_opex_week(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s309_s292_vix25(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S309: S292 (S215+above200MA) + VIX >= 25 (elevated fear in uptrend).
    Uptrend + weekly support + high fear (VIX25+) = panic dip in bull market.
    Hypothesis: high VIX dip in an uptrend = fastest bounce (fear overcorrection)."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if float(vix['close'].iloc[i]) >= 25 else 0


def sig_s310_s292_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S310: S292 (S215+above200MA) + TSLA below 50d SMA.
    200d MA uptrend but currently dipped below 50d MA — intermediate correction.
    Hypothesis: TSLA in bull-trend but 50d broken = momentum seller exhaustion."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    close  = float(tsla['close'].iloc[i])
    ma50   = float(tsla['close'].iloc[i - 50:i].mean())
    return 1 if close < ma50 else 0


SIGNALS_P8G: List[Tuple] = [
    # Phase 8G — 200MA breakthrough extensions + OPEX combos
    ('S301_s292_vix_rec',       sig_s301_s292_vix_rec,       10, 0.20, 50),
    ('S302_s292_compressed',    sig_s302_s292_compressed,    10, 0.20, 50),
    ('S303_s292_5d_down5pct',   sig_s303_s292_5d_down5pct,  10, 0.20, 50),
    ('S304_s292_opex_week',     sig_s304_s292_opex_week,     10, 0.20, 50),
    ('S305_s292_consec_down',   sig_s305_s292_consec_down,   10, 0.20, 50),
    ('S306_s224_above200ma',    sig_s306_s224_above200ma,    10, 0.20, 50),
    ('S307_s235_above200ma',    sig_s307_s235_above200ma,    10, 0.20, 50),
    ('S308_s293_above200ma',    sig_s308_s293_above200ma,    10, 0.20, 50),
    ('S309_s292_vix25',         sig_s309_s292_vix25,         10, 0.20, 50),
    ('S310_s292_below50ma',     sig_s310_s292_below50ma,     10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G)


# ── Phase 8H — below200MA signals (for bear-trend regimes like 2025) + S310 ───
# 2025 OOS: TSLA was below 200d MA all year — the above200MA signals had 0 trades.
# The below200MA signals (S291 family) are what fire in bear-trend regimes.
# Also extend S310 (above200MA+below50MA) which bridges the two regimes.

def sig_s311_s291_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S311: S291 (S215+below200MA, bear trend) + VIX recovery timing.
    Bear-trend weekly support bounce + fear cooling = counter-trend with momentum.
    Hypothesis: VIX recovery confirms the bounce attempt in a downtrend is real."""
    if sig_s291_s215_below200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s312_s291_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S312: S291 (S215+below200MA) + compressed ATR only.
    Bear-trend + weekly support + coiling compression.
    Hypothesis: compression in a downtrend at support = sellers running out of fuel."""
    if sig_s291_s215_below200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s313_s291_consec_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S313: S291 (S215+below200MA) + 3+ consecutive down closes.
    Bear-trend capitulation: 3 down days into weekly support in a downtrend.
    Hypothesis: acceleration of decline in downtrend to weekly support = peak exhaustion."""
    if sig_s291_s215_below200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    for j in range(3):
        if float(tsla['close'].iloc[i - j]) >= float(tsla['close'].iloc[i - j - 1]):
            return 0
    return 1


def sig_s314_s291_5d_down8pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S314: S291 (S215+below200MA) + 8% 5-day decline.
    Bear-trend rapid decline into weekly support.
    Hypothesis: 8%+ 5-day drop at weekly support in downtrend = max momentum exhaustion."""
    if sig_s291_s215_below200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now = float(tsla['close'].iloc[i])
    close_5d  = float(tsla['close'].iloc[i - 5])
    return 1 if (close_now - close_5d) / close_5d <= -0.08 else 0


def sig_s315_s310_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S315: S310 (above200MA+below50MA, 5/5yr WR=86%) + VIX recovery.
    Uptrend with 50MA pullback + VIX cooling = classic bull-market dip-buy entry.
    Hypothesis: intermediate pullback in uptrend peaks when VIX starts cooling."""
    if sig_s310_s292_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s316_s310_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S316: S310 (above200MA+below50MA) + compressed ATR.
    Bull-trend 50MA pullback + ATR coiling at weekly support.
    Hypothesis: triple compression (200↑, 50↓, ATR coiled) = coil before spring."""
    if sig_s310_s292_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s317_s310_5d_down5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S317: S310 (above200MA+below50MA) + 5% 5-day decline.
    Uptrend 50MA pullback accelerating into weekly support.
    Hypothesis: uptrend dip + recent selling velocity = highest mean-reversion momentum."""
    if sig_s310_s292_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now = float(tsla['close'].iloc[i])
    close_5d  = float(tsla['close'].iloc[i - 5])
    return 1 if (close_now - close_5d) / close_5d <= -0.05 else 0


def sig_s318_s292_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S318: S292 (above200MA) + NR7 (narrowest range of last 7 bars).
    Uptrend + weekly support + narrowest range = maximum compression in uptrend.
    S217 (S214+NR7) was 7/8yr WR=85%. Does uptrend context improve it?"""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 7:
        return 1
    cur_range = float(tsla['high'].iloc[i]) - float(tsla['low'].iloc[i])
    prev_ranges = [float(tsla['high'].iloc[i - j]) - float(tsla['low'].iloc[i - j])
                   for j in range(1, 7)]
    return 1 if cur_range <= min(prev_ranges) else 0


def sig_s319_s292_persistent_compress(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S319: S292 (above200MA) + persistent compression (3+ days ATR compressed).
    S189 (persistent compression) achieved 8/9yr. Add uptrend context.
    Hypothesis: uptrend + weekly support + multi-day ATR compression = sustained coiling."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    compress_streak = 0
    for j in range(5):
        if i - j < 20:
            break
        c = _atr_components(tsla, i - j)
        if c is None:
            break
        _, atr_5, _, atr_20 = c
        if atr_5 < 0.75 * atr_20:
            compress_streak += 1
        else:
            break
    return 1 if compress_streak >= 3 else 0


def sig_s320_s292_monthly_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S320: S292 (above200MA + weekly support + ATR extreme + VIX18) + monthly near lower.
    3-TF uptrend: TSLA above 200MA but approaching both weekly AND monthly lower channels.
    Hypothesis: multiple TF convergence in an overall uptrend = extreme dip-buy zone."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _monthly_channel_near_lower(tsla, i, n_months=12, frac=0.30) else 0


SIGNALS_P8H: List[Tuple] = [
    # Phase 8H — below200MA bear-trend signals + S310 extensions + S292 new filters
    ('S311_s291_vix_rec',        sig_s311_s291_vix_rec,       10, 0.20, 50),
    ('S312_s291_compressed',     sig_s312_s291_compressed,    10, 0.20, 50),
    ('S313_s291_consec_down',    sig_s313_s291_consec_down,   10, 0.20, 50),
    ('S314_s291_5d_down8pct',    sig_s314_s291_5d_down8pct,  10, 0.20, 50),
    ('S315_s310_vix_rec',        sig_s315_s310_vix_rec,       10, 0.20, 50),
    ('S316_s310_compressed',     sig_s316_s310_compressed,    10, 0.20, 50),
    ('S317_s310_5d_down5pct',    sig_s317_s310_5d_down5pct,  10, 0.20, 50),
    ('S318_s292_nr7',            sig_s318_s292_nr7,           10, 0.20, 50),
    ('S319_s292_persist_compress', sig_s319_s292_persistent_compress, 10, 0.20, 50),
    ('S320_s292_monthly_low',    sig_s320_s292_monthly_low,   10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H)


# ── Phase 8I — S319 extensions + S312 bear-regime + cross-family combos ────────
# S319 (above200MA+persist_compress) avg=$107K/trade is the highest-quality IS signal.
# S312 (below200MA+compressed, 5/5yr WR=88%) is the best bear-regime signal.
# Extend both into combinations with VIX, 5-day momentum, and SPY filters.

def sig_s321_s319_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S321: S319 (above200MA+persist_compress, avg=$107K) + VIX recovery.
    The highest avg/trade signal + optimal VIX timing = ultimate quality filter.
    Hypothesis: persistent coiling in uptrend + VIX cooling = multi-day supply exhaustion."""
    if sig_s319_s292_persistent_compress(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s322_s319_5d_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S322: S319 (above200MA+persist_compress) + 5% 5-day decline.
    3+ days compressed + declining 5% = quiet drift to support with no recovery attempts.
    Hypothesis: sellers not panicking (compressed) but persistently offering = final flush."""
    if sig_s319_s292_persistent_compress(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now = float(tsla['close'].iloc[i])
    close_5d  = float(tsla['close'].iloc[i - 5])
    return 1 if (close_now - close_5d) / close_5d <= -0.05 else 0


def sig_s323_s319_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S323: S319 (above200MA+persist_compress) + below 50d SMA.
    Uptrend + 50MA pullback + persistent compression at weekly support.
    Hypothesis: stock dipped below 50MA, compressed for 3+ days → spring release."""
    if sig_s319_s292_persistent_compress(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    close = float(tsla['close'].iloc[i])
    ma50  = float(tsla['close'].iloc[i - 50:i].mean())
    return 1 if close < ma50 else 0


def sig_s324_s312_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S324: S312 (below200MA+compressed, 5/5yr WR=88%) + VIX recovery.
    Best bear-regime signal + VIX timing.
    Hypothesis: compression in downtrend at weekly support + VIX cooling = temporary bottom."""
    if sig_s312_s291_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s325_s312_5d_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S325: S312 (below200MA+compressed) + 5% 5-day decline.
    Bear-trend compressed + short-term decline to weekly support.
    Hypothesis: bear + compressed + declining into support = multiple seller exhaustion."""
    if sig_s312_s291_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now = float(tsla['close'].iloc[i])
    close_5d  = float(tsla['close'].iloc[i - 5])
    return 1 if (close_now - close_5d) / close_5d <= -0.05 else 0


def sig_s326_s312_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S326: S312 (below200MA+compressed) + SPY below 50d SMA.
    Both TSLA and SPY in downtrend, with TSLA compressed at weekly support.
    Hypothesis: double-bear + compression = maximum institutional mean-reversion buying."""
    if sig_s312_s291_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_sma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close < spy_sma50 else 0


def sig_s327_s315_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S327: S315 (above200MA+below50MA+VIX_rec, 100% WR) + compressed ATR.
    4-way filter: above200MA + below50MA + VIX recovery + ATR compression.
    Hypothesis: all 4 factors = absolute premium setup in uptrend with 50MA pullback."""
    if sig_s315_s310_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s328_s316_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S328: S316 (above200MA+below50MA+compressed, 5/5yr WR=100%) + VIX recovery.
    Already 100% WR 5/5yr. Adding VIX timing to see if we can raise avg/trade.
    Hypothesis: keeps only the VIX-timed subset of the 100% WR signal."""
    if sig_s316_s310_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s329_s316_5d_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S329: S316 (above200MA+below50MA+compressed) + 5% 5-day decline.
    4-way combo: uptrend + 50MA break + compressed + recent 5% decline.
    Hypothesis: compressed stock that has been declining to support in uptrend."""
    if sig_s316_s310_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now = float(tsla['close'].iloc[i])
    close_5d  = float(tsla['close'].iloc[i - 5])
    return 1 if (close_now - close_5d) / close_5d <= -0.05 else 0


def sig_s330_s291_opex_week(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S330: S291 (S215+below200MA, bear trend) + OPEX week.
    Bear-trend weekly support bounce during option expiration.
    Hypothesis: OPEX pinning effect works in downtrends too (creates temporary floor)."""
    if sig_s291_s215_below200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    dt = tsla.index[i]
    to_next, from_last = _opex_proximity(dt)
    return 1 if (to_next <= 5 or from_last <= 5) else 0


SIGNALS_P8I: List[Tuple] = [
    # Phase 8I — S319 extensions + S312 bear-regime + cross-family combos
    ('S321_s319_vix_rec',       sig_s321_s319_vix_rec,       10, 0.20, 50),
    ('S322_s319_5d_down',       sig_s322_s319_5d_down,       10, 0.20, 50),
    ('S323_s319_below50ma',     sig_s323_s319_below50ma,     10, 0.20, 50),
    ('S324_s312_vix_rec',       sig_s324_s312_vix_rec,       10, 0.20, 50),
    ('S325_s312_5d_down',       sig_s325_s312_5d_down,       10, 0.20, 50),
    ('S326_s312_spy_weak',      sig_s326_s312_spy_weak,      10, 0.20, 50),
    ('S327_s315_compressed',    sig_s327_s315_compressed,    10, 0.20, 50),
    ('S328_s316_vix_rec',       sig_s328_s316_vix_rec,       10, 0.20, 50),
    ('S329_s316_5d_down',       sig_s329_s316_5d_down,       10, 0.20, 50),
    ('S330_s291_opex_week',     sig_s330_s291_opex_week,     10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I)


# ── Phase 8J — Relative strength vs SPY, MACD, persistence confirm, seasonal ──
# New dimensions:
#   Relative performance: TSLA vs SPY return divergence (individual vs broad market)
#   MACD histogram: momentum reversal indicator at weekly support
#   Persistence confirm: require 2-day signal before entry (reduce noise)
#   Seasonal: summer (Jul-Aug) and December effects

def _macd_histogram(tsla, i, fast: int = 12, slow: int = 26, signal_period: int = 9) -> Optional[float]:
    """Return MACD histogram (MACD line - signal line). Positive = bullish momentum."""
    if i < slow + signal_period:
        return None
    closes = tsla['close'].iloc[:i + 1].astype(float)
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    return float(macd_line.iloc[-1] - signal_line.iloc[-1])


def sig_s331_s215_tsla_rel_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S331: S215 + TSLA underperforming SPY by 5%+ over last 5 trading days.
    TSLA individual weakness at weekly support while broader market less weak.
    Hypothesis: excess individual selling = mean-reversion back to market performance."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    tsla_5d = (float(tsla['close'].iloc[i]) - float(tsla['close'].iloc[i-5])) / float(tsla['close'].iloc[i-5])
    spy_5d  = (float(spy['close'].iloc[i])  - float(spy['close'].iloc[i-5]))  / float(spy['close'].iloc[i-5])
    return 1 if (tsla_5d - spy_5d) <= -0.05 else 0


def sig_s332_s292_tsla_rel_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S332: S292 (above200MA) + TSLA underperforming SPY by 5%+ over 5 days.
    Uptrend + individual excess weakness vs market = strongest mean-reversion.
    Hypothesis: TSLA overshooting down vs market in an uptrend = elastic band snapping back."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    tsla_5d = (float(tsla['close'].iloc[i]) - float(tsla['close'].iloc[i-5])) / float(tsla['close'].iloc[i-5])
    spy_5d  = (float(spy['close'].iloc[i])  - float(spy['close'].iloc[i-5]))  / float(spy['close'].iloc[i-5])
    return 1 if (tsla_5d - spy_5d) <= -0.05 else 0


def sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S333: S215 + MACD histogram turning less negative (today > yesterday by 0.5+).
    MACD histogram rising = momentum starting to shift, even if still negative.
    Hypothesis: early momentum reversal at weekly support = leading indicator of bounce."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    # MACD histogram improving (rising) = bullish momentum shift
    improvement = hist_now - hist_prev
    return 1 if improvement > 0 else 0


def sig_s334_s292_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S334: S292 (above200MA) + MACD histogram turning positive (improving momentum).
    Uptrend + weekly support + MACD starting to turn = early bounce signal.
    Hypothesis: momentum shift in uptrend at support = higher conviction entry."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


def sig_s335_s215_dec_seasonal(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S335: S215 + December seasonality (tax-loss buying, Santa rally, year-end window).
    December = historically strong for equities (Santa Claus rally effect).
    Hypothesis: weekly support bounce in December amplified by seasonal fund flows."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if tsla.index[i].month == 12 else 0


def sig_s336_s215_summer(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S336: S215 + summer (July-August) seasonality.
    Summer = lower volume, mean-reversion driven, trending less.
    Hypothesis: weekly support bounce in low-volume summer = higher success rate."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    m = tsla.index[i].month
    return 1 if m in (7, 8) else 0


def sig_s337_s292_tsla_rel_strong(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S337: S292 + TSLA outperforming SPY by 2%+ over 5 days (TSLA showing resilience).
    Uptrend + weekly support + TSLA holding up better than market = institutional support.
    Hypothesis: relative strength at support = buyers accumulating before breakout up."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    tsla_5d = (float(tsla['close'].iloc[i]) - float(tsla['close'].iloc[i-5])) / float(tsla['close'].iloc[i-5])
    spy_5d  = (float(spy['close'].iloc[i])  - float(spy['close'].iloc[i-5]))  / float(spy['close'].iloc[i-5])
    return 1 if (tsla_5d - spy_5d) >= 0.02 else 0


def sig_s338_s215_q1(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S338: S215 + Q1 seasonality (January-March).
    Q1 = new year fund deployment, January effect, earnings season.
    Hypothesis: weekly support bounce in Q1 amplified by fund reallocation buying."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    m = tsla.index[i].month
    return 1 if m in (1, 2, 3) else 0


def sig_s339_s292_q2(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S339: S292 (above200MA) + Q2 (April-June) seasonality.
    Q2 = earnings season peak, pre-summer. Test if seasonal timing helps.
    Hypothesis: uptrend dip to weekly support in Q2 = near-term seasonal tailwind."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    m = tsla.index[i].month
    return 1 if m in (4, 5, 6) else 0


def sig_s340_s292_q4(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S340: S292 + Q4 (October-December) seasonality.
    Q4 = year-end performance chasing, tax-loss harvesting reversal.
    Hypothesis: October dips + November-December rally cycle amplifies weekly support."""
    if sig_s292_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    m = tsla.index[i].month
    return 1 if m in (10, 11, 12) else 0


SIGNALS_P8J: List[Tuple] = [
    # Phase 8J — Relative strength, MACD, seasonality
    ('S331_s215_tsla_rel_weak',  sig_s331_s215_tsla_rel_weak,  10, 0.20, 50),
    ('S332_s292_tsla_rel_weak',  sig_s332_s292_tsla_rel_weak,  10, 0.20, 50),
    ('S333_s215_macd_turning',   sig_s333_s215_macd_turning,   10, 0.20, 50),
    ('S334_s292_macd_turning',   sig_s334_s292_macd_turning,   10, 0.20, 50),
    ('S335_s215_dec_seasonal',   sig_s335_s215_dec_seasonal,   10, 0.20, 50),
    ('S336_s215_summer',         sig_s336_s215_summer,         10, 0.20, 50),
    ('S337_s292_tsla_rel_strong', sig_s337_s292_tsla_rel_strong, 10, 0.20, 50),
    ('S338_s215_q1',             sig_s338_s215_q1,             10, 0.20, 50),
    ('S339_s292_q2',             sig_s339_s292_q2,             10, 0.20, 50),
    ('S340_s292_q4',             sig_s340_s292_q4,             10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J)


# ── Phase 8K — MACD + Q4 + relative-weakness extensions ──────────────────────
# S333 (MACD turning, $1.51M, 8/8yr) is a major new indicator — extend aggressively.
# S340 (Q4, 100% WR) and S332 (relative weak in uptrend, 100% WR) need extension.

def sig_s341_s333_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S341: S333 (S215+MACD_turning, 8/8yr WR=86%) + compressed ATR.
    MACD momentum shift + ATR compression at weekly support.
    Hypothesis: MACD turning + compression = both momentum AND structure confirm bounce."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s342_s333_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S342: S333 (S215+MACD_turning) + VIX recovery.
    MACD momentum turning at weekly support + macro fear cooling simultaneously.
    Hypothesis: momentum shift aligned with VIX cooling = both signals confirm reversal."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s343_s333_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S343: S333 (S215+MACD_turning) + above 200d MA.
    Best MACD signal in uptrend context.
    Hypothesis: momentum turn at weekly support in bull trend = highest conviction."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s344_s333_5d_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S344: S333 (S215+MACD_turning) + 5% 5-day decline.
    MACD turning + recent velocity of decline + weekly support.
    Hypothesis: stock declining then MACD starts turning = first sign of exhaustion."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now = float(tsla['close'].iloc[i])
    close_5d  = float(tsla['close'].iloc[i - 5])
    return 1 if (close_now - close_5d) / close_5d <= -0.05 else 0


def sig_s345_s340_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S345: S340 (S292+Q4, 100% WR) + compressed ATR.
    Q4 uptrend + compression at weekly support.
    Hypothesis: year-end compression at support in bull trend = spring before Q4 rally."""
    if sig_s340_s292_q4(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s346_s340_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S346: S340 (S292+Q4, 100% WR) + VIX recovery.
    Q4 uptrend + VIX cooling = year-end seasonal + macro recovery timing.
    Hypothesis: institutional rebalancing + VIX drop = strong Q4 entry signal."""
    if sig_s340_s292_q4(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s347_s332_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S347: S332 (S292+TSLA rel weak, 100% WR) + compressed ATR.
    Uptrend + relative underperformance + compression at weekly support.
    Hypothesis: elastic band pulled furthest from SPY + compressed = max return potential."""
    if sig_s332_s292_tsla_rel_weak(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s348_s332_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S348: S332 (S292+TSLA rel weak) + VIX recovery.
    Uptrend + individual excess weakness + VIX cooling simultaneously.
    Hypothesis: individual relative capitulation + macro recovery = snap-back compound."""
    if sig_s332_s292_tsla_rel_weak(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s349_s333_spyweak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S349: S333 (S215+MACD_turning) + SPY below 50d SMA.
    MACD turning at weekly support during macro weakness.
    Hypothesis: market momentum shift at individual support + broad market weak = excess."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_sma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close < spy_sma50 else 0


def sig_s350_s333_monthly_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S350: S333 (S215+MACD_turning) + monthly near lower channel.
    MACD momentum turn at weekly support that also aligns with monthly support.
    Hypothesis: 2-TF structural support + MACD = momentum aligning with structure."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _monthly_channel_near_lower(tsla, i, n_months=12, frac=0.30) else 0


SIGNALS_P8K: List[Tuple] = [
    # Phase 8K — MACD extensions + Q4/relative-weakness combos
    ('S341_s333_compressed',    sig_s341_s333_compressed,    10, 0.20, 50),
    ('S342_s333_vix_rec',       sig_s342_s333_vix_rec,       10, 0.20, 50),
    ('S343_s333_above200ma',    sig_s343_s333_above200ma,    10, 0.20, 50),
    ('S344_s333_5d_down',       sig_s344_s333_5d_down,       10, 0.20, 50),
    ('S345_s340_compressed',    sig_s345_s340_compressed,    10, 0.20, 50),
    ('S346_s340_vix_rec',       sig_s346_s340_vix_rec,       10, 0.20, 50),
    ('S347_s332_compressed',    sig_s347_s332_compressed,    10, 0.20, 50),
    ('S348_s332_vix_rec',       sig_s348_s332_vix_rec,       10, 0.20, 50),
    ('S349_s333_spyweak',       sig_s349_s333_spyweak,       10, 0.20, 50),
    ('S350_s333_monthly_low',   sig_s350_s333_monthly_low,   10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K)


# ── Phase 8L — 3-way MACD combos + ADX + RSI-turn + new dimensions ────────────
# S341 (MACD+compressed, 8/8yr, WR=91%, $834K) and S349 (MACD+SPY weak, 6/6yr,
# WR=92%, $1.07M) are the new top signals. Extend aggressively into 3-way combos.
# Also introduce ADX (low = non-trending = good for bounces) as a new dimension.

def _adx(tsla, i, period: int = 14) -> Optional[float]:
    """Average Directional Index — low ADX (<20) = non-trending = range-bound."""
    if i < period * 2:
        return None
    highs  = tsla['high'].iloc[i - period * 2:i + 1].values.astype(float)
    lows   = tsla['low'].iloc[i - period * 2:i + 1].values.astype(float)
    closes = tsla['close'].iloc[i - period * 2:i + 1].values.astype(float)
    tr_vals, pdm_vals, ndm_vals = [], [], []
    for j in range(1, len(closes)):
        h, l, pc = highs[j], lows[j], closes[j - 1]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        pdm = max(h - highs[j - 1], 0) if (h - highs[j - 1]) > (lows[j - 1] - l) else 0
        ndm = max(lows[j - 1] - l, 0) if (lows[j - 1] - l) > (h - highs[j - 1]) else 0
        tr_vals.append(tr)
        pdm_vals.append(pdm)
        ndm_vals.append(ndm)
    # Smooth with Wilder (EMA with alpha=1/period)
    def wilder_smooth(vals, p):
        s = sum(vals[:p])
        res = [s]
        for v in vals[p:]:
            s = s - s / p + v
            res.append(s)
        return res
    tr_s  = wilder_smooth(tr_vals,  period)
    pdm_s = wilder_smooth(pdm_vals, period)
    ndm_s = wilder_smooth(ndm_vals, period)
    if len(tr_s) == 0 or tr_s[-1] < 1e-6:
        return None
    pdi = 100 * pdm_s[-1] / tr_s[-1]
    ndi = 100 * ndm_s[-1] / tr_s[-1]
    denom = pdi + ndi
    if denom < 1e-6:
        return None
    dx = 100 * abs(pdi - ndi) / denom
    # ADX = smoothed DX (use simple mean of last period DX values for simplicity)
    return dx   # approximate ADX via last DX; directional enough for filtering


def _rsi_turning_up(tsla, i, rsi_series, oversold: float = 35.0) -> bool:
    """RSI turning up from oversold: RSI was < oversold_threshold 1-5 bars ago
    and is now rising (current > prev). Signals early momentum reversal."""
    if i < 6:
        return False
    rsi_now  = rsi_series.iloc[i]
    rsi_prev = rsi_series.iloc[i - 1]
    if pd.isna(rsi_now) or pd.isna(rsi_prev):
        return False
    if rsi_now <= rsi_prev:   # must be rising
        return False
    # Was RSI below oversold_threshold in last 1-5 bars?
    for k in range(1, 6):
        if i - k < 0:
            break
        past = rsi_series.iloc[i - k]
        if not pd.isna(past) and past < oversold:
            return True
    return False


def sig_s351_s341_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S351: S341 (MACD+compressed, 8/8yr) + VIX recovery.
    3-way: MACD momentum shift + ATR compression + macro fear cooling.
    Hypothesis: all 3 confirm bounce → highest precision subset of S341."""
    if sig_s341_s333_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s352_s341_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S352: S341 (MACD+compressed, 8/8yr) + above 200d MA.
    3-way: MACD turning + compression + bull-trend regime.
    Hypothesis: best balanced signal (S341) filtered to uptrend = even higher win rate."""
    if sig_s341_s333_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s353_s349_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S353: S349 (MACD+SPY weak, 6/6yr, $1.07M) + compressed ATR.
    3-way: MACD turning + broad weakness + individual compression.
    Hypothesis: macro weakness + stock compression + momentum turn = highest-pf subset."""
    if sig_s349_s333_spyweak(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s354_s349_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S354: S349 (MACD+SPY weak) + VIX recovery.
    3-way: MACD turning + SPY below 50MA + VIX cooling.
    Hypothesis: SPY structural weakness + VIX macro cooling = double macro confirm."""
    if sig_s349_s333_spyweak(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s355_s333_below200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S355: S333 (MACD turning at weekly support) + below 200d MA.
    MACD momentum signal in bear-regime context. Fires in 2025 (TSLA below 200MA).
    Hypothesis: momentum turn in bear trend = oversold bounce, short but sharp."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close < ma200 else 0


def sig_s356_s342_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S356: S342 (MACD+VIX rec, 100% WR, 5/5yr) + above 200d MA.
    Best 100% WR signal filtered to confirmed uptrend.
    Hypothesis: tightest combination — all 3 macro/momentum/regime aligned."""
    if sig_s342_s333_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s357_s342_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S357: S342 (MACD+VIX rec, 100% WR) + compressed ATR.
    3-way: MACD turning + VIX cooling + ATR compression simultaneously.
    Hypothesis: tightest structure + momentum + macro = maximum convergence."""
    if sig_s342_s333_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s358_s341_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S358: S341 (MACD+compressed) + SPY below 50d SMA.
    3-way: MACD turning + ATR compression + broad market weakness.
    Hypothesis: MACD bounce signal most powerful when SPY structure also weak."""
    if sig_s341_s333_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_sma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close < spy_sma50 else 0


def sig_s359_s343_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S359: S343 (MACD+above200MA, uptrend) + VIX recovery.
    Triple: MACD turning + confirmed bull regime + VIX cooling.
    Hypothesis: 3-way momentum+regime+macro = maximum bull-regime precision."""
    if sig_s343_s333_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s360_s333_low_adx(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S360: S333 (MACD turning at weekly support) + low ADX (<20).
    NEW DIMENSION: Low ADX = stock is range-bound/consolidating, not trending.
    Hypothesis: MACD turn at weekly support + range-bound structure = clean bounce
    with no directional momentum fighting against it."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    adx = _adx(tsla, i, period=14)
    if adx is None:
        return 1   # not enough data — allow
    return 1 if adx < 22 else 0


SIGNALS_P8L: List[Tuple] = [
    # Phase 8L — 3-way MACD combos + ADX + bear-regime MACD
    ('S351_s341_vix_rec',       sig_s351_s341_vix_rec,       10, 0.20, 50),
    ('S352_s341_above200ma',    sig_s352_s341_above200ma,    10, 0.20, 50),
    ('S353_s349_compressed',    sig_s353_s349_compressed,    10, 0.20, 50),
    ('S354_s349_vix_rec',       sig_s354_s349_vix_rec,       10, 0.20, 50),
    ('S355_s333_below200ma',    sig_s355_s333_below200ma,    10, 0.20, 50),
    ('S356_s342_above200ma',    sig_s356_s342_above200ma,    10, 0.20, 50),
    ('S357_s342_compressed',    sig_s357_s342_compressed,    10, 0.20, 50),
    ('S358_s341_spy_weak',      sig_s358_s341_spy_weak,      10, 0.20, 50),
    ('S359_s343_vix_rec',       sig_s359_s343_vix_rec,       10, 0.20, 50),
    ('S360_s333_low_adx',       sig_s360_s333_low_adx,       10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L)


# ── Phase 8M — ADX extensions + RSI-turn + SPY momentum + VIX percentile ──────
# S360 (MACD+low ADX, 7/7yr, WR=82%, PF=12.74) confirms ADX as a new dimension.
# S355 (MACD+below200MA, 6/6yr, 100% WR) is our best bear-regime signal.
# New: RSI turning up from oversold, SPY consecutive up days, VIX elevated percentile.

def _spy_consec_up(spy, i, n: int = 3) -> bool:
    """SPY has closed up n consecutive days (momentum confirmation)."""
    if i < n:
        return False
    for k in range(n):
        if spy['close'].iloc[i - k] <= spy['close'].iloc[i - k - 1]:
            return False
    return True


def _vix_elevated_pct(vix, i, window: int = 252, pct: float = 0.70) -> bool:
    """VIX is in top (1-pct)% of its trailing window — structurally elevated fear."""
    if i < window:
        return True   # not enough history → assume elevated
    vix_now  = float(vix['close'].iloc[i])
    vix_hist = vix['close'].iloc[i - window:i].values.astype(float)
    return bool(vix_now > np.percentile(vix_hist, pct * 100))


def sig_s361_s360_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S361: S360 (MACD+low ADX, 7/7yr) + VIX recovery.
    3-way: MACD turning + range-bound structure + macro fear cooling.
    Hypothesis: ADX confirms no counter-trend momentum; VIX cooling = macro clear."""
    if sig_s360_s333_low_adx(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s362_s360_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S362: S360 (MACD+low ADX) + compressed ATR.
    3-way: range-bound ADX + ATR compression + MACD turn = double structure confirm.
    Hypothesis: ADX says non-trending, ATR says tight range — coiled spring."""
    if sig_s360_s333_low_adx(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s363_s360_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S363: S360 (MACD+low ADX) + above 200d MA.
    ADX-based signal in confirmed uptrend.
    Hypothesis: range-bound non-trending + in bull regime = best bounce setup."""
    if sig_s360_s333_low_adx(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s364_s360_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S364: S360 (MACD+low ADX) + SPY below 50d SMA.
    ADX signal during macro weakness — bear-market bounce.
    Hypothesis: TSLA range-bound with MACD turn + SPY weak = excess discount, snap-back."""
    if sig_s360_s333_low_adx(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_sma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close < spy_sma50 else 0


def sig_s365_s355_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S365: S355 (MACD+below200MA, best bear signal) + compressed ATR.
    Bear regime + range contraction + MACD turn.
    Hypothesis: bear trend pausing (compression) + MACD turning up = exhaustion bounce."""
    if sig_s355_s333_below200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s366_s355_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S366: S355 (MACD+below200MA) + VIX recovery.
    Bear-regime MACD bounce + VIX macro recovery simultaneously.
    Hypothesis: individual momentum turn in bear trend + macro relief = double catalyst."""
    if sig_s355_s333_below200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s367_s215_rsi_turn(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S367: S215 (weekly support) + RSI turning up from oversold (<35).
    NEW DIMENSION: RSI was oversold 1-5 bars ago and is now rising.
    Hypothesis: weekly support at RSI exhaustion point + early RSI recovery = best entry."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _rsi_turning_up(tsla, i, rt, oversold=35.0) else 0


def sig_s368_s333_rsi_turn(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S368: S333 (MACD turning at weekly support) + RSI turning up from oversold.
    Double momentum confirmation: MACD histogram + RSI both reversing simultaneously.
    Hypothesis: MACD+RSI double momentum reversal at weekly support = highest precision."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _rsi_turning_up(tsla, i, rt, oversold=35.0) else 0


def sig_s369_s215_spy_consec_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S369: S215 (weekly support) + SPY 3 consecutive up days while TSLA at support.
    NEW: SPY showing consecutive strength while TSLA hasn't followed yet.
    Hypothesis: SPY momentum building + TSLA lagging at support = pull-up imminent."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_consec_up(spy, i, n=3) else 0


def sig_s370_s333_spy_consec_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S370: S333 (MACD turning) + SPY 3 consecutive up days.
    MACD momentum turn at weekly support + SPY building momentum.
    Hypothesis: MACD reversal confirmed by SPY upside momentum = best lift setup."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_consec_up(spy, i, n=3) else 0


SIGNALS_P8M: List[Tuple] = [
    # Phase 8M — ADX extensions + bear MACD + RSI-turn + SPY momentum
    ('S361_s360_vix_rec',       sig_s361_s360_vix_rec,       10, 0.20, 50),
    ('S362_s360_compressed',    sig_s362_s360_compressed,    10, 0.20, 50),
    ('S363_s360_above200ma',    sig_s363_s360_above200ma,    10, 0.20, 50),
    ('S364_s360_spy_weak',      sig_s364_s360_spy_weak,      10, 0.20, 50),
    ('S365_s355_compressed',    sig_s365_s355_compressed,    10, 0.20, 50),
    ('S366_s355_vix_rec',       sig_s366_s355_vix_rec,       10, 0.20, 50),
    ('S367_s215_rsi_turn',      sig_s367_s215_rsi_turn,      10, 0.20, 50),
    ('S368_s333_rsi_turn',      sig_s368_s333_rsi_turn,      10, 0.20, 50),
    ('S369_s215_spy_consec_up', sig_s369_s215_spy_consec_up, 10, 0.20, 50),
    ('S370_s333_spy_consec_up', sig_s370_s333_spy_consec_up, 10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M)


# ── Phase 8N — ADX top extensions + NR7 + volume spike + gap-down ─────────────
# S362 (ADX+compressed, 6/6yr) and S361 (ADX+VIX, PF=269, 4/4yr) are the new leaders.
# New dimensions: NR7 (narrowest-range day), volume spike, gap-down >3%.
# Also extend S365 (best bear signal: MACD+below200MA+compressed → 100% WR).

def _nr7(tsla, i) -> bool:
    """Narrowest Range in 7 days: today's (high-low) < all of last 6 days.
    NR7 = stock about to make a directional move. At channel support → expect up."""
    if i < 7:
        return False
    today_range = float(tsla['high'].iloc[i]) - float(tsla['low'].iloc[i])
    for k in range(1, 7):
        prev_range = float(tsla['high'].iloc[i - k]) - float(tsla['low'].iloc[i - k])
        if today_range >= prev_range:
            return False
    return True


def _volume_spike(tsla, i, mult: float = 2.0, period: int = 20) -> bool:
    """Today's volume >= mult × 20d average (capitulation/exhaustion volume)."""
    if i < period:
        return False
    avg_vol = float(tsla['volume'].iloc[i - period:i].mean())
    if avg_vol < 1:
        return False
    return float(tsla['volume'].iloc[i]) >= mult * avg_vol


def _gap_down(tsla, i, pct: float = 0.02) -> bool:
    """Today's open < yesterday's close by >= pct (gap-down = panic open)."""
    if i < 1:
        return False
    prev_close = float(tsla['close'].iloc[i - 1])
    today_open = float(tsla['open'].iloc[i])
    return (prev_close - today_open) / prev_close >= pct


def sig_s371_s362_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S371: S362 (ADX+compressed, 6/6yr) + VIX recovery.
    4-way: MACD + low ADX + compressed + VIX cooling.
    Hypothesis: all 4 structure/momentum/macro dimensions aligned simultaneously."""
    if sig_s362_s360_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s372_s362_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S372: S362 (ADX+compressed) + above 200d MA.
    Best ADX extension in uptrend context.
    Hypothesis: range-bound + compressed in bull regime = spring-load setup."""
    if sig_s362_s360_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s373_s361_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S373: S361 (ADX+VIX rec, PF=269) + above 200d MA.
    Extreme-PF signal filtered to uptrend.
    Hypothesis: ADX+VIX signal most powerful in confirmed bull trend."""
    if sig_s361_s360_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s374_s365_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S374: S365 (bear MACD+compressed, 100% WR, 5/5yr) + VIX recovery.
    3-way bear: MACD turn + compression + VIX cooling in bear regime.
    Hypothesis: tightest bear signal with all 3 layers = maximum precision."""
    if sig_s365_s355_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s375_s215_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S375: S215 (weekly support + VIX>=18) + NR7 (narrowest range in 7d).
    NEW DIMENSION: NR7 at channel support = range contraction before directional move.
    Hypothesis: tightest day at weekly support = maximum compression before bounce."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s376_s333_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S376: S333 (MACD turning at weekly support) + NR7.
    NR7 + MACD turn = structure AND momentum converge on same day.
    Hypothesis: momentum turning on narrowest day = highest-precision entry."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s377_s215_vol_spike(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S377: S215 (weekly support) + volume spike (2× 20d avg).
    Volume capitulation at weekly support — smart-money absorption.
    Hypothesis: panic selling volume at support = seller exhaustion, reversal imminent."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_spike(tsla, i, mult=2.0) else 0


def sig_s378_s333_vol_spike(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S378: S333 (MACD turning) + volume spike.
    MACD turn on capitulation volume at weekly support = strongest reversal signal.
    Hypothesis: momentum confirming with high-conviction institutional volume."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_spike(tsla, i, mult=2.0) else 0


def sig_s379_s215_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S379: S215 (weekly support) + gap-down open ≥2%.
    Weekly support + panic gap open = extreme entry opportunity.
    Hypothesis: institutional gap-fill tendency at weekly support = 1-2d snap-back."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _gap_down(tsla, i, pct=0.02) else 0


def sig_s380_s333_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S380: S333 (MACD turning) + gap-down open ≥2%.
    MACD momentum turn + gap-down at weekly support = gap-fill + momentum confluence.
    Hypothesis: MACD confirms buy side + gap below creates gap-fill target."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _gap_down(tsla, i, pct=0.02) else 0


SIGNALS_P8N: List[Tuple] = [
    # Phase 8N — ADX top extensions + NR7 + volume spike + gap-down
    ('S371_s362_vix_rec',       sig_s371_s362_vix_rec,       10, 0.20, 50),
    ('S372_s362_above200ma',    sig_s372_s362_above200ma,    10, 0.20, 50),
    ('S373_s361_above200ma',    sig_s373_s361_above200ma,    10, 0.20, 50),
    ('S374_s365_vix_rec',       sig_s374_s365_vix_rec,       10, 0.20, 50),
    ('S375_s215_nr7',           sig_s375_s215_nr7,           10, 0.20, 50),
    ('S376_s333_nr7',           sig_s376_s333_nr7,           10, 0.20, 50),
    ('S377_s215_vol_spike',     sig_s377_s215_vol_spike,     10, 0.20, 50),
    ('S378_s333_vol_spike',     sig_s378_s333_vol_spike,     10, 0.20, 50),
    ('S379_s215_gap_down',      sig_s379_s215_gap_down,      10, 0.20, 50),
    ('S380_s333_gap_down',      sig_s380_s333_gap_down,      10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N)


# ── Phase 8O — NR7 extensions + volume-absorption + VIX percentile ────────────
# S375 (S215+NR7, 5/6yr, WR=89%, PF=33) is the breakout from Phase 8N.
# Extend NR7 aggressively. Also: volume absorption (vol spike + NR7 together).
# New: VIX structurally elevated (>70th percentile trailing 1yr) as regime filter.

def _persistent_nr(tsla, i, n: int = 2) -> bool:
    """NR for n consecutive days — price compressing for multiple sessions."""
    if i < n + 6:
        return False
    for d in range(n):
        if not _nr7(tsla, i - d):
            return False
    return True


def sig_s381_s375_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S381: S375 (S215+NR7, 5/6yr) + VIX recovery.
    3-way: weekly support + NR7 + macro fear cooling.
    Hypothesis: tightest day at support + VIX cooling = highest precision subset of S375."""
    if sig_s375_s215_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s382_s375_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S382: S375 (S215+NR7) + compressed ATR.
    Double compression: NR7 says today tightest in 7d + ATR says 5d tight vs 20d.
    Hypothesis: two independent compression measures both flagging = maximum squeeze."""
    if sig_s375_s215_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s383_s376_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S383: S376 (MACD+NR7, 4/5yr) + VIX recovery.
    3-way: MACD turning + NR7 + macro cooling.
    Hypothesis: momentum+structure+macro triple convergence on tightest day."""
    if sig_s376_s333_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s384_s376_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S384: S376 (MACD+NR7) + above 200d MA.
    Best MACD+NR7 signal in uptrend.
    Hypothesis: momentum+structure both confirm in bull trend = highest WR."""
    if sig_s376_s333_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s385_s375_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S385: S375 (S215+NR7) + above 200d MA.
    NR7 at weekly support in bull trend.
    Hypothesis: weekly support + tightest day + bull regime = spring-load in uptrend."""
    if sig_s375_s215_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s386_s377_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S386: S377 (S215+volume spike) + VIX recovery.
    3-way: weekly support + capitulation volume + VIX cooling.
    Hypothesis: heavy selling at support + VIX cooling = both sellers exhausted."""
    if sig_s377_s215_vol_spike(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s387_s377_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S387: S377 (S215+vol spike) + above 200d MA.
    Volume capitulation at weekly support in bull trend.
    Hypothesis: panic volume at support in uptrend = fast recovery."""
    if sig_s377_s215_vol_spike(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s388_s215_nr7_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S388: S215 (weekly support) + NR7 + compressed ATR.
    Triple compression: weekly support + today tightest + ATR compressed.
    Hypothesis: all three compression layers = maximum coiling energy."""
    if sig_s375_s215_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s389_s333_nr7_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S389: S333 (MACD turning) + NR7 + compressed ATR.
    4-way: weekly support + MACD momentum + NR7 + ATR compressed.
    Hypothesis: momentum turning on tightest compressed day = most precise entry ever."""
    if sig_s376_s333_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s390_s333_nr7_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S390: S333 (MACD turning) + NR7 + VIX recovery.
    4-way: MACD + weekly support + NR7 + macro cooling.
    Hypothesis: tightest momentum day at support + VIX cooling = 4-layer convergence."""
    if sig_s376_s333_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


SIGNALS_P8O: List[Tuple] = [
    # Phase 8O — NR7 extensions + volume absorption + multi-compression
    ('S381_s375_vix_rec',        sig_s381_s375_vix_rec,        10, 0.20, 50),
    ('S382_s375_compressed',     sig_s382_s375_compressed,     10, 0.20, 50),
    ('S383_s376_vix_rec',        sig_s383_s376_vix_rec,        10, 0.20, 50),
    ('S384_s376_above200ma',     sig_s384_s376_above200ma,     10, 0.20, 50),
    ('S385_s375_above200ma',     sig_s385_s375_above200ma,     10, 0.20, 50),
    ('S386_s377_vix_rec',        sig_s386_s377_vix_rec,        10, 0.20, 50),
    ('S387_s377_above200ma',     sig_s387_s377_above200ma,     10, 0.20, 50),
    ('S388_s215_nr7_compressed', sig_s388_s215_nr7_compressed, 10, 0.20, 50),
    ('S389_s333_nr7_compressed', sig_s389_s333_nr7_compressed, 10, 0.20, 50),
    ('S390_s333_nr7_vix_rec',    sig_s390_s333_nr7_vix_rec,    10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O)


# ── Phase 8P — NR7 cross-signals + milestone S400 ─────────────────────────────
# S382 (NR7+compressed 100% WR 5/5), S389 (MACD+NR7+compressed, 4/4yr, $111K avg).
# Now cross NR7 with high-n signals (S349, S292, S341) for larger trade counts.
# Milestone: S400 = 4-way in uptrend (MACD+compressed+NR7+above200MA).

def sig_s391_s382_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S391: S382 (NR7+compressed, 100% WR) + VIX recovery.
    3-way compression: NR7 + ATR compressed + VIX cooling.
    Hypothesis: triple-compressed + macro improvement = tightest precision signal."""
    if sig_s382_s375_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s392_s382_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S392: S382 (NR7+compressed, 100% WR) + above 200d MA.
    Double compression in confirmed uptrend.
    Hypothesis: NR7+ATR compressed in bull trend = coiled spring with tailwind."""
    if sig_s382_s375_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s393_s382_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S393: S382 (NR7+compressed) + SPY below 50d SMA.
    Double compression at weekly support during broad market weakness.
    Hypothesis: maximum range compression in bear market = snap-back potential."""
    if sig_s382_s375_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_sma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close < spy_sma50 else 0


def sig_s394_s349_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S394: S349 (MACD+SPY weak, 6/6yr, n=14) + NR7.
    Cross high-frequency signal with NR7 compression filter.
    Hypothesis: MACD+SPY weak fires often (14 trades) → NR7 filters to best entries."""
    if sig_s349_s333_spyweak(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s395_s376_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S395: S376 (MACD+NR7, 4/5yr) + SPY below 50d SMA.
    NR7 momentum signal during broad market weakness.
    Hypothesis: MACD+NR7 confluence + SPY weakness = excess discount entry."""
    if sig_s376_s333_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_sma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close < spy_sma50 else 0


def sig_s396_s215_nr7_vix22(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S396: S215 (weekly support) + NR7 + VIX >= 22 (elevated fear).
    NR7 at weekly support with VIX elevated (not just recovering).
    Hypothesis: structured tightest day at support when market structurally fearful."""
    if sig_s375_s215_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vix_now = float(vix['close'].iloc[i])
    return 1 if vix_now >= 22 else 0


def sig_s397_s375_5d_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S397: S375 (S215+NR7) + 5% 5-day decline.
    NR7 at weekly support after a meaningful pullback.
    Hypothesis: stock declined 5%+ then tightest day at support = seller exhaustion."""
    if sig_s375_s215_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 1
    close_now = float(tsla['close'].iloc[i])
    close_5d  = float(tsla['close'].iloc[i - 5])
    return 1 if (close_now - close_5d) / close_5d <= -0.05 else 0


def sig_s398_s375_monthly_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S398: S375 (S215+NR7) + monthly near lower channel.
    NR7 at weekly support that also aligns with monthly support.
    Hypothesis: 2-TF support (weekly+monthly) + NR7 = highest structural precision."""
    if sig_s375_s215_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _monthly_channel_near_lower(tsla, i, n_months=12, frac=0.30) else 0


def sig_s399_s362_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S399: S362 (ADX+compressed, 6/6yr) + NR7.
    Cross ADX signal with NR7 — both structural indicators.
    Hypothesis: ADX (non-trending) + ATR compressed + NR7 = triple structure convergence."""
    if sig_s362_s360_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s400_s333_nr7_compressed_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S400 MILESTONE: MACD turning + NR7 + compressed ATR + above 200d MA.
    4-layer uptrend signal: weekly support + MACD momentum + NR7 structure + bull regime.
    Hypothesis: all four layers in uptrend = maximum precision in best market conditions."""
    if sig_s389_s333_nr7_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


SIGNALS_P8P: List[Tuple] = [
    # Phase 8P — NR7 cross-combos + S400 milestone
    ('S391_s382_vix_rec',              sig_s391_s382_vix_rec,              10, 0.20, 50),
    ('S392_s382_above200ma',           sig_s392_s382_above200ma,           10, 0.20, 50),
    ('S393_s382_spy_weak',             sig_s393_s382_spy_weak,             10, 0.20, 50),
    ('S394_s349_nr7',                  sig_s394_s349_nr7,                  10, 0.20, 50),
    ('S395_s376_spy_weak',             sig_s395_s376_spy_weak,             10, 0.20, 50),
    ('S396_s215_nr7_vix22',            sig_s396_s215_nr7_vix22,            10, 0.20, 50),
    ('S397_s375_5d_down',              sig_s397_s375_5d_down,              10, 0.20, 50),
    ('S398_s375_monthly_low',          sig_s398_s375_monthly_low,          10, 0.20, 50),
    ('S399_s362_nr7',                  sig_s399_s362_nr7,                  10, 0.20, 50),
    ('S400_macd_nr7_comp_above200ma',  sig_s400_s333_nr7_compressed_above200ma, 10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P)


# ── Phase 8Q — SPY RSI + VIX percentile + DOW + TSLA losing streak ────────────
# Explore macro SPY RSI (market oversold), VIX structurally elevated vs 1yr history,
# day-of-week effect, and consecutive TSLA down bars (losing streak = exhaustion).

def _tsla_consec_down(tsla, i, n: int = 3) -> bool:
    """n consecutive TSLA daily closes declining (losing streak = seller exhaustion)."""
    if i < n:
        return False
    for k in range(n):
        if tsla['close'].iloc[i - k] >= tsla['close'].iloc[i - k - 1]:
            return False
    return True


def sig_s401_s215_spy_rsi_os(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S401: S215 (weekly support) + SPY RSI < 35 (macro oversold).
    NEW DIMENSION: SPY itself reaching oversold RSI territory while TSLA at support.
    Hypothesis: market RSI exhaustion at TSLA weekly support = broadest reversal signal."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    spy_rsi = rs.iloc[i]
    if pd.isna(spy_rsi):
        return 0
    return 1 if spy_rsi < 35 else 0


def sig_s402_s333_spy_rsi_os(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S402: S333 (MACD turning at weekly support) + SPY RSI < 40.
    MACD momentum turn + SPY approaching oversold.
    Hypothesis: momentum reversal when SPY structurally weak = best macro timing."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    spy_rsi = rs.iloc[i]
    if pd.isna(spy_rsi):
        return 0
    return 1 if spy_rsi < 40 else 0


def sig_s403_s215_spy_rsi_turn(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S403: S215 (weekly support) + SPY RSI turning up from < 40.
    SPY RSI recovery from oversold while TSLA at support.
    Hypothesis: SPY macro recovery turning + TSLA at support = tide coming in."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _rsi_turning_up(spy, i, rs, oversold=40.0) else 0


def sig_s404_s333_spy_rsi_turn(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S404: S333 (MACD turning) + SPY RSI turning up from < 40.
    Double macro recovery: TSLA MACD turning + SPY RSI turning simultaneously.
    Hypothesis: both individual and macro momentum reversing = synchronized reversal."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _rsi_turning_up(spy, i, rs, oversold=40.0) else 0


def sig_s405_s215_tsla_3down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S405: S215 (weekly support) + 3 consecutive TSLA down days.
    Losing streak at weekly support = seller exhaustion signal.
    Hypothesis: 3 consecutive down bars = momentum exhausting, bounce imminent."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=3) else 0


def sig_s406_s333_tsla_3down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S406: S333 (MACD turning at weekly support) + 3 consecutive TSLA down days.
    MACD turning up on the 3rd consecutive down day = classic reversal candle.
    Hypothesis: MACD momentum shift confirming on losing streak day = perfect entry."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=3) else 0


def sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S407: S215 (weekly support) + VIX > 70th percentile of trailing 1yr.
    VIX structurally elevated vs its own history — not just a spike but regime fear.
    Hypothesis: when VIX is structurally high, bounces at support are more violent."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s408_s333_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S408: S333 (MACD turning) + VIX > 70th percentile of trailing 1yr.
    MACD momentum reversal when market is in a structurally fearful regime.
    Hypothesis: MACD turn + persistent fear regime = contrarian momentum signal."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s409_s215_dow_monday(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S409: S215 (weekly support) + entry on Monday (day-of-week filter).
    DOW filter: Monday dips often reverse by week end (weekend risk premium unwind).
    Hypothesis: Monday panic sell at weekly support = best entry timing."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if tsla.index[i].dayofweek == 0 else 0   # 0 = Monday


def sig_s410_s333_dow_monday(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S410: S333 (MACD turning) + Monday entry.
    MACD momentum turn on a Monday — start-of-week reversal.
    Hypothesis: MACD confirming on Monday = strong signal with 4d momentum window."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if tsla.index[i].dayofweek == 0 else 0   # 0 = Monday


SIGNALS_P8Q: List[Tuple] = [
    # Phase 8Q — SPY RSI + VIX percentile + DOW + losing streak
    ('S401_s215_spy_rsi_os',    sig_s401_s215_spy_rsi_os,    10, 0.20, 50),
    ('S402_s333_spy_rsi_os',    sig_s402_s333_spy_rsi_os,    10, 0.20, 50),
    ('S403_s215_spy_rsi_turn',  sig_s403_s215_spy_rsi_turn,  10, 0.20, 50),
    ('S404_s333_spy_rsi_turn',  sig_s404_s333_spy_rsi_turn,  10, 0.20, 50),
    ('S405_s215_tsla_3down',    sig_s405_s215_tsla_3down,    10, 0.20, 50),
    ('S406_s333_tsla_3down',    sig_s406_s333_tsla_3down,    10, 0.20, 50),
    ('S407_s215_vix_pct70',     sig_s407_s215_vix_pct70,     10, 0.20, 50),
    ('S408_s333_vix_pct70',     sig_s408_s333_vix_pct70,     10, 0.20, 50),
    ('S409_s215_dow_monday',    sig_s409_s215_dow_monday,    10, 0.20, 50),
    ('S410_s333_dow_monday',    sig_s410_s333_dow_monday,    10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q)


# ── Phase 8R — VIX-pct aggressive extensions + cross with NR7/compressed/200MA ──
# S407 ($1.45M, 7/7yr) and S408 (93% WR, 6/6yr) are the best signals ever found.
# VIX percentile identifies sustained fear regimes — extend this dimension aggressively.
# Cross with NR7, compressed, above200MA, SPY weak, VIX rec, monthly low, OPEX.

def sig_s411_s408_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S411: S408 (MACD+VIX pct>70, 93% WR) + compressed ATR.
    Best signal (S408) + structural compression filter.
    Hypothesis: MACD turning + sustained fear + ATR compressed = maximum precision."""
    if sig_s408_s333_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s412_s408_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S412: S408 (MACD+VIX pct>70) + above 200d MA.
    Best signal in confirmed uptrend — bull regime + sustained fear = panic dip in bull market.
    Hypothesis: sustained fear within an uptrend = aggressive buy-the-dip."""
    if sig_s408_s333_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s413_s408_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S413: S408 (MACD+VIX pct>70) + SPY below 50d SMA.
    Best signal + broad market in downtrend.
    Hypothesis: MACD turning + fear regime + bear trend = maximum excess discount."""
    if sig_s408_s333_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_sma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close < spy_sma50 else 0


def sig_s414_s408_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S414: S408 (MACD+VIX pct>70) + NR7.
    Best signal + NR7 tightest day filter.
    Hypothesis: sustained fear + MACD turning on tightest day = maximum entry precision."""
    if sig_s408_s333_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s415_s408_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S415: S408 (MACD+VIX pct>70) + VIX recovery (VIX also starting to cool).
    VIX percentile AND VIX cooling simultaneously.
    Hypothesis: fear regime + first signs of macro cooling = perfect asymmetric entry."""
    if sig_s408_s333_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s416_s407_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S416: S407 (S215+VIX pct>70, $1.45M, 7/7yr) + compressed ATR.
    Best high-volume signal + compression.
    Hypothesis: sustained fear + weekly support + ATR compressed = coiled spring."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s417_s407_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S417: S407 (S215+VIX pct>70) + above 200d MA.
    High-volume signal in uptrend.
    Hypothesis: 25 trades at WR=80% split by trend — uptrend subset should be higher."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s418_s407_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S418: S407 (S215+VIX pct>70) + SPY below 50d SMA.
    High-volume signal + broad weakness context.
    Hypothesis: sustained fear in bear trend = most discounted weekly support entries."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_sma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close < spy_sma50 else 0


def sig_s419_s407_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S419: S407 (S215+VIX pct>70) + NR7.
    High-volume signal + NR7 tightest day.
    Hypothesis: fear regime at weekly support on tightest day = best timing for S407."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s420_s407_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S420: S407 (S215+VIX pct>70) + VIX recovery.
    VIX structurally elevated + VIX starting to cool = two VIX signals in sync.
    Hypothesis: when structural fear AND immediate cooling coincide = ideal entry."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


SIGNALS_P8R: List[Tuple] = [
    # Phase 8R — VIX percentile extensions (best indicator ever found)
    ('S411_s408_compressed',    sig_s411_s408_compressed,    10, 0.20, 50),
    ('S412_s408_above200ma',    sig_s412_s408_above200ma,    10, 0.20, 50),
    ('S413_s408_spy_weak',      sig_s413_s408_spy_weak,      10, 0.20, 50),
    ('S414_s408_nr7',           sig_s414_s408_nr7,           10, 0.20, 50),
    ('S415_s408_vix_rec',       sig_s415_s408_vix_rec,       10, 0.20, 50),
    ('S416_s407_compressed',    sig_s416_s407_compressed,    10, 0.20, 50),
    ('S417_s407_above200ma',    sig_s417_s407_above200ma,    10, 0.20, 50),
    ('S418_s407_spy_weak',      sig_s418_s407_spy_weak,      10, 0.20, 50),
    ('S419_s407_nr7',           sig_s419_s407_nr7,           10, 0.20, 50),
    ('S420_s407_vix_rec',       sig_s420_s407_vix_rec,       10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R)


# ── Phase 8S — VIX-pct 4-way combos + stricter threshold + monthly ────────────
# S413, S416, S417, S418 are the new top signals. Extend to 4-way combos.
# Also test VIX pct>80 (stricter threshold) and VIX pct 6-month window.

def sig_s421_s413_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S421: S413 (MACD+VIX pct+SPY weak, 6/6yr, WR=92%) + compressed ATR.
    4-way: MACD + sustained fear + broad weakness + ATR compression.
    Hypothesis: all macro/momentum/structure dimensions aligned = highest precision."""
    if sig_s413_s408_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s422_s413_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S422: S413 (MACD+VIX pct+SPY weak) + NR7.
    4-way: MACD + sustained fear + SPY weak + tightest day.
    Hypothesis: best macro signal filtered to optimal entry day."""
    if sig_s413_s408_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s423_s416_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S423: S416 (S407+compressed, PF=267, 6/6yr) + MACD turning.
    4-way: VIX pct + weekly support + compressed + MACD momentum.
    Hypothesis: adding MACD momentum turn to PF=267 signal = perfect entry."""
    if sig_s416_s407_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None:
        return 1   # not enough data → allow
    hist_prev = _macd_histogram(tsla, i - 1)
    if hist_prev is None:
        return 1
    return 1 if (hist > hist_prev and hist < 0) else 0


def sig_s424_s417_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S424: S417 (VIX pct+above200MA, WR=92%, PF=39.64) + compressed ATR.
    4-way: VIX pct + weekly support + uptrend + ATR compression.
    Hypothesis: fear regime + bull trend + compression = absolute best entry."""
    if sig_s417_s407_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s425_s417_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S425: S417 (VIX pct+above200MA) + MACD turning.
    4-way: sustained fear + weekly support + bull trend + MACD momentum.
    Hypothesis: MACD adds momentum confirmation to the best balanced signal."""
    if sig_s417_s407_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None:
        return 1
    hist_prev = _macd_histogram(tsla, i - 1)
    if hist_prev is None:
        return 1
    return 1 if (hist > hist_prev and hist < 0) else 0


def sig_s426_s418_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S426: S418 (VIX pct+SPY weak+S215, n=20, WR=80%) + compressed ATR.
    Best high-volume signal + compression filter.
    Hypothesis: S418 has n=20 trades; adding compression raises WR significantly."""
    if sig_s418_s407_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s427_s418_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S427: S418 (VIX pct+SPY weak+S215, n=20) + MACD turning.
    Add MACD momentum to the highest-n VIX pct signal.
    Hypothesis: filtering n=20 high-volume signal with MACD = higher precision subset."""
    if sig_s418_s407_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None:
        return 1
    hist_prev = _macd_histogram(tsla, i - 1)
    if hist_prev is None:
        return 1
    return 1 if (hist > hist_prev and hist < 0) else 0


def sig_s428_s215_vix_pct80(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S428: S215 (weekly support) + VIX > 80th percentile (stricter threshold).
    Hypothesis: top 20% fear days produce even more extreme bounces than top 30%."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.80) else 0


def sig_s429_s333_vix_pct80(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S429: S333 (MACD turning) + VIX > 80th percentile.
    Stricter fear threshold for MACD signal.
    Hypothesis: top 20% fear + MACD momentum = tightest macro+momentum combo."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.80) else 0


def sig_s430_s407_monthly_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S430: S407 (S215+VIX pct>70, $1.45M, 7/7yr) + monthly near lower channel.
    Multi-TF support at weekly+monthly + sustained fear regime.
    Hypothesis: 2-TF structural support in fear regime = maximum recovery signal."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _monthly_channel_near_lower(tsla, i, n_months=12, frac=0.30) else 0


SIGNALS_P8S: List[Tuple] = [
    # Phase 8S — VIX pct 4-way combos + stricter threshold + monthly
    ('S421_s413_compressed',    sig_s421_s413_compressed,    10, 0.20, 50),
    ('S422_s413_nr7',           sig_s422_s413_nr7,           10, 0.20, 50),
    ('S423_s416_macd',          sig_s423_s416_macd,          10, 0.20, 50),
    ('S424_s417_compressed',    sig_s424_s417_compressed,    10, 0.20, 50),
    ('S425_s417_macd',          sig_s425_s417_macd,          10, 0.20, 50),
    ('S426_s418_compressed',    sig_s426_s418_compressed,    10, 0.20, 50),
    ('S427_s418_macd',          sig_s427_s418_macd,          10, 0.20, 50),
    ('S428_s215_vix_pct80',     sig_s428_s215_vix_pct80,     10, 0.20, 50),
    ('S429_s333_vix_pct80',     sig_s429_s333_vix_pct80,     10, 0.20, 50),
    ('S430_s407_monthly_low',   sig_s430_s407_monthly_low,   10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S)


# ── Phase 8T — VIX-pct 3-way extensions + VIX trend approach + pct60 ──────────
# Continue extending top signals. Also test: VIX above its 20d MA (trend approach)
# and VIX pct>60 (looser) to see frequency vs quality trade-off.

def _vix_above_ma(vix, i, period: int = 20, mult: float = 1.10) -> bool:
    """VIX is above its own 20d MA by mult factor — VIX in rising trend (fear regime)."""
    if i < period:
        return True
    vix_now = float(vix['close'].iloc[i])
    vix_ma  = float(vix['close'].iloc[i - period:i].mean())
    return vix_now > vix_ma * mult


def sig_s431_s429_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S431: S429 (MACD+VIX pct>80, 5/5yr, WR=92%) + compressed ATR.
    3-way strict: MACD + top 20% fear + ATR compression.
    Hypothesis: strictest fear threshold + MACD + compression = maximum precision."""
    if sig_s429_s333_vix_pct80(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s432_s429_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S432: S429 (MACD+VIX pct>80) + above 200d MA.
    3-way: MACD + top 20% fear + bull regime.
    Hypothesis: fear regime within bull trend = panic dip at its worst (best entry)."""
    if sig_s429_s333_vix_pct80(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s433_s429_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S433: S429 (MACD+VIX pct>80) + NR7.
    3-way: MACD + top fear threshold + tightest day.
    Hypothesis: momentum turning on structurally fearful + tightest day = perfect timing."""
    if sig_s429_s333_vix_pct80(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s434_s428_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S434: S428 (S215+VIX pct>80) + compressed ATR.
    3-way: weekly support + top 20% fear + ATR compression.
    Hypothesis: structural support + extreme fear + compression = spring-load in fear."""
    if sig_s428_s215_vix_pct80(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s435_s428_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S435: S428 (S215+VIX pct>80) + NR7.
    3-way: weekly support + top 20% fear + tightest day.
    Hypothesis: strictest fear + tightest day at support = most precise fear entry."""
    if sig_s428_s215_vix_pct80(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s436_s427_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S436: S427 (MACD+VIX pct+SPY weak, 100% WR 5/5yr) + above 200d MA.
    4-way: MACD + fear + SPY weak + bull regime simultaneously.
    Hypothesis: SPY below 50MA but TSLA above 200MA = TSLA stronger than index."""
    if sig_s427_s418_macd(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s437_s427_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S437: S427 (MACD+VIX pct+SPY weak, 100% WR 5/5yr) + compressed ATR.
    4-way: MACD + fear + SPY weak + ATR compression.
    Hypothesis: best 100% WR signal + compression = 4-layer maximum filter."""
    if sig_s427_s418_macd(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s438_s215_vix_above_ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S438: S215 (weekly support) + VIX above its 20d MA by 10%.
    NEW APPROACH: VIX trend (above own MA) vs VIX percentile.
    Hypothesis: VIX trending up (above 20d MA) = fear in acceleration = better bounces."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_above_ma(vix, i, period=20, mult=1.10) else 0


def sig_s439_s333_vix_above_ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S439: S333 (MACD turning at weekly support) + VIX above its 20d MA by 10%.
    VIX trend approach: MACD + VIX trending upward.
    Hypothesis: MACD turn + VIX in rising trend = momentum reversal in fear phase."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_above_ma(vix, i, period=20, mult=1.10) else 0


def sig_s440_s215_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S440: S215 (weekly support) + VIX > 60th percentile (broader fear threshold).
    Looser threshold: how many trades does 60th pct give vs 70th (S407) vs 80th (S428)?
    Hypothesis: lower threshold = more trades, check if quality degrades."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


SIGNALS_P8T: List[Tuple] = [
    # Phase 8T — VIX pct 3-way + VIX trend approach + pct60 comparison
    ('S431_s429_compressed',    sig_s431_s429_compressed,    10, 0.20, 50),
    ('S432_s429_above200ma',    sig_s432_s429_above200ma,    10, 0.20, 50),
    ('S433_s429_nr7',           sig_s433_s429_nr7,           10, 0.20, 50),
    ('S434_s428_compressed',    sig_s434_s428_compressed,    10, 0.20, 50),
    ('S435_s428_nr7',           sig_s435_s428_nr7,           10, 0.20, 50),
    ('S436_s427_above200ma',    sig_s436_s427_above200ma,    10, 0.20, 50),
    ('S437_s427_compressed',    sig_s437_s427_compressed,    10, 0.20, 50),
    ('S438_s215_vix_above_ma',  sig_s438_s215_vix_above_ma,  10, 0.20, 50),
    ('S439_s333_vix_above_ma',  sig_s439_s333_vix_above_ma,  10, 0.20, 50),
    ('S440_s215_vix_pct60',     sig_s440_s215_vix_pct60,     10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T)


# ── Phase 8U — VIX-pct60 extensions + 52-week proximity + SPY strength ────────
# S440 (pct60, $1.47M, 7/8yr) explores looser threshold. Now extend it.
# New: TSLA near 52-week low (annual oversold). SPY near 52-week high (SPY strong, TSLA lagging).
# Also: cross best signals with VIX pct60 to see if more frequency at same quality.

def _near_52wk_low(tsla, i, frac: float = 0.15) -> bool:
    """TSLA close is within frac of its trailing 52-week (252d) low."""
    lookback = min(i, 252)
    if lookback < 20:
        return False
    low_52wk = float(tsla['low'].iloc[i - lookback:i].min())
    close = float(tsla['close'].iloc[i])
    return close <= low_52wk * (1 + frac)


def _spy_near_52wk_high(spy, i, frac: float = 0.05) -> bool:
    """SPY close is within frac of its trailing 52-week (252d) high."""
    lookback = min(i, 252)
    if lookback < 20:
        return False
    high_52wk = float(spy['high'].iloc[i - lookback:i].max())
    close = float(spy['close'].iloc[i])
    return close >= high_52wk * (1 - frac)


def sig_s441_s440_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S441: S440 (S215+VIX pct>60, $1.47M, 7/8yr) + MACD turning.
    Adding MACD momentum filter to the broadest VIX pct signal.
    Hypothesis: pct60 gives n=27 trades → MACD should raise WR toward 90%+."""
    if sig_s440_s215_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None:
        return 1
    hist_prev = _macd_histogram(tsla, i - 1)
    if hist_prev is None:
        return 1
    return 1 if (hist > hist_prev and hist < 0) else 0


def sig_s442_s440_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S442: S440 (VIX pct60) + compressed ATR.
    Broad fear threshold + ATR compression.
    Hypothesis: 60th pct gives more trades than 70th → compressed subset should be strong."""
    if sig_s440_s215_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s443_s440_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S443: S440 (VIX pct60) + above 200d MA.
    Broadest fear signal filtered to uptrend.
    Hypothesis: 60th pct + bull regime = high volume signals in healthy market."""
    if sig_s440_s215_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s444_s440_spy_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S444: S440 (VIX pct60) + SPY below 50d SMA.
    Broad fear threshold + macro bear trend.
    Hypothesis: weekly support + mild fear + SPY weakness = high-frequency bear signal."""
    if sig_s440_s215_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 1
    spy_close = float(spy['close'].iloc[i])
    spy_sma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close < spy_sma50 else 0


def sig_s445_s440_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S445: S440 (VIX pct60) + NR7.
    Broadest fear threshold + NR7 tightest day.
    Hypothesis: pct60 fires often → NR7 keeps only best-timed entries."""
    if sig_s440_s215_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s446_s215_near_52wk_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S446: S215 (weekly support) + TSLA within 15% of 52-week low.
    NEW DIMENSION: Annual oversold level at weekly support.
    Hypothesis: 52-week lows = forced selling exhaustion + annual support confluence."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _near_52wk_low(tsla, i, frac=0.15) else 0


def sig_s447_s333_near_52wk_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S447: S333 (MACD turning at weekly support) + TSLA near 52-week low.
    MACD momentum reversal at annual low + weekly support convergence.
    Hypothesis: annual exhaustion + MACD turn = strongest mean-reversion entry."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _near_52wk_low(tsla, i, frac=0.15) else 0


def sig_s448_s407_near_52wk_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S448: S407 (VIX pct>70+S215, best signal) + TSLA near 52-week low.
    Best signal + annual oversold level.
    Hypothesis: sustained fear + annual oversold at weekly support = maximum discount."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _near_52wk_low(tsla, i, frac=0.15) else 0


def sig_s449_s215_spy_at_52wk_high(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S449: S215 (weekly support) + SPY within 5% of 52-week high.
    SPY at strength while TSLA at weekly support — classic lag setup.
    Hypothesis: SPY near all-time highs + TSLA at support = highest lag-catch potential."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_near_52wk_high(spy, i, frac=0.05) else 0


def sig_s450_s333_spy_at_52wk_high(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S450: S333 (MACD turning) + SPY within 5% of 52-week high.
    MACD reversal at weekly support when SPY is at year highs.
    Hypothesis: SPY at strength + TSLA MACD turning = lag catch at best macro timing."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_near_52wk_high(spy, i, frac=0.05) else 0


SIGNALS_P8U: List[Tuple] = [
    # Phase 8U — VIX pct60 extensions + 52-week proximity + SPY strength
    ('S441_s440_macd',              sig_s441_s440_macd,              10, 0.20, 50),
    ('S442_s440_compressed',        sig_s442_s440_compressed,        10, 0.20, 50),
    ('S443_s440_above200ma',        sig_s443_s440_above200ma,        10, 0.20, 50),
    ('S444_s440_spy_weak',          sig_s444_s440_spy_weak,          10, 0.20, 50),
    ('S445_s440_nr7',               sig_s445_s440_nr7,               10, 0.20, 50),
    ('S446_s215_near_52wk_low',     sig_s446_s215_near_52wk_low,     10, 0.20, 50),
    ('S447_s333_near_52wk_low',     sig_s447_s333_near_52wk_low,     10, 0.20, 50),
    ('S448_s407_near_52wk_low',     sig_s448_s407_near_52wk_low,     10, 0.20, 50),
    ('S449_s215_spy_at_52wk_high',  sig_s449_s215_spy_at_52wk_high,  10, 0.20, 50),
    ('S450_s333_spy_at_52wk_high',  sig_s450_s333_spy_at_52wk_high,  10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U)


# ── Phase 8V — SPY-strength extensions + 4-way VIX pct combos + S500 prep ─────
# S449 (SPY near 52wk high+S215, 86% WR, 5/5yr) is a major new pattern.
# S441 (VIX pct60+MACD, 100% WR 6/6yr) and S442 (VIX pct60+compressed, PF=280).
# Extend aggressively: add VIX pct to SPY strength, cross best signals together.

def sig_s451_s449_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S451: S449 (SPY near 52wk high+S215, 5/5yr) + VIX pct>70.
    3-way: SPY at annual strength + TSLA at weekly support + sustained fear.
    Hypothesis: the fear/lag combo maximized — SPY strong but VIX elevated = TSLA panic."""
    if sig_s449_s215_spy_at_52wk_high(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s452_s449_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S452: S449 (SPY near 52wk high+S215) + compressed ATR.
    SPY strength + TSLA weekly support + ATR compression.
    Hypothesis: SPY at highs + TSLA compressed at support = maximum reversion energy."""
    if sig_s449_s215_spy_at_52wk_high(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s453_s449_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S453: S449 (SPY near 52wk high+S215) + MACD turning.
    SPY at highs + TSLA at weekly support + MACD momentum reversal.
    Hypothesis: lag pattern + MACD confirms individual stock turning = strongest lag entry."""
    if sig_s449_s215_spy_at_52wk_high(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None:
        return 1
    hist_prev = _macd_histogram(tsla, i - 1)
    if hist_prev is None:
        return 1
    return 1 if (hist > hist_prev and hist < 0) else 0


def sig_s454_s449_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S454: S449 (SPY near 52wk high) + TSLA above 200d MA.
    SPY at highs + TSLA in bull regime at weekly support.
    Hypothesis: bull-trend pullback to support when SPY is ripping = snap-back."""
    if sig_s449_s215_spy_at_52wk_high(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s455_s441_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S455: S441 (VIX pct60+MACD, 100% WR 6/6yr) + compressed ATR.
    4-way: VIX pct + weekly support + MACD + compressed.
    Hypothesis: filtering the 100% WR signal with compression = ultra-precision."""
    if sig_s441_s440_macd(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s456_s441_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S456: S441 (VIX pct60+MACD, 100% WR) + above 200d MA.
    100% WR signal in confirmed uptrend.
    Hypothesis: VIX fear + MACD momentum in bull trend = best combo."""
    if sig_s441_s440_macd(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s457_s442_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S457: S442 (VIX pct60+compressed, PF=280, 7/7yr) + MACD turning.
    4-way: VIX pct + weekly support + compressed + MACD momentum.
    Hypothesis: adding MACD to PF=280 signal → 100% WR subset."""
    if sig_s442_s440_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None:
        return 1
    hist_prev = _macd_histogram(tsla, i - 1)
    if hist_prev is None:
        return 1
    return 1 if (hist > hist_prev and hist < 0) else 0


def sig_s458_s442_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S458: S442 (VIX pct60+compressed, PF=280) + above 200d MA.
    Best PF signal filtered to uptrend.
    Hypothesis: extreme PF signal in bull trend = most consistent entries."""
    if sig_s442_s440_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s459_s407_spy_high(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S459: S407 (VIX pct>70+S215, $1.45M, 7/7yr) + SPY near 52wk high.
    PARADOX signal: sustained fear (VIX high) + SPY at annual strength simultaneously.
    Hypothesis: when VIX is elevated but SPY is strong, TSLA is the panic selloff target."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_near_52wk_high(spy, i, frac=0.05) else 0


def sig_s460_s408_spy_high(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S460: S408 (MACD+VIX pct>70, 93% WR, 6/6yr) + SPY near 52wk high.
    Best signal (S408) + SPY at strength.
    Hypothesis: MACD turning + sustained fear + SPY ripping = TSLA snap-back."""
    if sig_s408_s333_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_near_52wk_high(spy, i, frac=0.05) else 0


SIGNALS_P8V: List[Tuple] = [
    # Phase 8V — SPY strength extensions + 4-way VIX pct combos
    ('S451_s449_vix_pct70',     sig_s451_s449_vix_pct70,     10, 0.20, 50),
    ('S452_s449_compressed',    sig_s452_s449_compressed,    10, 0.20, 50),
    ('S453_s449_macd',          sig_s453_s449_macd,          10, 0.20, 50),
    ('S454_s449_above200ma',    sig_s454_s449_above200ma,    10, 0.20, 50),
    ('S455_s441_compressed',    sig_s455_s441_compressed,    10, 0.20, 50),
    ('S456_s441_above200ma',    sig_s456_s441_above200ma,    10, 0.20, 50),
    ('S457_s442_macd',          sig_s457_s442_macd,          10, 0.20, 50),
    ('S458_s442_above200ma',    sig_s458_s442_above200ma,    10, 0.20, 50),
    ('S459_s407_spy_high',      sig_s459_s407_spy_high,      10, 0.20, 50),
    ('S460_s408_spy_high',      sig_s460_s408_spy_high,      10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V)


# ── Phase 8W — SPY-200MA cross + S454 extensions + Bollinger %B ───────────────
# S454 (SPY near 52wk high+above200MA, 91% WR, n=11) → extend aggressively.
# New: SPY above its own 200d MA (SPY uptrend filter, different from SPY level).
# New: Bollinger Band %B (how far price is below lower BB) — oversold on BB basis.

def _spy_above_200ma(spy, i) -> bool:
    """SPY is in its own uptrend (above 200d MA) — broad market bull regime."""
    if i < 200:
        return True   # not enough history → assume bull
    spy_close = float(spy['close'].iloc[i])
    spy_ma200 = float(spy['close'].iloc[i - 200:i].mean())
    return spy_close > spy_ma200


def _bb_pct_b(tsla, i, period: int = 20, std_mult: float = 2.0) -> Optional[float]:
    """Bollinger Band %B: (close - lower_band) / (upper_band - lower_band).
    %B < 0 = below lower band (oversold). %B near 0 = near lower band."""
    if i < period:
        return None
    closes = tsla['close'].iloc[i - period:i + 1].values.astype(float)
    mean = closes.mean()
    std  = float(np.std(closes, ddof=1))
    if std < 1e-6:
        return None
    upper = mean + std_mult * std
    lower = mean - std_mult * std
    band_width = upper - lower
    if band_width < 1e-6:
        return None
    return (closes[-1] - lower) / band_width   # %B: 0 = lower band, 1 = upper band


def sig_s461_s454_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S461: S454 (SPY high+above200MA+S215, 91% WR) + VIX pct>70.
    4-way: SPY strength + TSLA uptrend + weekly support + sustained fear.
    Hypothesis: the contradiction (SPY strong + VIX elevated) = TSLA panic dip target."""
    if sig_s454_s449_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s462_s454_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S462: S454 (SPY high+above200MA, 91% WR) + compressed ATR.
    4-way: SPY strength + TSLA uptrend + weekly support + compression.
    Hypothesis: lag at annual high + pullback compressed = maximum spring energy."""
    if sig_s454_s449_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s463_s454_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S463: S454 (SPY high+above200MA) + MACD turning.
    4-way: SPY strength + TSLA uptrend + weekly support + MACD momentum.
    Hypothesis: adding MACD to S454 (91% WR) → should push to 100% WR."""
    if sig_s454_s449_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None:
        return 1
    hist_prev = _macd_histogram(tsla, i - 1)
    if hist_prev is None:
        return 1
    return 1 if (hist > hist_prev and hist < 0) else 0


def sig_s464_s215_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S464: S215 (weekly support) + SPY above its own 200d MA.
    NEW: SPY in uptrend context (not just near high — sustained bull market).
    Hypothesis: TSLA weekly support when SPY is in confirmed uptrend = bull dip."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_above_200ma(spy, i) else 0


def sig_s465_s333_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S465: S333 (MACD turning at weekly support) + SPY above 200d MA.
    MACD momentum reversal in SPY uptrend context.
    Hypothesis: MACD turn at support when SPY is in confirmed bull = cleanest entry."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_above_200ma(spy, i) else 0


def sig_s466_s407_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S466: S407 (VIX pct>70+S215, $1.45M) + SPY above 200d MA.
    VIX fear regime + SPY uptrend simultaneously.
    Hypothesis: sustained fear within a bull market = TSLA panic selloff in good conditions."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_above_200ma(spy, i) else 0


def sig_s467_s408_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S467: S408 (MACD+VIX pct>70, 93% WR) + SPY above 200d MA.
    Best signal filtered to SPY uptrend.
    Hypothesis: 93% WR signal during bull market = highest precision combo."""
    if sig_s408_s333_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_above_200ma(spy, i) else 0


def sig_s468_s215_bb_pct_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S468: S215 (weekly support) + Bollinger %B < 0.15 (near/below lower BB).
    NEW: Bollinger oversold — price near or below 20d lower Bollinger Band.
    Hypothesis: weekly channel support + BB oversold = double technical support."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    pct_b = _bb_pct_b(tsla, i)
    if pct_b is None:
        return 1
    return 1 if pct_b < 0.15 else 0


def sig_s469_s333_bb_pct_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S469: S333 (MACD turning) + Bollinger %B < 0.15.
    MACD momentum reversal when near Bollinger lower band.
    Hypothesis: MACD turning at BB oversold = double momentum + structure confirm."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    pct_b = _bb_pct_b(tsla, i)
    if pct_b is None:
        return 1
    return 1 if pct_b < 0.15 else 0


def sig_s470_s407_bb_pct_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S470: S407 (VIX pct>70+S215, $1.45M) + Bollinger %B < 0.15.
    Best signal + Bollinger oversold confirmation.
    Hypothesis: VIX fear + weekly support + BB oversold = 3 layers of support."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    pct_b = _bb_pct_b(tsla, i)
    if pct_b is None:
        return 1
    return 1 if pct_b < 0.15 else 0


SIGNALS_P8W: List[Tuple] = [
    # Phase 8W — S454 extensions + SPY-200MA cross + Bollinger %B
    ('S461_s454_vix_pct70',       sig_s461_s454_vix_pct70,       10, 0.20, 50),
    ('S462_s454_compressed',      sig_s462_s454_compressed,      10, 0.20, 50),
    ('S463_s454_macd',            sig_s463_s454_macd,            10, 0.20, 50),
    ('S464_s215_spy_above200ma',  sig_s464_s215_spy_above200ma,  10, 0.20, 50),
    ('S465_s333_spy_above200ma',  sig_s465_s333_spy_above200ma,  10, 0.20, 50),
    ('S466_s407_spy_above200ma',  sig_s466_s407_spy_above200ma,  10, 0.20, 50),
    ('S467_s408_spy_above200ma',  sig_s467_s408_spy_above200ma,  10, 0.20, 50),
    ('S468_s215_bb_pct_low',      sig_s468_s215_bb_pct_low,      10, 0.20, 50),
    ('S469_s333_bb_pct_low',      sig_s469_s333_bb_pct_low,      10, 0.20, 50),
    ('S470_s407_bb_pct_low',      sig_s470_s407_bb_pct_low,      10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W)


# ── Phase 8X — S464/S467 extensions + new: SPY momentum + stochastic ──────────
# S464 ($1.04M, 83% WR, n=18) and S467 (100% WR, 4/4yr) → extend aggressively.
# New: SPY made new 10d high today (SPY momentum), stochastic %K oversold.

def _spy_new_nd_high(spy, i, n: int = 10) -> bool:
    """SPY closed at/near new n-day high today — upside momentum."""
    if i < n:
        return False
    spy_high = float(spy['high'].iloc[i - n:i].max())
    spy_close = float(spy['close'].iloc[i])
    return spy_close >= spy_high * 0.99   # within 1% of n-day high


def sig_s471_s464_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S471: S464 (SPY>200MA+S215, $1.04M, n=18) + compressed ATR.
    SPY bull uptrend + TSLA weekly support + ATR compression.
    Hypothesis: S464 with n=18 trades → compression filter should raise WR significantly."""
    if sig_s464_s215_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s472_s464_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S472: S464 (SPY>200MA+S215, n=18) + NR7.
    SPY bull uptrend + TSLA weekly support + tightest day.
    Hypothesis: S464 filtered by NR7 = highest precision within the $1M signal."""
    if sig_s464_s215_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s473_s464_vix_rec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S473: S464 (SPY>200MA+S215) + VIX recovery.
    SPY bull uptrend + TSLA at weekly support + VIX starting to cool.
    Hypothesis: bull market + VIX fear cooling = best timing in uptrend."""
    if sig_s464_s215_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 1
    vix_now = float(vix['close'].iloc[i])
    vix_5d  = float(vix['close'].iloc[i - 5])
    vix_10d = float(vix['close'].iloc[i - 10])
    return 1 if ((vix_5d > 20 or vix_10d > 20) and vix_now < vix_5d * 0.90) else 0


def sig_s474_s464_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S474: S464 (SPY>200MA+S215) + TSLA above 200d MA.
    Both SPY and TSLA in uptrend at TSLA weekly support.
    Hypothesis: both stocks in bull regime + TSLA at support = cleanest bull dip."""
    if sig_s464_s215_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close > ma200 else 0


def sig_s475_s467_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S475: S467 (MACD+VIX pct+SPY>200MA, 100% WR, 4/4yr) + compressed ATR.
    4-way + compression → 5-layer signal.
    Hypothesis: adding compression to 100% WR signal = tightest possible entry."""
    if sig_s467_s408_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s476_s464_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S476: S464 (SPY>200MA+S215, n=18) + VIX pct>70.
    SPY bull uptrend + TSLA support + sustained fear = S466 (already done!).
    This IS S466 — skip for dedup. Using for VIX pct60 instead."""
    # S466 = S407 + SPY>200MA = S215 + VIX pct70 + SPY>200MA
    # S464 + VIX pct70 = S215 + SPY>200MA + VIX pct70 = same signal
    # Use VIX pct60 instead to test looser threshold with SPY uptrend
    if sig_s464_s215_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s477_s215_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S477: S215 (weekly support) + SPY near 10d high (momentum confirmation).
    NEW: SPY showing intraday/recent momentum while TSLA lags at weekly support.
    Hypothesis: SPY at 10d high + TSLA at support = immediate lag catch signal."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_new_nd_high(spy, i, n=10) else 0


def sig_s478_s333_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S478: S333 (MACD turning at weekly support) + SPY near 10d high.
    MACD turning + SPY momentum = both indicators confirming separately.
    Hypothesis: TSLA momentum reversing + SPY building = strongest simultaneous signal."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_new_nd_high(spy, i, n=10) else 0


def sig_s479_s407_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S479: S407 (VIX pct70+S215, $1.45M) + SPY near 10d high.
    Best signal + SPY momentum confirmation.
    Hypothesis: sustained fear + SPY momentum = TSLA temporarily discounted."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_new_nd_high(spy, i, n=10) else 0


def sig_s480_s215_stoch_os(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S480: S215 (weekly support) + Stochastic %K < 20 (oversold).
    NEW: Stochastic oversold at weekly support.
    Hypothesis: weekly support + stochastic oversold = double technical bottom signal."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    stoch = _stoch_k(tsla, i, period=14)
    if stoch is None:
        return 1
    return 1 if stoch < 20 else 0


SIGNALS_P8X: List[Tuple] = [
    # Phase 8X — S464/S467 extensions + SPY momentum + stochastic
    ('S471_s464_compressed',     sig_s471_s464_compressed,     10, 0.20, 50),
    ('S472_s464_nr7',            sig_s472_s464_nr7,            10, 0.20, 50),
    ('S473_s464_vix_rec',        sig_s473_s464_vix_rec,        10, 0.20, 50),
    ('S474_s464_above200ma',     sig_s474_s464_above200ma,     10, 0.20, 50),
    ('S475_s467_compressed',     sig_s475_s467_compressed,     10, 0.20, 50),
    ('S476_s464_vix_pct60',      sig_s476_s464_vix_pct70,      10, 0.20, 50),
    ('S477_s215_spy_momentum',   sig_s477_s215_spy_momentum,   10, 0.20, 50),
    ('S478_s333_spy_momentum',   sig_s478_s333_spy_momentum,   10, 0.20, 50),
    ('S479_s407_spy_momentum',   sig_s479_s407_spy_momentum,   10, 0.20, 50),
    ('S480_s215_stoch_os',       sig_s480_s215_stoch_os,       10, 0.20, 50),
]

# ── Phase 8Y — SPY momentum extensions + double-uptrend combos ────────────────
# S477 (SPY new 10d high + S215) confirmed at 90% WR, 5/5yr.
# Extend into: compressed, NR7, VIX pct, above200MA (3-way combos).
# Also extend S474 (both SPY+TSLA>200MA) and best VIX pct signals with SPY momentum.

def sig_s481_s477_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S481: S477 (SPY momentum + weekly support) + ATR compression.
    Triple: SPY near 10d high + TSLA at weekly support + compressed volatility."""
    if sig_s477_s215_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s482_s477_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S482: S477 (SPY momentum + weekly support) + NR7 (narrowest range in 7d).
    Triple: SPY at new high + TSLA pinching at weekly support."""
    if sig_s477_s215_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s483_s477_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S483: S477 (SPY momentum) + VIX > 60th pct of trailing 252d.
    Triple: SPY momentum + weekly support + structurally elevated fear."""
    if sig_s477_s215_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s484_s477_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S484: S477 (SPY momentum) + TSLA above own 200d MA.
    Triple: SPY at high + TSLA at weekly support + TSLA in bull uptrend."""
    if sig_s477_s215_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 0
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if float(tsla['close'].iloc[i]) > ma200 else 0


def sig_s485_s474_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S485: S474 (SPY>200MA + TSLA>200MA + weekly support) + ATR compression.
    Both in bull uptrend + volatility pinch = explosive spring-loaded setup."""
    if sig_s474_s464_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s486_s474_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S486: S474 (double uptrend + weekly support) + NR7.
    Both in bull trend + narrowest range = extreme compression before bounce."""
    if sig_s474_s464_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s487_s474_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S487: S474 (double uptrend + weekly support) + VIX > 70th pct.
    Triple: both in uptrend + weekly support + sustained fear."""
    if sig_s474_s464_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s488_s440_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S488: S440 (VIX pct60, $1.47M best signal) + SPY near 10d high.
    Two completely independent confirmations: fear regime + SPY building momentum."""
    if sig_s440_s215_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_new_nd_high(spy, i, n=10) else 0


def sig_s489_s408_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S489: S408 (MACD+VIX pct70, WR=93%) + SPY near 10d high.
    4-way: MACD turning + VIX fear regime + weekly support + SPY momentum."""
    if sig_s408_s333_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_new_nd_high(spy, i, n=10) else 0


def sig_s490_s478_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S490: S478 (S333+SPY momentum, WR=86%) + ATR compression.
    MACD turning + SPY momentum + volatility pinch = triple confirmation."""
    if sig_s478_s333_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


SIGNALS_P8Y: List[Tuple] = [
    # Phase 8Y — SPY momentum extensions + double-uptrend combos (S481-S490)
    ('S481_s477_compressed',     sig_s481_s477_compressed,     10, 0.20, 50),
    ('S482_s477_nr7',            sig_s482_s477_nr7,            10, 0.20, 50),
    ('S483_s477_vix_pct60',      sig_s483_s477_vix_pct60,      10, 0.20, 50),
    ('S484_s477_above200ma',     sig_s484_s477_above200ma,     10, 0.20, 50),
    ('S485_s474_compressed',     sig_s485_s474_compressed,     10, 0.20, 50),
    ('S486_s474_nr7',            sig_s486_s474_nr7,            10, 0.20, 50),
    ('S487_s474_vix_pct70',      sig_s487_s474_vix_pct70,      10, 0.20, 50),
    ('S488_s440_spy_momentum',   sig_s488_s440_spy_momentum,   10, 0.20, 50),
    ('S489_s408_spy_momentum',   sig_s489_s408_spy_momentum,   10, 0.20, 50),
    ('S490_s478_compressed',     sig_s490_s478_compressed,     10, 0.20, 50),
]

# ── Phase 8Z — Capitulation signals + S500 milestone ─────────────────────────
# New dimensions: consecutive TSLA down days (exhaustion), new n-day low (flush),
# volume dry-up (quiet capitulation), and S500 milestone combining VIX pct + NR7 + weekly support.

def _new_nd_low(tsla, i, n: int = 10) -> bool:
    """Today's close is the lowest of the last n days — price flush."""
    if i < n:
        return False
    today_close = float(tsla['close'].iloc[i])
    hist_min = float(tsla['close'].iloc[i - n:i].min())
    return today_close <= hist_min * 1.005  # within 0.5% of n-day low


def _volume_below_avg(tsla, i, mult: float = 0.70, period: int = 20) -> bool:
    """Today's volume < mult × 20d average (quiet day = no panic = capitulation drying up)."""
    if i < period:
        return True
    if 'volume' not in tsla.columns:
        return True
    avg_vol = float(tsla['volume'].iloc[i - period:i].mean())
    if avg_vol <= 0:
        return True
    return float(tsla['volume'].iloc[i]) < mult * avg_vol


def sig_s491_s215_consec_down3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S491: S215 (weekly support) + TSLA 3+ consecutive down closes.
    NEW: Selling exhaustion after sequential multi-day decline at weekly support.
    Hypothesis: capitulation into support = buyers stepping in en masse."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=3) else 0


def sig_s492_s333_consec_down3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S492: S333 (MACD turning) + TSLA 3+ consecutive down closes.
    MACD momentum inflection + sequential selling = reversal confirmation.
    Hypothesis: MACD curvature shifts as sellers exhaust into weekly support."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=3) else 0


def sig_s493_s407_consec_down3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S493: S407 (VIX pct>70 + weekly support, $1.45M) + TSLA 3 consec down.
    Triple: sustained fear regime + weekly support + sequential selling exhaustion."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=3) else 0


def sig_s494_s215_new_10d_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S494: S215 (weekly support) + today is new 10-day closing low.
    Price flush: TSLA hitting new short-term low at weekly support = capitulation."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _new_nd_low(tsla, i, n=10) else 0


def sig_s495_s215_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S495: S215 (weekly support) + below-average volume (<70% of 20d avg).
    Volume dry-up at support: no panic selling, institutional quietly accumulating."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s496_s494_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S496: S494 (new 10d low at weekly support) + ATR compression.
    Price flush + volatility coiling = paradox spring (new low but very quiet)."""
    if sig_s494_s215_new_10d_low(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s497_s481_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S497: S481 (SPY momentum + weekly support + compressed) + VIX pct>70.
    4-way: SPY momentum + TSLA weekly support + ATR compressed + VIX fear.
    Ultimate confluence: every dimension fired simultaneously."""
    if sig_s481_s477_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s498_s464_consec_down3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S498: S464 (SPY>200MA + weekly support, $1.04M) + TSLA 3 consec down.
    Bull regime + weekly support + sequential capitulation = strongest setup in uptrend."""
    if sig_s464_s215_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=3) else 0


def sig_s499_s477_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S499: S477 (SPY momentum + weekly support) + VIX pct>60.
    Triple: SPY at new high + TSLA at weekly support + VIX elevated.
    Cross-dimension: market strength + individual weakness + macro fear."""
    if sig_s477_s215_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s500_milestone_vix_pct_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S500 MILESTONE: S215 (weekly support) + VIX pct>70 + NR7.
    Three strongest discovered dimensions combined:
    - Weekly support (multi-TF confluence, base of all best signals)
    - VIX sustained fear percentile (highest P&L add-on found)
    - NR7 narrowest range in 7 days (highest WR compression signal)
    Hypothesis: VIX fear + price pinch at support = ultimate coiled spring."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _vix_elevated_pct(vix, i, window=252, pct=0.70):
        return 0
    return 1 if _nr7(tsla, i) else 0


SIGNALS_P8Z: List[Tuple] = [
    # Phase 8Z — Capitulation signals + S500 milestone (S491-S500)
    ('S491_s215_consec_down3',   sig_s491_s215_consec_down3,   10, 0.20, 50),
    ('S492_s333_consec_down3',   sig_s492_s333_consec_down3,   10, 0.20, 50),
    ('S493_s407_consec_down3',   sig_s493_s407_consec_down3,   10, 0.20, 50),
    ('S494_s215_new_10d_low',    sig_s494_s215_new_10d_low,    10, 0.20, 50),
    ('S495_s215_vol_dryup',      sig_s495_s215_vol_dryup,      10, 0.20, 50),
    ('S496_s494_compressed',     sig_s496_s494_compressed,     10, 0.20, 50),
    ('S497_s481_vix_pct70',      sig_s497_s481_vix_pct70,      10, 0.20, 50),
    ('S498_s464_consec_down3',   sig_s498_s464_consec_down3,   10, 0.20, 50),
    ('S499_s477_vix_pct60',      sig_s499_s477_vix_pct60,      10, 0.20, 50),
    ('S500_milestone_vix_nr7',   sig_s500_milestone_vix_pct_nr7, 10, 0.20, 50),
]

# ── Phase 9A — Volume dry-up extensions + inside bar pattern ──────────────────
# S495 (vol dry-up at weekly support) = 100% WR, n=9, 6/6yr, $842K — BREAKTHROUGH.
# Extend into: compressed, NR7, VIX regimes, SPY momentum, MACD.
# Add inside bar (today's range inside yesterday's) as new dimension.

def _inside_bar(tsla, i) -> bool:
    """Today's high-low range is entirely within yesterday's high-low (inside bar).
    Extreme compression: market participants unwilling to extend range either direction."""
    if i < 1:
        return False
    today_high = float(tsla['high'].iloc[i])
    today_low  = float(tsla['low'].iloc[i])
    prev_high  = float(tsla['high'].iloc[i - 1])
    prev_low   = float(tsla['low'].iloc[i - 1])
    return today_high <= prev_high and today_low >= prev_low


def sig_s501_s495_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S501: S495 (vol dry-up + weekly support) + ATR compression.
    Double quiet: low volume AND tight range = maximum capitulation signature."""
    if sig_s495_s215_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s502_s495_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S502: S495 (vol dry-up + weekly support) + NR7.
    Volume quiet AND range at 7d minimum = extreme dual compression."""
    if sig_s495_s215_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s503_s333_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S503: S333 (MACD turning at weekly support) + below-average volume.
    MACD momentum shift on quiet day = institutional buying without fanfare."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s504_s407_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S504: S407 (VIX pct>70 + weekly support, $1.45M) + below-average volume.
    Fear regime + quiet capitulation = sellers exhausted in high-fear environment."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s505_s477_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S505: S477 (SPY momentum + weekly support) + below-average volume.
    SPY building + TSLA quiet at support = low-key divergence before catch-up."""
    if sig_s477_s215_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s506_s464_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S506: S464 (SPY>200MA + weekly support, $1.04M) + below-average volume.
    Bull regime + quiet accumulation at support = low-noise institutional entry."""
    if sig_s464_s215_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s507_s495_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S507: S495 (vol dry-up + weekly support) + VIX pct>70.
    Triple: quiet accumulation + macro fear + multi-TF support."""
    if sig_s495_s215_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s508_s495_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S508: S495 (vol dry-up + weekly support) + SPY near 10d high.
    Volume capitulation + SPY building = perfect divergence: TSLA silent, SPY loud."""
    if sig_s495_s215_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_new_nd_high(spy, i, n=10) else 0


def sig_s509_s215_inside_bar(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S509: S215 (weekly support) + inside bar (today's range inside yesterday's).
    NEW: inside bar at weekly support = ultimate range compression before reversal."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _inside_bar(tsla, i) else 0


def sig_s510_s495_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S510: S495 (vol dry-up + weekly support) + TSLA above 200d MA.
    Quiet accumulation in bull uptrend = institutions adding in best context."""
    if sig_s495_s215_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 0
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if float(tsla['close'].iloc[i]) > ma200 else 0


SIGNALS_P9A: List[Tuple] = [
    # Phase 9A — Vol dry-up extensions + inside bar (S501-S510)
    ('S501_s495_compressed',     sig_s501_s495_compressed,     10, 0.20, 50),
    ('S502_s495_nr7',            sig_s502_s495_nr7,            10, 0.20, 50),
    ('S503_s333_vol_dryup',      sig_s503_s333_vol_dryup,      10, 0.20, 50),
    ('S504_s407_vol_dryup',      sig_s504_s407_vol_dryup,      10, 0.20, 50),
    ('S505_s477_vol_dryup',      sig_s505_s477_vol_dryup,      10, 0.20, 50),
    ('S506_s464_vol_dryup',      sig_s506_s464_vol_dryup,      10, 0.20, 50),
    ('S507_s495_vix_pct70',      sig_s507_s495_vix_pct70,      10, 0.20, 50),
    ('S508_s495_spy_momentum',   sig_s508_s495_spy_momentum,   10, 0.20, 50),
    ('S509_s215_inside_bar',     sig_s509_s215_inside_bar,     10, 0.20, 50),
    ('S510_s495_above200ma',     sig_s510_s495_above200ma,     10, 0.20, 50),
]

# ── Phase 9B — Inside bar extensions + 4-way compression combos ───────────────
# S509 (inside bar at weekly support) = 83% WR, 4/5yr — good new dimension.
# Extend inside bar with every strong filter. Also test 4-way extreme compression.

def sig_s511_s509_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S511: S509 (inside bar + weekly support) + ATR compression.
    Double compression: today's range inside yesterday's AND tight vs 20d ATR."""
    if sig_s509_s215_inside_bar(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s512_s509_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S512: S509 (inside bar + weekly support) + NR7.
    Inside bar AND narrowest range in 7 days = maximum range compression signal."""
    if sig_s509_s215_inside_bar(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s513_s509_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S513: S509 (inside bar + weekly support) + volume dry-up.
    Inside bar + quiet volume = range AND volume both compressed simultaneously."""
    if sig_s509_s215_inside_bar(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s514_s333_inside_bar(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S514: S333 (MACD turning) + inside bar.
    MACD inflection + inside bar = momentum shift signaled under maximum compression."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _inside_bar(tsla, i) else 0


def sig_s515_s407_inside_bar(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S515: S407 (VIX pct>70 + weekly support, $1.45M) + inside bar.
    Sustained fear + inside bar compression = coiled fear spring."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _inside_bar(tsla, i) else 0


def sig_s516_s477_inside_bar(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S516: S477 (SPY momentum + weekly support) + inside bar.
    SPY at 10d high + TSLA inside bar = SPY breaking out while TSLA compresses."""
    if sig_s477_s215_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _inside_bar(tsla, i) else 0


def sig_s517_s464_inside_bar(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S517: S464 (SPY>200MA + weekly support, $1.04M) + inside bar.
    Bull macro regime + inside bar compression at support."""
    if sig_s464_s215_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _inside_bar(tsla, i) else 0


def sig_s518_s495_inside_bar(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S518: S495 (vol dry-up + weekly support) + inside bar.
    Volume quiet AND range inside yesterday = BOTH volume and price fully compressed."""
    if sig_s495_s215_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _inside_bar(tsla, i) else 0


def sig_s519_s509_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S519: S509 (inside bar + weekly support) + VIX pct>60.
    Inside bar compression + broad fear regime = coiled spring in fear environment."""
    if sig_s509_s215_inside_bar(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s520_s501_inside_bar(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S520: S501 (vol dry-up + weekly support + ATR compressed) + inside bar.
    4-way extreme: ATR compression + volume quiet + inside bar + weekly support."""
    if sig_s501_s495_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _inside_bar(tsla, i) else 0


SIGNALS_P9B: List[Tuple] = [
    # Phase 9B — Inside bar extensions + 4-way compression (S511-S520)
    ('S511_s509_compressed',     sig_s511_s509_compressed,     10, 0.20, 50),
    ('S512_s509_nr7',            sig_s512_s509_nr7,            10, 0.20, 50),
    ('S513_s509_vol_dryup',      sig_s513_s509_vol_dryup,      10, 0.20, 50),
    ('S514_s333_inside_bar',     sig_s514_s333_inside_bar,     10, 0.20, 50),
    ('S515_s407_inside_bar',     sig_s515_s407_inside_bar,     10, 0.20, 50),
    ('S516_s477_inside_bar',     sig_s516_s477_inside_bar,     10, 0.20, 50),
    ('S517_s464_inside_bar',     sig_s517_s464_inside_bar,     10, 0.20, 50),
    ('S518_s495_inside_bar',     sig_s518_s495_inside_bar,     10, 0.20, 50),
    ('S519_s509_vix_pct60',      sig_s519_s509_vix_pct60,      10, 0.20, 50),
    ('S520_s501_inside_bar',     sig_s520_s501_inside_bar,     10, 0.20, 50),
]

# ── Phase 9C — VIX cooldown (x18 finding) + RSI recovery signals ──────────────
# x18 discovery: when VIX was elevated and now DROPPING, high RSI marks the
# BEGINNING of a run, not exhaustion. Captures the transition from fear→risk-on.
# Also test RSI rising from oversold as standalone dimension.

def _vix_was_elevated_now_cooling(vix, i, lookback: int = 10,
                                   spike_pct: float = 0.70,
                                   recovery_pct: float = 0.90) -> bool:
    """VIX was above spike_pct-th percentile of trailing year within last lookback days,
    AND current VIX is below that peak by at least (1-recovery_pct).
    This is the 'VIX cooldown window' — fear peaked, now receding = re-risking begins."""
    if i < 252 + lookback:
        return False
    vix_hist = vix['close'].iloc[i - 252:i].values.astype(float)
    threshold = float(np.percentile(vix_hist, spike_pct * 100))
    vix_window = vix['close'].iloc[max(0, i - lookback):i].values.astype(float)
    vix_peak = float(vix_window.max())
    if vix_peak < threshold:
        return False   # VIX was never elevated in lookback
    vix_now = float(vix['close'].iloc[i])
    return vix_now < vix_peak * recovery_pct   # Dropped 10%+ from peak


def _rsi_rising(rt, i, lookback: int = 3, min_rsi: float = 35.0) -> bool:
    """RSI has been rising over last `lookback` days and is above min_rsi.
    Captures momentum recovery: sellers exhausted, buyers returning."""
    if i < lookback:
        return False
    rsi_now = float(rt.iloc[i])
    rsi_prev = float(rt.iloc[i - lookback])
    return rsi_now > rsi_prev and rsi_now >= min_rsi


def _rsi_recovering_from_oversold(rt, i, oversold: float = 40.0,
                                   lookback: int = 5) -> bool:
    """RSI was below oversold threshold within last lookback days and is now above it.
    Captures the exact crossing: oversold → recovering = momentum reversal confirmed."""
    if i < lookback:
        return False
    rsi_now = float(rt.iloc[i])
    if rsi_now < oversold:
        return False   # still oversold, no recovery yet
    for j in range(1, lookback + 1):
        if i - j >= 0 and float(rt.iloc[i - j]) < oversold:
            return True   # was oversold, now recovered
    return False


def sig_s521_s215_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S521: S215 (weekly support) + VIX cooldown (was elevated, now dropping).
    VIX spike followed by cooldown at weekly support = transition from panic to recovery."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s522_s521_rsi_rising(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S522: S521 (VIX cooldown + weekly support) + RSI rising.
    x18 core finding: VIX fear receding + RSI recovering = BEGINNING of run, not end.
    High RSI during VIX cooldown is momentum resumption, not overbought."""
    if sig_s521_s215_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _rsi_rising(rt, i, lookback=3, min_rsi=35.0) else 0


def sig_s523_s521_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S523: S521 (VIX cooldown + weekly support) + ATR compression.
    VIX cooling + price still coiled = spring loading after fear storm passes."""
    if sig_s521_s215_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s524_s521_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S524: S521 (VIX cooldown + weekly support) + NR7.
    Fear receding + narrowest range bar = ultimate compression at peak fear exit."""
    if sig_s521_s215_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s525_s521_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S525: S521 (VIX cooldown + weekly support) + volume dry-up.
    VIX fear receding + quiet accumulation = sellers exiting, smart money loading."""
    if sig_s521_s215_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S526: S333 (MACD turning) + VIX cooldown.
    MACD momentum inflection coincides with VIX fear transition = dual confirmation."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s527_s477_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S527: S477 (SPY momentum + weekly support) + VIX cooldown.
    SPY building momentum while VIX recedes = ultimate macro + micro confirmation."""
    if sig_s477_s215_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s528_s215_rsi_recover(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S528: S215 (weekly support) + RSI recovering from oversold (crossed above 40).
    RSI was below 40 within last 5 days, now above = momentum reversal confirmed."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _rsi_recovering_from_oversold(rt, i, oversold=40.0, lookback=5) else 0


def sig_s529_s333_rsi_recover(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S529: S333 (MACD turning) + RSI recovering from oversold.
    MACD turns + RSI crosses above oversold = double momentum confirmation."""
    if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _rsi_recovering_from_oversold(rt, i, oversold=40.0, lookback=5) else 0


def sig_s530_s407_rsi_rising(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S530: S407 (VIX pct>70, $1.45M) + RSI rising (not yet overbought).
    x18 insight: in VIX fear regime, RSI rising = acceleration, not exhaustion.
    RSI at 45-65 rising during elevated VIX = early momentum, not late-stage."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _rsi_rising(rt, i, lookback=3, min_rsi=40.0) else 0


SIGNALS_P9C: List[Tuple] = [
    # Phase 9C — VIX cooldown + RSI recovery signals (S521-S530)
    ('S521_s215_vix_cooldown',   sig_s521_s215_vix_cooldown,   10, 0.20, 50),
    ('S522_s521_rsi_rising',     sig_s522_s521_rsi_rising,     10, 0.20, 50),
    ('S523_s521_compressed',     sig_s523_s521_compressed,     10, 0.20, 50),
    ('S524_s521_nr7',            sig_s524_s521_nr7,            10, 0.20, 50),
    ('S525_s521_vol_dryup',      sig_s525_s521_vol_dryup,      10, 0.20, 50),
    ('S526_s333_vix_cooldown',   sig_s526_s333_vix_cooldown,   10, 0.20, 50),
    ('S527_s477_vix_cooldown',   sig_s527_s477_vix_cooldown,   10, 0.20, 50),
    ('S528_s215_rsi_recover',    sig_s528_s215_rsi_recover,    10, 0.20, 50),
    ('S529_s333_rsi_recover',    sig_s529_s333_rsi_recover,    10, 0.20, 50),
    ('S530_s407_rsi_rising',     sig_s530_s407_rsi_rising,     10, 0.20, 50),
]

# ── Phase 9D — VIX cooldown extensions + breakout prediction ──────────────────
# S526 ($1.079M) and S522 (90% WR) confirmed VIX cooldown is highly predictive.
# Extend: VIX cooldown + NR7 + vol_dryup 3-ways; breakout prediction (upper channel).
# Also test shorter and longer VIX cooldown lookbacks.

def sig_s531_s526_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S531: S526 (MACD+VIX cooldown, $1.079M) + ATR compression.
    MACD turning + VIX fear receding + price still coiled = triple convergence."""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s532_s526_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S532: S526 (MACD+VIX cooldown) + NR7.
    MACD + VIX cooldown + narrowest range = all convergence signals firing."""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s533_s526_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S533: S526 (MACD+VIX cooldown) + volume dry-up.
    MACD momentum shift + VIX fear receding + quiet accumulation = ideal entry."""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s534_s526_rsi_rising(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S534: S526 (MACD+VIX cooldown) + RSI rising.
    3-way: MACD turning + VIX receding + RSI accelerating = max momentum signal."""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _rsi_rising(rt, i, lookback=3, min_rsi=35.0) else 0


def sig_s535_s407_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S535: S407 (VIX pct>70 + weekly support, $1.45M) + VIX now cooling.
    Was in top 30% fear, now receding — catches the exact transition point.
    Hypothesis: VIX pct70 is a sustained fear filter; adding cooldown = better timing."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s536_s464_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S536: S464 (SPY>200MA + weekly support, $1.04M) + VIX cooldown.
    Bull regime + VIX fear receding = risk-on rotation into underperforming TSLA."""
    if sig_s464_s215_spy_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s537_s495_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S537: S495 (vol dry-up + weekly support, 100% WR) + VIX cooldown.
    Volume already quiet + VIX cooling confirms sellers fully exhausted."""
    if sig_s495_s215_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s538_s522_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S538: S522 (VIX cooldown + RSI rising + weekly support) + NR7.
    4-way: x18 core finding + narrowest range = maximum precision entry."""
    if sig_s522_s521_rsi_rising(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s539_s522_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S539: S522 (VIX cooldown + RSI rising + weekly support) + ATR compression.
    x18 finding + ATR coiled = double momentum + compression confirmation."""
    if sig_s522_s521_rsi_rising(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s540_s526_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S540: S526 (MACD+VIX cooldown, $1.079M) + TSLA above 200d MA.
    Best cooldown signal restricted to bull uptrend only — avoids bear-market traps."""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 0
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if float(tsla['close'].iloc[i]) > ma200 else 0


SIGNALS_P9D: List[Tuple] = [
    # Phase 9D — VIX cooldown extensions (S531-S540)
    ('S531_s526_compressed',     sig_s531_s526_compressed,     10, 0.20, 50),
    ('S532_s526_nr7',            sig_s532_s526_nr7,            10, 0.20, 50),
    ('S533_s526_vol_dryup',      sig_s533_s526_vol_dryup,      10, 0.20, 50),
    ('S534_s526_rsi_rising',     sig_s534_s526_rsi_rising,     10, 0.20, 50),
    ('S535_s407_vix_cooldown',   sig_s535_s407_vix_cooldown,   10, 0.20, 50),
    ('S536_s464_vix_cooldown',   sig_s536_s464_vix_cooldown,   10, 0.20, 50),
    ('S537_s495_vix_cooldown',   sig_s537_s495_vix_cooldown,   10, 0.20, 50),
    ('S538_s522_nr7',            sig_s538_s522_nr7,            10, 0.20, 50),
    ('S539_s522_compressed',     sig_s539_s522_compressed,     10, 0.20, 50),
    ('S540_s526_above200ma',     sig_s540_s526_above200ma,     10, 0.20, 50),
]

# ── Phase 9E — Below-50MA golden zone + triple dimension combos ───────────────
# New dimension: TSLA below its 50d MA at weekly support = deep pullback opportunity.
# "Golden zone": TSLA above 200MA (bull trend) but below 50MA (medium-term pullback).
# Also test missing 3-way combos: NR7+vol_dryup, SPY_momentum+vol_dryup.

def _below_50ma(tsla, i) -> bool:
    """TSLA close is below its 50d simple moving average — medium-term oversold."""
    if i < 50:
        return True
    ma50 = float(tsla['close'].iloc[i - 50:i].mean())
    return float(tsla['close'].iloc[i]) < ma50


def sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S541: S215 (weekly support) + TSLA below 50d MA.
    NEW: Pullback signal — at weekly support AND below medium-term trend = deep discount."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _below_50ma(tsla, i) else 0


def sig_s542_s541_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S542: S541 (weekly support + below 50MA) + TSLA above 200d MA.
    'Golden zone': bull trend (>200MA) + medium-term pullback (<50MA) = maximum discount in uptrend.
    Price is down from its recent trend but the big trend is still intact."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 0
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if float(tsla['close'].iloc[i]) > ma200 else 0


def sig_s543_s541_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S543: S541 (below 50MA + weekly support) + ATR compression.
    Deep pullback + coiling volatility = spring-loaded at support."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s544_s541_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S544: S541 (below 50MA + weekly support) + VIX pct>60.
    Deep pullback + macro fear = maximum discount in elevated-fear regime."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s545_s541_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S545: S541 (below 50MA + weekly support) + volume dry-up.
    Below medium-term MA + quiet accumulation = institutional buying at discount."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s546_s526_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S546: S526 (MACD+VIX cooldown, $1.079M) + SPY near 10d high.
    Triple: MACD momentum + VIX fear receding + SPY building = every dimension aligned."""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_new_nd_high(spy, i, n=10) else 0


def sig_s547_s215_nr7_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S547: S215 (weekly support) + NR7 + volume dry-up.
    Range AND volume both compressed simultaneously = double quiet at support."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _nr7(tsla, i):
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s548_s408_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S548: S408 (MACD+VIX pct>70, WR=93%) + volume dry-up.
    3-way: MACD momentum + sustained fear + quiet accumulation = maximum precision."""
    if sig_s408_s333_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s549_s477_vol_dryup_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S549: S477 (SPY momentum + weekly support) + NR7 + volume dry-up.
    3-way: SPY at high + NR7 + volume quiet = SPY momentum while TSLA ultra-compressed."""
    if sig_s477_s215_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _nr7(tsla, i):
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s550_mega_3way(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S550 MEGA: S215 (weekly support) + NR7 + vol_dryup + VIX pct>70.
    Three strongest non-MACD dimensions combined:
    - VIX pct70 (sustained fear regime)
    - NR7 (narrowest range in 7 days)
    - Volume dry-up (below-average accumulation)
    Hypothesis: fear + range compression + volume quiet at weekly support = maximum edge."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _vix_elevated_pct(vix, i, window=252, pct=0.70):
        return 0
    if not _nr7(tsla, i):
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


SIGNALS_P9E: List[Tuple] = [
    # Phase 9E — Below-50MA golden zone + triple combos (S541-S550)
    ('S541_s215_below50ma',      sig_s541_s215_below50ma,      10, 0.20, 50),
    ('S542_s541_above200ma',     sig_s542_s541_above200ma,     10, 0.20, 50),
    ('S543_s541_compressed',     sig_s543_s541_compressed,     10, 0.20, 50),
    ('S544_s541_vix_pct60',      sig_s544_s541_vix_pct60,      10, 0.20, 50),
    ('S545_s541_vol_dryup',      sig_s545_s541_vol_dryup,      10, 0.20, 50),
    ('S546_s526_spy_momentum',   sig_s546_s526_spy_momentum,   10, 0.20, 50),
    ('S547_s215_nr7_vol_dryup',  sig_s547_s215_nr7_vol_dryup,  10, 0.20, 50),
    ('S548_s408_vol_dryup',      sig_s548_s408_vol_dryup,      10, 0.20, 50),
    ('S549_s477_vol_nr7',        sig_s549_s477_vol_dryup_nr7,  10, 0.20, 50),
    ('S550_mega_3way',           sig_s550_mega_3way,           10, 0.20, 50),
]

# ── Phase 9F — Below-50MA deeper extensions + energy at support ───────────────
# S541 ($1.87M) and S543 ($1.24M, WR=94%) are elite tier. Push further:
# - Extend below50MA with NR7, VIX cooldown, SPY momentum
# - Add "energy at support" concept: use ATR expansion at support as BUY (energy counter-indicator)
# - Test below-20MA (extreme oversold) for even deeper discount setups

def sig_s551_s541_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S551: S541 (below50MA + weekly support) + NR7.
    Below medium-term MA + narrowest range in 7d = deep pullback with compression."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s552_s541_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S552: S541 (below50MA + weekly support) + VIX cooldown.
    Medium-term oversold + VIX fear receding = double confirmation of recovery."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s553_s541_spy_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S553: S541 (below50MA + weekly support) + SPY near 10d high.
    TSLA medium-term oversold + SPY momentum = lag-catch at discount."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_new_nd_high(spy, i, n=10) else 0


def sig_s554_s541_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S554: S541 (below50MA + weekly support) + MACD histogram turning.
    Below 50MA + MACD momentum shift = medium-term oversold with momentum reversing."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


def sig_s555_s543_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S555: S543 (below50MA + compressed, WR=94%) + VIX pct>60.
    Adds fear regime to the highest-WR signal: 3 dimensions in perfect alignment."""
    if sig_s543_s541_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s556_s543_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S556: S543 (below50MA + compressed) + NR7.
    WR=94% base + NR7 = triple compression at medium-term oversold level."""
    if sig_s543_s541_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s557_s543_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S557: S543 (below50MA + compressed) + volume dry-up.
    Triple compression: below50MA + ATR tight + volume quiet = maximum exhaustion."""
    if sig_s543_s541_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s558_s541_rsi_rising(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S558: S541 (below50MA + weekly support) + RSI rising.
    Below medium-term MA + RSI recovering = early-stage reversal confirmation."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _rsi_rising(rt, i, lookback=3, min_rsi=35.0) else 0


def sig_s559_s541_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S559: S541 (below50MA + weekly support) + VIX pct>70.
    Below medium-term MA at weekly support + sustained fear = max discount in high-fear."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s560_s543_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S560: S543 (below50MA + compressed, WR=94%) + TSLA above 200d MA.
    'Perfect golden zone': bull trend (>200MA) + medium oversold (<50MA) + ATR coiled.
    All three conditions simultaneously = maximum precision entry in bull cycle."""
    if sig_s543_s541_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 0
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if float(tsla['close'].iloc[i]) > ma200 else 0


SIGNALS_P9F: List[Tuple] = [
    # Phase 9F — Below-50MA deeper extensions (S551-S560)
    ('S551_s541_nr7',            sig_s551_s541_nr7,            10, 0.20, 50),
    ('S552_s541_vix_cooldown',   sig_s552_s541_vix_cooldown,   10, 0.20, 50),
    ('S553_s541_spy_momentum',   sig_s553_s541_spy_momentum,   10, 0.20, 50),
    ('S554_s541_macd',           sig_s554_s541_macd,           10, 0.20, 50),
    ('S555_s543_vix_pct60',      sig_s555_s543_vix_pct60,      10, 0.20, 50),
    ('S556_s543_nr7',            sig_s556_s543_nr7,            10, 0.20, 50),
    ('S557_s543_vol_dryup',      sig_s557_s543_vol_dryup,      10, 0.20, 50),
    ('S558_s541_rsi_rising',     sig_s558_s541_rsi_rising,     10, 0.20, 50),
    ('S559_s541_vix_pct70',      sig_s559_s541_vix_pct70,      10, 0.20, 50),
    ('S560_s543_above200ma',     sig_s560_s543_above200ma,     10, 0.20, 50),
]

# ── Phase 9G — Large decline magnitude + weekly RSI + relative lag ────────────
# New dimensions: how FAR has TSLA fallen? Deeper pullbacks at support = more urgent buyers.
# - n-day decline magnitude: TSLA fell >X% in last n days
# - Relative lag vs SPY: TSLA underperformed SPY by X% over lookback
# - Below 20d MA (tighter than 50d — more recent oversold)
# - High-energy at support (energy counter-indicator flip: high energy at LOWER channel = BUY)

def _tsla_declined_pct(tsla, i, lookback: int = 10, min_decline: float = 0.08) -> bool:
    """TSLA has declined at least min_decline over the last lookback days."""
    if i < lookback:
        return False
    price_now  = float(tsla['close'].iloc[i])
    price_prev = float(tsla['close'].iloc[i - lookback])
    if price_prev <= 0:
        return False
    decline = (price_prev - price_now) / price_prev
    return decline >= min_decline


def _tsla_lagging_spy(tsla, spy, i, lookback: int = 20, lag: float = 0.08) -> bool:
    """TSLA has underperformed SPY by at least lag over lookback days.
    Captures the TSLA-specific lag effect: SPY recovered but TSLA didn't."""
    if i < lookback:
        return False
    tsla_ret = (float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - lookback])) - 1.0
    spy_ret  = (float(spy['close'].iloc[i])  / float(spy['close'].iloc[i - lookback]))  - 1.0
    return (spy_ret - tsla_ret) >= lag


def _below_20ma(tsla, i) -> bool:
    """TSLA close is below its 20d simple MA — short-term oversold."""
    if i < 20:
        return True
    ma20 = float(tsla['close'].iloc[i - 20:i].mean())
    return float(tsla['close'].iloc[i]) < ma20


def sig_s561_s215_declined10pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S561: S215 (weekly support) + TSLA declined 10%+ over last 10 days.
    Magnitude filter: stock that dropped >10% is more urgently oversold at support."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_declined_pct(tsla, i, lookback=10, min_decline=0.10) else 0


def sig_s562_s215_declined15pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S562: S215 (weekly support) + TSLA declined 15%+ over last 15 days.
    Deeper pullback = even stronger mean-reversion candidate."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_declined_pct(tsla, i, lookback=15, min_decline=0.15) else 0


def sig_s563_s541_declined10pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S563: S541 (below50MA + weekly support) + TSLA declined 10%+ in 10 days.
    Below medium-term MA + substantial decline = double oversold at support."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_declined_pct(tsla, i, lookback=10, min_decline=0.10) else 0


def sig_s564_s215_tsla_lag_spy(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S564: S215 (weekly support) + TSLA underperformed SPY by 8%+ over 20 days.
    TSLA is lagging the market: SPY recovered but TSLA hasn't yet = lag catch incoming."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.08) else 0


def sig_s565_s407_tsla_lag_spy(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S565: S407 (VIX pct>70 + weekly support) + TSLA lags SPY by 8%+ over 20d.
    Fear regime + TSLA lags market = discount in high-fear = maximum catch-up opportunity."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.08) else 0


def sig_s566_s215_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S566: S215 (weekly support) + TSLA below 20d MA (short-term oversold).
    Tighter than 50MA: below 20MA at weekly support = recent trend broken but structural low holds."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _below_20ma(tsla, i) else 0


def sig_s567_s543_tsla_lag_spy(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S567: S543 (below50MA + compressed, WR=94%) + TSLA lags SPY 8%+.
    WR=94% setup + TSLA is specifically underperforming = targeted lag-catch entry."""
    if sig_s543_s541_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.08) else 0


def sig_s568_s541_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S568: S541 (below50MA + weekly support) + below 20d MA.
    Double MA oversold: below both 50MA and 20MA at weekly support = deep oversold."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _below_20ma(tsla, i) else 0


def sig_s569_s559_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S569: S559 (below50MA + VIX pct70, $1.19M) + ATR compression.
    Best $1M signal with ATR compression = maximum multi-dimension alignment."""
    if sig_s559_s541_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s570_s554_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S570: S554 (below50MA + MACD, $1.10M) + VIX pct>60.
    3-way: below medium-term MA + MACD turning + fear regime = max precision."""
    if sig_s554_s541_macd(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


SIGNALS_P9G: List[Tuple] = [
    # Phase 9G — Decline magnitude + relative lag + double MA oversold (S561-S570)
    ('S561_s215_declined10pct',  sig_s561_s215_declined10pct,  10, 0.20, 50),
    ('S562_s215_declined15pct',  sig_s562_s215_declined15pct,  10, 0.20, 50),
    ('S563_s541_declined10pct',  sig_s563_s541_declined10pct,  10, 0.20, 50),
    ('S564_s215_tsla_lag_spy',   sig_s564_s215_tsla_lag_spy,   10, 0.20, 50),
    ('S565_s407_tsla_lag_spy',   sig_s565_s407_tsla_lag_spy,   10, 0.20, 50),
    ('S566_s215_below20ma',      sig_s566_s215_below20ma,      10, 0.20, 50),
    ('S567_s543_tsla_lag_spy',   sig_s567_s543_tsla_lag_spy,   10, 0.20, 50),
    ('S568_s541_below20ma',      sig_s568_s541_below20ma,      10, 0.20, 50),
    ('S569_s559_compressed',     sig_s569_s559_compressed,     10, 0.20, 50),
    ('S570_s554_vix_pct60',      sig_s570_s554_vix_pct60,      10, 0.20, 50),
]

# ── Phase 9H — Cross-dimension combos: lag+fear, below-MA+fear, RSI deep ─────
# Combine the strongest new dimensions (TSLA lag SPY, below 20MA) with existing
# proven filters (VIX fear, RSI oversold, vol dry-up) for maximum precision.

def sig_s571_s564_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S571: S564 (S215+TSLA_lag_SPY, $1.18M) + VIX pct>60.
    3-way: weekly support + TSLA specific lag + fear regime = maximum catch-up setup."""
    if sig_s564_s215_tsla_lag_spy(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s572_s541_tsla_lag_spy(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S572: S541 (below50MA + weekly support, $1.87M) + TSLA lags SPY 8%+.
    Highest P&L base signal + TSLA-specific lag = extended dislocation at support."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.08) else 0


def sig_s573_s566_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S573: S566 (below20MA + weekly support, $1.22M) + VIX pct>60.
    Short-term oversold at structural low in fear regime = max urgency BUY."""
    if sig_s566_s215_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s574_s568_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S574: S568 (double MA: below50 + below20 + weekly support) + VIX pct>60.
    Triple filter: two MA layers + structural support + fear."""
    if sig_s568_s541_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s575_s541_consec_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S575: S541 (below50MA + weekly support) + 3+ consecutive down days.
    Momentum exhaustion signal: extended red streak at structural low."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=3) else 0


def sig_s576_s564_rsi_rising(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S576: S564 (S215+TSLA_lag_SPY) + RSI rising (momentum turning).
    Lag catch entry + first sign of TSLA momentum reversal = ideal timing."""
    if sig_s564_s215_tsla_lag_spy(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _rsi_rising(rt, i) else 0


def sig_s577_s541_rsi_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S577: S541 (below50MA + weekly support) + RSI below 40 (deeply oversold).
    RSI<40 at structural low: quantified oversold at medium-term support."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 40.0 else 0


def sig_s578_s526_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S578: S526 (MACD+VIX cooldown, $1.08M, 92% WR) + below 50MA.
    VIX fear receding + MACD turning + below medium-term MA = perfect recovery timing."""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _below_50ma(tsla, i) else 0


def sig_s579_s566_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S579: S566 (below20MA + weekly support) + volume dry-up.
    Short-term oversold at support with sellers exhausted = capitulation floor."""
    if sig_s566_s215_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s580_s567_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S580: S567 (S543+TSLA_lag_SPY, 100% WR, n=11) + VIX pct>60.
    Perfect WR base + fear filter = highest conviction entry."""
    if sig_s567_s543_tsla_lag_spy(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


SIGNALS_P9H: List[Tuple] = [
    # Phase 9H — Cross-dimension combos (S571-S580)
    ('S571_s564_vix_pct60',    sig_s571_s564_vix_pct60,    10, 0.20, 50),
    ('S572_s541_tsla_lag_spy', sig_s572_s541_tsla_lag_spy, 10, 0.20, 50),
    ('S573_s566_vix_pct60',    sig_s573_s566_vix_pct60,    10, 0.20, 50),
    ('S574_s568_vix_pct60',    sig_s574_s568_vix_pct60,    10, 0.20, 50),
    ('S575_s541_consec_down',  sig_s575_s541_consec_down,  10, 0.20, 50),
    ('S576_s564_rsi_rising',   sig_s576_s564_rsi_rising,   10, 0.20, 50),
    ('S577_s541_rsi_low',      sig_s577_s541_rsi_low,      10, 0.20, 50),
    ('S578_s526_below50ma',    sig_s578_s526_below50ma,    10, 0.20, 50),
    ('S579_s566_vol_dryup',    sig_s579_s566_vol_dryup,    10, 0.20, 50),
    ('S580_s567_vix_pct60',    sig_s580_s567_vix_pct60,    10, 0.20, 50),
]

# ── Phase 9I — New-dimension pairs: lag+20MA, lag+fear+MA, RSI/MA depth ──────
# Stack the two new verified dimensions (TSLA_lag_SPY + below_20MA) against each
# other and against proven filters. Also explore shorter lag thresholds.

def sig_s581_s566_tsla_lag_spy(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S581: S566 (below20MA + weekly support) + TSLA lags SPY 8%+.
    Two orthogonal new dimensions stacked: MA oversold AND specific TSLA underperformance."""
    if sig_s566_s215_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.08) else 0


def sig_s582_s541_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S582: S541 (below50MA + weekly support) + TSLA lags SPY 5%+ (looser threshold).
    More trades than 8% lag — tests whether looser lag filter still adds edge."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s583_s541_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S583: S541 (below50MA + weekly support, $1.87M) + VIX cooldown.
    Highest P&L signal + VIX fear receding = two independent strength indicators."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s584_s526_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S584: S526 (MACD+VIX_cooldown, $1.08M, 92% WR) + below 20MA.
    Short-term MA breakdown + VIX fear receding + MACD turning = multi-layer confirmation."""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _below_20ma(tsla, i) else 0


def sig_s585_s564_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S585: S564 (S215+TSLA_lag_SPY, $1.18M, 86% WR) + below 20MA.
    TSLA-specific lag + short-term MA breakdown at weekly support = dual new dimensions."""
    if sig_s564_s215_tsla_lag_spy(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _below_20ma(tsla, i) else 0


def sig_s586_s566_rsi_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S586: S566 (below20MA + weekly support) + RSI below 40.
    Short-term MA breakdown at support + quantified oversold level = depth confirmation."""
    if sig_s566_s215_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 40.0 else 0


def sig_s587_s541_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S587: S541 (below50MA + weekly support) + above 200d MA (long-term uptrend).
    Golden zone filter: only take below-50MA dips when long-term trend is still bullish."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if float(tsla['close'].iloc[i]) > ma200 else 0


def sig_s588_s564_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S588: S564 (TSLA_lag_SPY at weekly support) + volume dry-up.
    TSLA-specific lag catch + sellers exhausted = strong timing for lag reversal."""
    if sig_s564_s215_tsla_lag_spy(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s589_s566_consec_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S589: S566 (below20MA + weekly support) + 3+ consecutive down days.
    Recent MA breakdown at support with extended red streak = momentum exhaustion."""
    if sig_s566_s215_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=3) else 0


def sig_s590_s578_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S590: S578 (S526+below50MA, $966K, 91% WR) + below 20MA.
    4-way: MACD turning + VIX cooling + below 50MA + below 20MA = maximum depth at support."""
    if sig_s578_s526_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _below_20ma(tsla, i) else 0


SIGNALS_P9I: List[Tuple] = [
    # Phase 9I — New-dimension pairs + depth combos (S581-S590)
    ('S581_s566_tsla_lag_spy', sig_s581_s566_tsla_lag_spy, 10, 0.20, 50),
    ('S582_s541_lag5pct',      sig_s582_s541_lag5pct,      10, 0.20, 50),
    ('S583_s541_vix_cooldown', sig_s583_s541_vix_cooldown, 10, 0.20, 50),
    ('S584_s526_below20ma',    sig_s584_s526_below20ma,    10, 0.20, 50),
    ('S585_s564_below20ma',    sig_s585_s564_below20ma,    10, 0.20, 50),
    ('S586_s566_rsi_low',      sig_s586_s566_rsi_low,      10, 0.20, 50),
    ('S587_s541_above200ma',   sig_s587_s541_above200ma,   10, 0.20, 50),
    ('S588_s564_vol_dryup',    sig_s588_s564_vol_dryup,    10, 0.20, 50),
    ('S589_s566_consec_down',  sig_s589_s566_consec_down,  10, 0.20, 50),
    ('S590_s578_below20ma',    sig_s590_s578_below20ma,    10, 0.20, 50),
]

# ── Phase 9J — S582 extensions + lag parameter sweep + S600 milestone ─────────
# S582 (S541+lag5pct, $1.376M, 8/8yr) is now the highest-coverage strong signal.
# Extend it with proven filters: VIX fear, RSI, vol dry-up, below-20MA.
# Also test: larger lag threshold (12%), shorter lookback (10d).

def sig_s591_s582_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S591: S582 (S541+lag5pct, 8/8yr) + VIX pct>60.
    Coverage champ + fear regime = precision on the most coverage-proven setup."""
    if sig_s582_s541_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s592_s582_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S592: S582 (S541+lag5pct) + VIX pct>70 (high fear only).
    Tighter fear filter on 8/8yr base — tests whether fear regime boosts WR further."""
    if sig_s582_s541_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s593_s582_rsi_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S593: S582 (S541+lag5pct) + RSI below 40.
    Below-50MA + TSLA lag + deep RSI oversold = triple quantified depth."""
    if sig_s582_s541_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 40.0 else 0


def sig_s594_s582_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S594: S582 (S541+lag5pct) + volume dry-up.
    TSLA lag at support + sellers gone = perfect timing for lag catch."""
    if sig_s582_s541_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s595_s582_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S595: S582 (S541+lag5pct) + below 20MA.
    Two MA layers: below 50d (medium) + below 20d (short) while TSLA lags SPY."""
    if sig_s582_s541_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _below_20ma(tsla, i) else 0


def sig_s596_s541_lag12pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S596: S541 (below50MA + weekly support) + TSLA lags SPY 12%+ over 20d.
    Extreme dislocation: TSLA 12%+ behind SPY at structural low = max urgency."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.12) else 0


def sig_s597_s215_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S597: S215 (weekly support) + TSLA lags SPY 5%+ over 20d (no 50MA req).
    Looser lag threshold without 50MA filter — how much does 50MA constraint add?"""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s598_s541_lag10d_5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S598: S541 (below50MA + weekly support) + TSLA lags SPY 5%+ over 10 days.
    Shorter lookback: recent 10d lag vs SPY — captures faster relative weakness."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=10, lag=0.05) else 0


def sig_s599_s566_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S599: S566 (below20MA + weekly support) + TSLA lags SPY 5%+ over 20d.
    Strongest new below-MA signal + looser lag filter — deep short-term oversold + lag."""
    if sig_s566_s215_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s600_milestone_triple_oversold(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S600 MILESTONE: Triple oversold — below20MA + VIX pct>60 + TSLA lag5pct at weekly support.
    Maximum multi-dimension alignment: structural low + short-term MA breakdown + fear + lag."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _below_20ma(tsla, i):
        return 0
    if not _vix_elevated_pct(vix, i, window=252, pct=0.60):
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


SIGNALS_P9J: List[Tuple] = [
    # Phase 9J — S582 extensions + lag sweeps + S600 milestone (S591-S600)
    ('S591_s582_vix_pct60',       sig_s591_s582_vix_pct60,       10, 0.20, 50),
    ('S592_s582_vix_pct70',       sig_s592_s582_vix_pct70,       10, 0.20, 50),
    ('S593_s582_rsi_low',         sig_s593_s582_rsi_low,         10, 0.20, 50),
    ('S594_s582_vol_dryup',       sig_s594_s582_vol_dryup,       10, 0.20, 50),
    ('S595_s582_below20ma',       sig_s595_s582_below20ma,       10, 0.20, 50),
    ('S596_s541_lag12pct',        sig_s596_s541_lag12pct,        10, 0.20, 50),
    ('S597_s215_lag5pct',         sig_s597_s215_lag5pct,         10, 0.20, 50),
    ('S598_s541_lag10d_5pct',     sig_s598_s541_lag10d_5pct,     10, 0.20, 50),
    ('S599_s566_lag5pct',         sig_s599_s566_lag5pct,         10, 0.20, 50),
    ('S600_milestone_triple_os',  sig_s600_milestone_triple_oversold, 10, 0.20, 50),
]

# ── Phase 9K — SPY health divergence, Bollinger Band oversold, S597 extensions ─
# New helpers: _spy_above_ma(), _tsla_below_bb()
# Key idea: if SPY is still healthy (above its own MA) while TSLA lags,
# the lag is TSLA-specific (not a broad market drop) = stronger catch-up thesis.
# Also explore BB %B as alternative oversold measure.

def _spy_above_ma(spy, i, period: int = 20) -> bool:
    """SPY is above its n-day SMA — market is healthy, TSLA lag is stock-specific."""
    if i < period:
        return True
    ma = float(spy['close'].iloc[i - period:i].mean())
    return float(spy['close'].iloc[i]) > ma


def _tsla_below_bb(tsla, i, period: int = 20, n_std: float = 2.0) -> bool:
    """TSLA close is below lower Bollinger Band (extreme oversold by BB measure)."""
    if i < period:
        return False
    closes = tsla['close'].iloc[i - period:i]
    ma = float(closes.mean())
    std = float(closes.std())
    lower_band = ma - n_std * std
    return float(tsla['close'].iloc[i]) < lower_band


def sig_s601_s597_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S601: S597 (8/8yr: weekly support+lag5pct) + VIX cooldown.
    Broadest-coverage signal + VIX fear receding = timing confirmation."""
    if sig_s597_s215_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s602_s597_spy_healthy(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S602: S597 (weekly support+lag5pct) + SPY above 20d MA.
    TSLA lags but SPY is healthy = TSLA-specific weakness, not market-wide drop."""
    if sig_s597_s215_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_above_ma(spy, i, period=20) else 0


def sig_s603_s541_spy_healthy(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S603: S541 (below50MA + weekly support) + SPY above 20d MA.
    TSLA below 50d MA at structural low while market is still in uptrend."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_above_ma(spy, i, period=20) else 0


def sig_s604_s566_spy_healthy(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S604: S566 (below20MA + weekly support) + SPY above 20d MA.
    Short-term TSLA breakdown at support while SPY still healthy = TSLA-specific."""
    if sig_s566_s215_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_above_ma(spy, i, period=20) else 0


def sig_s605_s215_below_bb(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S605: S215 (weekly support) + TSLA below lower Bollinger Band (2-std).
    Extreme statistical oversold at structural support = very high mean-reversion."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_bb(tsla, i) else 0


def sig_s606_s541_below_bb(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S606: S541 (below50MA + weekly support) + below Bollinger Band.
    Medium-term MA breakdown at structural low + extreme short-term oversold."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_bb(tsla, i) else 0


def sig_s607_s597_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S607: S597 (8/8yr: weekly support+lag5pct) + volume dry-up.
    Highest-coverage signal + exhausted sellers = ideal mean-reversion timing."""
    if sig_s597_s215_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s608_s597_rsi_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S608: S597 (8/8yr: weekly support+lag5pct) + RSI below 40.
    8/8yr coverage signal + quantified oversold depth filter."""
    if sig_s597_s215_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 40.0 else 0


def sig_s609_s597_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S609: S597 (8/8yr: weekly support+lag5pct) + VIX pct>70.
    8/8yr base with highest fear threshold = deepest discount entry."""
    if sig_s597_s215_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s610_s215_lag30d_5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S610: S215 (weekly support) + TSLA lags SPY 5%+ over 30 days.
    Longer lookback for lag: 30-day extended underperformance at structural low."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=30, lag=0.05) else 0


SIGNALS_P9K: List[Tuple] = [
    # Phase 9K — SPY divergence + BB oversold + S597 extensions (S601-S610)
    ('S601_s597_vix_cooldown',  sig_s601_s597_vix_cooldown,  10, 0.20, 50),
    ('S602_s597_spy_healthy',   sig_s602_s597_spy_healthy,   10, 0.20, 50),
    ('S603_s541_spy_healthy',   sig_s603_s541_spy_healthy,   10, 0.20, 50),
    ('S604_s566_spy_healthy',   sig_s604_s566_spy_healthy,   10, 0.20, 50),
    ('S605_s215_below_bb',      sig_s605_s215_below_bb,      10, 0.20, 50),
    ('S606_s541_below_bb',      sig_s606_s541_below_bb,      10, 0.20, 50),
    ('S607_s597_vol_dryup',     sig_s607_s597_vol_dryup,     10, 0.20, 50),
    ('S608_s597_rsi_low',       sig_s608_s597_rsi_low,       10, 0.20, 50),
    ('S609_s597_vix_pct70',     sig_s609_s597_vix_pct70,     10, 0.20, 50),
    ('S610_s215_lag30d_5pct',   sig_s610_s215_lag30d_5pct,   10, 0.20, 50),
]

# ── Phase 9L — Distance from 52-wk high, co-move (SPY also dropping), 3-way ──
# Lessons from 9K: market co-move is REQUIRED for best bounces.
# New dimensions: TSLA distance from 52-week high, SPY below 20MA explicitly.

def _tsla_below_52wk_pct(tsla, i, pct: float = 0.20) -> bool:
    """TSLA close is at least pct% below its 52-week (252d) high.
    High discount from recent high = greater mean-reversion potential."""
    lookback = min(252, i)
    if lookback < 20:
        return False
    high_52wk = float(tsla['close'].iloc[i - lookback:i].max())
    if high_52wk <= 0:
        return False
    discount = (high_52wk - float(tsla['close'].iloc[i])) / high_52wk
    return discount >= pct


def sig_s611_s597_spy_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S611: S597 (8/8yr) + SPY below 20d MA (market also dropping).
    Co-move setup: both TSLA and SPY are under pressure, TSLA just more so."""
    if sig_s597_s215_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _spy_above_ma(spy, i, period=20) else 1


def sig_s612_s541_spy_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S612: S541 (below50MA + weekly support) + SPY below 20d MA (co-move).
    Broad market weakness + TSLA at structural low = correlated bounce setup."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _spy_above_ma(spy, i, period=20) else 1


def sig_s613_s215_52wk_discount20(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S613: S215 (weekly support) + TSLA 20%+ below 52-week high.
    Structural low + significant discount from recent peak = strong mean-reversion."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.20) else 0


def sig_s614_s541_52wk_discount20(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S614: S541 (below50MA + weekly support) + TSLA 20%+ below 52-week high.
    Two-layer oversold: medium-term MA breakdown + large peak discount at support."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.20) else 0


def sig_s615_s597_52wk_discount20(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S615: S597 (8/8yr) + TSLA 20%+ below 52-week high.
    Broadest-coverage signal + substantial discount from peak = deep value setup."""
    if sig_s597_s215_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.20) else 0


def sig_s616_s215_bb_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S616: S215 (weekly support) + BB extreme + TSLA lag5pct (3-way).
    All three independent oversold measures aligned: structural + BB + relative lag."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _tsla_below_bb(tsla, i):
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s617_s526_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S617: S526 (MACD+VIX_cooldown, $1.08M, 92% WR) + TSLA lags SPY 5%+.
    Fear receding + MACD turning + TSLA specific lag = timing + depth confirmation."""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s618_s564_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S618: S564 (S215+TSLA_lag_SPY8%, $1.18M) + VIX cooldown.
    Two independent high-alpha signals stacked: TSLA lag + VIX fear receding."""
    if sig_s564_s215_tsla_lag_spy(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s619_s541_lag10pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S619: S541 (below50MA + weekly support) + TSLA lags SPY 10%+ over 20d.
    Between 8% and 12%: test optimal lag threshold for WR vs trade count."""
    if sig_s541_s215_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.10) else 0


def sig_s620_s597_below_bb(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S620: S597 (8/8yr) + below Bollinger Band.
    8/8yr coverage base + BB extreme = statistical confirmation of depth."""
    if sig_s597_s215_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_bb(tsla, i) else 0


SIGNALS_P9L: List[Tuple] = [
    # Phase 9L — 52-wk discount, co-move, 3-way combos (S611-S620)
    ('S611_s597_spy_below20ma',    sig_s611_s597_spy_below20ma,    10, 0.20, 50),
    ('S612_s541_spy_below20ma',    sig_s612_s541_spy_below20ma,    10, 0.20, 50),
    ('S613_s215_52wk_disc20',      sig_s613_s215_52wk_discount20,  10, 0.20, 50),
    ('S614_s541_52wk_disc20',      sig_s614_s541_52wk_discount20,  10, 0.20, 50),
    ('S615_s597_52wk_disc20',      sig_s615_s597_52wk_discount20,  10, 0.20, 50),
    ('S616_s215_bb_lag5pct',       sig_s616_s215_bb_lag5pct,       10, 0.20, 50),
    ('S617_s526_lag5pct',          sig_s617_s526_lag5pct,          10, 0.20, 50),
    ('S618_s564_vix_cooldown',     sig_s618_s564_vix_cooldown,     10, 0.20, 50),
    ('S619_s541_lag10pct',         sig_s619_s541_lag10pct,         10, 0.20, 50),
    ('S620_s597_below_bb',         sig_s620_s597_below_bb,         10, 0.20, 50),
]

# ── Phase 9M — Extend S613 (record signal) + 52wk discount threshold sweep ───
# S613 (weekly support + 20%+ below 52wk high) = $1.46M, 8/8yr — best signal.
# Now test what filters add value on top, and optimal discount threshold.

def sig_s621_s613_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S621: S613 (52wk_disc20 + weekly support) + VIX pct>60.
    Deep discount from peak at structural low in fear regime = max urgency BUY."""
    if sig_s613_s215_52wk_discount20(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s622_s613_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S622: S613 (52wk_disc20 + weekly support) + TSLA lags SPY 5%+.
    52wk peak discount + TSLA-specific underperformance = double independent dimensions."""
    if sig_s613_s215_52wk_discount20(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s623_s613_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S623: S613 (52wk_disc20 + weekly support) + below 20d MA.
    Three layers: structural support + peak discount + short-term MA breakdown."""
    if sig_s613_s215_52wk_discount20(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _below_20ma(tsla, i) else 0


def sig_s624_s613_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S624: S613 (52wk_disc20 + weekly support) + volume dry-up.
    Deep discount from peak + sellers exhausted = ideal mean-reversion timing."""
    if sig_s613_s215_52wk_discount20(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s625_s613_rsi_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S625: S613 (52wk_disc20 + weekly support) + RSI below 40.
    52wk discount at support + quantified RSI depth = dual oversold confirmation."""
    if sig_s613_s215_52wk_discount20(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 40.0 else 0


def sig_s626_s613_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S626: S613 (52wk_disc20 + weekly support) + VIX cooldown.
    Deep discount from peak + VIX fear receding = multi-layer recovery signal."""
    if sig_s613_s215_52wk_discount20(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s627_s613_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S627: S613 (52wk_disc20 + weekly support) + ATR compression.
    Deep discount from peak at support + low volatility = compression before expansion."""
    if sig_s613_s215_52wk_discount20(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s628_s215_52wk_disc30(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S628: S215 (weekly support) + TSLA 30%+ below 52-week high.
    Deeper discount threshold: only the most distressed setups at structural low."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.30) else 0


def sig_s629_s215_52wk_disc15(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S629: S215 (weekly support) + TSLA 15%+ below 52-week high.
    Looser discount: captures more trades — tests if 20% threshold is optimal."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.15) else 0


def sig_s630_s215_52wk_disc40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S630: S215 (weekly support) + TSLA 40%+ below 52-week high.
    Extreme discount: rare but highest urgency mean-reversion setups."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.40) else 0


SIGNALS_P9M: List[Tuple] = [
    # Phase 9M — S613 extensions + 52wk discount sweep (S621-S630)
    ('S621_s613_vix_pct60',    sig_s621_s613_vix_pct60,    10, 0.20, 50),
    ('S622_s613_lag5pct',      sig_s622_s613_lag5pct,      10, 0.20, 50),
    ('S623_s613_below20ma',    sig_s623_s613_below20ma,    10, 0.20, 50),
    ('S624_s613_vol_dryup',    sig_s624_s613_vol_dryup,    10, 0.20, 50),
    ('S625_s613_rsi_low',      sig_s625_s613_rsi_low,      10, 0.20, 50),
    ('S626_s613_vix_cooldown', sig_s626_s613_vix_cooldown, 10, 0.20, 50),
    ('S627_s613_compressed',   sig_s627_s613_compressed,   10, 0.20, 50),
    ('S628_s215_52wk_disc30',  sig_s628_s215_52wk_disc30,  10, 0.20, 50),
    ('S629_s215_52wk_disc15',  sig_s629_s215_52wk_disc15,  10, 0.20, 50),
    ('S630_s215_52wk_disc40',  sig_s630_s215_52wk_disc40,  10, 0.20, 50),
]

# ── Phase 9N — Complete threshold sweep + S629 extensions (optimal signal) ───
# S629 ($2M, 9/9yr) = best signal. Test 10%/12.5% thresholds to confirm 15% optimal.
# Extend S629 with proven top filters.

def sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S631: S215 (weekly support) + TSLA 10%+ below 52-week high.
    Very loose threshold: captures even modest pullbacks at structural support."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.10) else 0


def sig_s632_s215_52wk_disc12(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S632: S215 (weekly support) + TSLA 12%+ below 52-week high.
    12.5% threshold between 10% and 15% — fine-tuning the optimal point."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.12) else 0


def sig_s633_s629_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S633: S629 (15% disc + weekly support, $2M, 9/9yr) + VIX pct>60.
    Best signal + fear regime = highest conviction entries from record signal."""
    if sig_s629_s215_52wk_disc15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s634_s629_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S634: S629 (15% disc + weekly support) + TSLA lags SPY 5%+.
    Best signal + TSLA-specific underperformance = dual independent oversold."""
    if sig_s629_s215_52wk_disc15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s635_s629_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S635: S629 (15% disc + weekly support) + below 20d MA.
    Best signal + short-term MA breakdown = multi-layer oversold at peak discount."""
    if sig_s629_s215_52wk_disc15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _below_20ma(tsla, i) else 0


def sig_s636_s629_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S636: S629 (15% disc + weekly support) + volume dry-up.
    Best signal + sellers exhausted = lowest-risk entry timing."""
    if sig_s629_s215_52wk_disc15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s637_s629_rsi_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S637: S629 (15% disc + weekly support) + RSI < 40.
    Best signal + RSI quantified oversold = depth confirmation."""
    if sig_s629_s215_52wk_disc15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 40.0 else 0


def sig_s638_s629_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S638: S629 (15% disc + weekly support) + ATR compression.
    Best signal + low volatility compression = breakout about to happen."""
    if sig_s629_s215_52wk_disc15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s639_s629_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S639: S629 (15% disc + weekly support) + VIX pct>70 (high fear).
    Best signal + highest fear tier = deepest discount + maximum fear = max mean-reversion."""
    if sig_s629_s215_52wk_disc15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s640_s629_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S640: S629 (15% disc + weekly support) + VIX cooldown.
    Best signal + VIX fear receding = timing confirmation for 52wk discount entries."""
    if sig_s629_s215_52wk_disc15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


SIGNALS_P9N: List[Tuple] = [
    # Phase 9N — threshold sweep completion + S629 extensions (S631-S640)
    ('S631_s215_52wk_disc10',  sig_s631_s215_52wk_disc10,  10, 0.20, 50),
    ('S632_s215_52wk_disc12',  sig_s632_s215_52wk_disc12,  10, 0.20, 50),
    ('S633_s629_vix_pct60',    sig_s633_s629_vix_pct60,    10, 0.20, 50),
    ('S634_s629_lag5pct',      sig_s634_s629_lag5pct,      10, 0.20, 50),
    ('S635_s629_below20ma',    sig_s635_s629_below20ma,    10, 0.20, 50),
    ('S636_s629_vol_dryup',    sig_s636_s629_vol_dryup,    10, 0.20, 50),
    ('S637_s629_rsi_low',      sig_s637_s629_rsi_low,      10, 0.20, 50),
    ('S638_s629_compressed',   sig_s638_s629_compressed,   10, 0.20, 50),
    ('S639_s629_vix_pct70',    sig_s639_s629_vix_pct70,    10, 0.20, 50),
    ('S640_s629_vix_cooldown', sig_s640_s629_vix_cooldown, 10, 0.20, 50),
]

# ── Phase 9O — Lower discount thresholds (7.5%/5%) + S631 extensions ─────────
# S631 (10% disc, $2.06M, 9/9yr) = new optimal base. Test 7.5%/5% to find floor.
# Also: S631 + proven best filters for highest-conviction entries.

def sig_s641_s215_52wk_disc7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S641: S215 (weekly support) + TSLA 7.5%+ below 52-week high.
    Very loose: even mild pullbacks to weekly support captured."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.075) else 0


def sig_s642_s215_52wk_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S642: S215 (weekly support) + TSLA 5%+ below 52-week high.
    Threshold floor test: nearly any weekly support touch fires."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.05) else 0


def sig_s643_s631_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S643: S631 (10% disc + weekly support, $2M) + VIX pct>60.
    Best signal base + fear regime = max precision on optimal setup."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s644_s631_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S644: S631 (10% disc + weekly support) + TSLA lags SPY 5%+.
    Two independent oversold dimensions: 10% from peak + TSLA-specific underperformance."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s645_s631_below20ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S645: S631 (10% disc + weekly support) + below 20d MA.
    Peak discount at support + short-term MA breakdown = three-layer oversold."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _below_20ma(tsla, i) else 0


def sig_s646_s631_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S646: S631 (10% disc + weekly support) + volume dry-up.
    Best signal + exhausted sellers = max conviction entry timing."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s647_s631_rsi_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S647: S631 (10% disc + weekly support) + RSI < 40.
    10% from peak at weekly support + RSI quantified depth = dual confirmation."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 40.0 else 0


def sig_s648_s631_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S648: S631 (10% disc + weekly support) + ATR compression.
    Best signal + low vol compression at peak discount = pre-breakout entry."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s649_s631_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S649: S631 (10% disc + weekly support) + above 200d MA.
    Long-term uptrend confirmed: only take discount dips in secular bull periods."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if float(tsla['close'].iloc[i]) > ma200 else 0


def sig_s650_s631_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S650: S631 (10% disc + weekly support) + VIX pct>70.
    Best signal + extreme fear tier = deepest discount in highest fear = max opportunity."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


SIGNALS_P9O: List[Tuple] = [
    # Phase 9O — threshold floor + S631 extensions (S641-S650)
    ('S641_s215_52wk_disc7',   sig_s641_s215_52wk_disc7,   10, 0.20, 50),
    ('S642_s215_52wk_disc5',   sig_s642_s215_52wk_disc5,   10, 0.20, 50),
    ('S643_s631_vix_pct60',    sig_s643_s631_vix_pct60,    10, 0.20, 50),
    ('S644_s631_lag5pct',      sig_s644_s631_lag5pct,      10, 0.20, 50),
    ('S645_s631_below20ma',    sig_s645_s631_below20ma,    10, 0.20, 50),
    ('S646_s631_vol_dryup',    sig_s646_s631_vol_dryup,    10, 0.20, 50),
    ('S647_s631_rsi_low',      sig_s647_s631_rsi_low,      10, 0.20, 50),
    ('S648_s631_compressed',   sig_s648_s631_compressed,   10, 0.20, 50),
    ('S649_s631_above200ma',   sig_s649_s631_above200ma,   10, 0.20, 50),
    ('S650_s631_vix_pct70',    sig_s650_s631_vix_pct70,    10, 0.20, 50),
]

# ── Phase 9P — WR=93% S648 extensions + above-200MA combos + final 52wk ──────
# S648 (S631+compressed) = WR=93%, $1M+, 8/8yr — unprecedented precision.
# Extend it and also explore the 200MA uptrend filter as additive dimension.

def sig_s651_s648_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S651: S648 (WR=93%, compressed + 10% disc + weekly) + VIX pct>60.
    Best-WR signal + fear regime — can we push WR above 93%?"""
    if sig_s648_s631_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s652_s648_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S652: S648 (WR=93% compressed) + TSLA lags SPY 5%+.
    WR=93% base + TSLA-specific lag = max independent signal alignment."""
    if sig_s648_s631_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s653_s642_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S653: S642 (5% disc + weekly support) + ATR compression.
    Broadest discount threshold with compression filter: highest trade count at WR=93%."""
    if sig_s642_s215_52wk_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s654_s642_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S654: S642 (5% disc + weekly support) + volume dry-up.
    Most trades (n=34) + sellers exhausted = max throughput 100% WR setups."""
    if sig_s642_s215_52wk_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s655_s649_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S655: S649 (above200MA + 10% disc, WR=88%, PF=19) + VIX pct>60.
    Long-term uptrend confirmed + peak discount + fear regime = highest quality."""
    if sig_s649_s631_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s656_s649_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S656: S649 (above200MA + 10% disc) + TSLA lags SPY 5%+.
    Long-term trend healthy + near peak + TSLA-specific lag = bull dip entry."""
    if sig_s649_s631_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s657_s649_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S657: S649 (above200MA + 10% disc, WR=88%) + ATR compression.
    Uptrend + peak discount + compression = three-way timing signal."""
    if sig_s649_s631_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s658_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S658: S215 (weekly support) + above 200d MA.
    Base weekly support signal restricted to long-term uptrend periods only."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if float(tsla['close'].iloc[i]) > ma200 else 0


def sig_s659_s526_52wk_disc15(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S659: S526 (MACD+VIX_cooldown, $1.08M, 92% WR) + 15%+ below 52wk high.
    Two proven $1M signals stacked: MACD timing + VIX cooldown + peak discount."""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.15) else 0


def sig_s660_s407_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S660: S407 (VIX pct>70 + weekly support, $1.80M, 8/8yr) + 10%+ below 52wk.
    Best historical signals + peak discount = maximum multi-signal confluence."""
    if sig_s407_s215_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.10) else 0


SIGNALS_P9P: List[Tuple] = [
    # Phase 9P — WR=93% extensions + above-200MA + cross-signal combos (S651-S660)
    ('S651_s648_vix_pct60',    sig_s651_s648_vix_pct60,    10, 0.20, 50),
    ('S652_s648_lag5pct',      sig_s652_s648_lag5pct,      10, 0.20, 50),
    ('S653_s642_compressed',   sig_s653_s642_compressed,   10, 0.20, 50),
    ('S654_s642_vol_dryup',    sig_s654_s642_vol_dryup,    10, 0.20, 50),
    ('S655_s649_vix_pct60',    sig_s655_s649_vix_pct60,    10, 0.20, 50),
    ('S656_s649_lag5pct',      sig_s656_s649_lag5pct,      10, 0.20, 50),
    ('S657_s649_compressed',   sig_s657_s649_compressed,   10, 0.20, 50),
    ('S658_s215_above200ma',   sig_s658_s215_above200ma,   10, 0.20, 50),
    ('S659_s526_52wk_disc15',  sig_s659_s526_52wk_disc15,  10, 0.20, 50),
    ('S660_s407_52wk_disc10',  sig_s660_s407_52wk_disc10,  10, 0.20, 50),
]

# ── Phase 9Q — Precision signal extensions + uptrend filter + gap-down ────────
# Extend the best precision signals. New: gap-down to support, above-200MA combos.

def _tsla_gap_down_pct(tsla, i, min_gap: float = 0.02) -> bool:
    """Today's open is at least min_gap% below yesterday's close (gap down)."""
    if i < 1:
        return False
    prev_close = float(tsla['close'].iloc[i - 1])
    today_open = float(tsla['open'].iloc[i]) if 'open' in tsla.columns else prev_close
    if prev_close <= 0:
        return False
    gap = (prev_close - today_open) / prev_close
    return gap >= min_gap


def sig_s661_s631_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S661: S631 (10% disc + weekly support) + gap down >2% today.
    Panic open to structural support with 10% from peak = extreme fear selling."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_gap_down_pct(tsla, i, min_gap=0.02) else 0


def sig_s662_s658_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S662: S658 (above200MA + weekly support) + 10% below 52wk high.
    Long-term uptrend + structural support + meaningful peak discount = quality dip-buy."""
    if sig_s658_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.10) else 0


def sig_s663_s658_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S663: S658 (above200MA + weekly support) + TSLA lags SPY 5%+.
    Uptrend + structural support + TSLA-specific lag = bull market lag catch."""
    if sig_s658_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s664_s658_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S664: S658 (above200MA + weekly support) + ATR compression.
    Long-term uptrend + structural support + volatility squeeze = breakout incoming."""
    if sig_s658_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s665_s648_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S665: S648 (WR=94%, compressed + 10% disc + weekly) + above 200MA.
    Best-WR signal restricted to long-term uptrend: can we push to 97%+ WR?"""
    if sig_s648_s631_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if float(tsla['close'].iloc[i]) > ma200 else 0


def sig_s666_s652_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S666: S652 (100% WR: compressed+lag+disc) + above 200MA.
    Perfect WR signal restricted to long-term uptrend = maximum conviction."""
    if sig_s652_s648_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if float(tsla['close'].iloc[i]) > ma200 else 0


def sig_s667_s631_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S667: S631 (10% disc + weekly support) + NR7 (narrowest range in 7 days).
    Peak discount at support + tightest compression = pre-expansion coiling."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _nr7(tsla, i) else 0


def sig_s668_s629_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S668: S629 (15% disc + weekly support, 9/9yr) + ATR compression.
    Best-coverage signal + compression = S627 variant at 15% disc threshold."""
    if sig_s629_s215_52wk_disc15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s669_s526_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S669: S526 (MACD+VIX_cooldown, $1.08M, 92% WR) + 10% below 52wk.
    Slight relaxation of S659 (15% disc) — more trades at same quality?"""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.10) else 0


def sig_s670_s648_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S670: S648 (WR=94% compressed+disc) + VIX pct>70 (highest fear).
    Near-perfect WR base + extreme fear = maximally distressed but highest-quality entry."""
    if sig_s648_s631_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


SIGNALS_P9Q: List[Tuple] = [
    # Phase 9Q — Precision extensions + uptrend + gap-down (S661-S670)
    ('S661_s631_gap_down',     sig_s661_s631_gap_down,     10, 0.20, 50),
    ('S662_s658_52wk_disc10',  sig_s662_s658_52wk_disc10,  10, 0.20, 50),
    ('S663_s658_lag5pct',      sig_s663_s658_lag5pct,      10, 0.20, 50),
    ('S664_s658_compressed',   sig_s664_s658_compressed,   10, 0.20, 50),
    ('S665_s648_above200ma',   sig_s665_s648_above200ma,   10, 0.20, 50),
    ('S666_s652_above200ma',   sig_s666_s652_above200ma,   10, 0.20, 50),
    ('S667_s631_nr7',          sig_s667_s631_nr7,          10, 0.20, 50),
    ('S668_s629_compressed',   sig_s668_s629_compressed,   10, 0.20, 50),
    ('S669_s526_52wk_disc10',  sig_s669_s526_52wk_disc10,  10, 0.20, 50),
    ('S670_s648_vix_pct70',    sig_s670_s648_vix_pct70,    10, 0.20, 50),
]

# ── Phase 9R — Inverse gap-down + consecutive-week decline + VIX term structure ─
# Lessons learned: gap-down = bad. No-gap (quiet) OR gap-up to support = better.
# New: multi-week consecutive decline (weekly chart pattern), 5%+ gap-down filter SKIP.

def _tsla_no_gap_down(tsla, i) -> bool:
    """Today did NOT gap down more than 1% from yesterday's close (quiet open)."""
    if i < 1:
        return True
    prev_close = float(tsla['close'].iloc[i - 1])
    today_open = float(tsla['open'].iloc[i]) if 'open' in tsla.columns else prev_close
    if prev_close <= 0:
        return True
    gap = (prev_close - today_open) / prev_close
    return gap < 0.01  # gap down less than 1% = "quiet" open


def sig_s671_s631_no_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S671: S631 (10% disc + weekly support) + no significant gap-down.
    Best signal excluding gap-down entries — tests if removing gap-down trades helps."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_no_gap_down(tsla, i) else 0


def sig_s672_s648_no_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S672: S648 (WR=94%, compressed+disc) + no significant gap-down.
    Highest WR signal without gap-down entries = further WR improvement."""
    if sig_s648_s631_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_no_gap_down(tsla, i) else 0


def sig_s673_s652_no_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S673: S652 (100% WR) + no significant gap-down.
    Perfect WR signal without gap-down — should maintain 100% WR."""
    if sig_s652_s648_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_no_gap_down(tsla, i) else 0


def sig_s674_s631_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S674: S631 (10% disc + weekly support) + VIX cooldown timing.
    Best coverage signal + fear receding = timing confirmation."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s675_s631_above200ma_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S675: S631 (10% disc + weekly) + above200MA + compressed (3-way filter).
    Long-term uptrend + discount from peak + volatility squeeze = maximum conviction."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 200:
        ma200 = float(tsla['close'].iloc[i - 200:i].mean())
        if float(tsla['close'].iloc[i]) <= ma200:
            return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s676_s631_lag5pct_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S676: S631 (10% disc + weekly) + TSLA lag5pct + ATR compression (3-way).
    Peak discount + relative lag + volatility squeeze = three independent dimensions."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s677_s631_vix_pct60_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S677: S631 (10% disc + weekly) + VIX pct>60 + ATR compression (3-way).
    Fear regime + peak discount at support + volatility squeeze = max setup alignment."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _vix_elevated_pct(vix, i, window=252, pct=0.60):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s678_s631_rsi_low_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S678: S631 (10% disc + weekly) + RSI<40 + ATR compression (3-way).
    Peak discount at support + RSI depth + volatility squeeze = three-way oversold."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    if rsi_val >= 40.0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s679_s648_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S679: S648 (WR=94%) + volume dry-up.
    94% WR + sellers exhausted = the highest-purity entry combination."""
    if sig_s648_s631_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s680_s631_vol_dryup_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S680: S631 (10% disc + weekly) + vol_dryup + compressed (3-way timing).
    Best base signal + both timing confirmers: sellers gone AND volatility quiet."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _volume_below_avg(tsla, i):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


SIGNALS_P9R: List[Tuple] = [
    # Phase 9R — No-gap filter + 3-way combos + timing + S680 (S671-S680)
    ('S671_s631_no_gap_down',          sig_s671_s631_no_gap_down,          10, 0.20, 50),
    ('S672_s648_no_gap_down',          sig_s672_s648_no_gap_down,          10, 0.20, 50),
    ('S673_s652_no_gap_down',          sig_s673_s652_no_gap_down,          10, 0.20, 50),
    ('S674_s631_vix_cooldown',         sig_s674_s631_vix_cooldown,         10, 0.20, 50),
    ('S675_s631_above200ma_compressed',sig_s675_s631_above200ma_compressed,10, 0.20, 50),
    ('S676_s631_lag5pct_compressed',   sig_s676_s631_lag5pct_compressed,   10, 0.20, 50),
    ('S677_s631_vix60_compressed',     sig_s677_s631_vix_pct60_compressed, 10, 0.20, 50),
    ('S678_s631_rsi40_compressed',     sig_s678_s631_rsi_low_compressed,   10, 0.20, 50),
    ('S679_s648_vol_dryup',            sig_s679_s648_vol_dryup,            10, 0.20, 50),
    ('S680_s631_vol_dryup_compressed', sig_s680_s631_vol_dryup_compressed, 10, 0.20, 50),
]

# ── Phase 9S — RSI divergence, DOW, inside bar, multi-vol patterns (S681-S690) ─
# Explore day-of-week effects, RSI divergence vs SPY, and alternative combinations.

def _tsla_rsi_below_spy_rsi(rt, rs, i, spread: float = 15.0) -> bool:
    """TSLA RSI is at least spread points below SPY RSI — TSLA-specific oversold vs market."""
    if i >= len(rt) or i >= len(rs):
        return False
    tsla_rsi = float(rt.iloc[i])
    spy_rsi = float(rs.iloc[i])
    return (spy_rsi - tsla_rsi) >= spread


def _is_monday(tsla, i) -> bool:
    """Signal fires on a Monday (weekly fresh-start buying pressure)."""
    try:
        return tsla.index[i].weekday() == 0
    except Exception:
        return False


def _is_friday(tsla, i) -> bool:
    """Signal fires on a Friday (position squaring before weekend)."""
    try:
        return tsla.index[i].weekday() == 4
    except Exception:
        return False


def sig_s681_s631_rsi_divergence(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S681: S631 (10% disc + weekly support) + TSLA RSI ≥15 pts below SPY RSI.
    TSLA-specific oversold vs market: TSLA falling while SPY holds = catch-up."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_rsi_below_spy_rsi(rt, rs, i, spread=15.0) else 0


def sig_s682_s631_monday(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S682: S631 (10% disc + weekly support) + Monday entry.
    Weekly fresh-start effect: Monday at structural support = highest weekly urgency."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _is_monday(tsla, i) else 0


def sig_s683_s648_rsi_divergence(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S683: S648 (WR=94%, compressed+disc) + TSLA RSI ≥15pts below SPY RSI.
    Best WR base + specific TSLA-oversold-vs-market divergence."""
    if sig_s648_s631_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_rsi_below_spy_rsi(rt, rs, i, spread=15.0) else 0


def sig_s684_s631_inside_bar(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S684: S631 (10% disc + weekly support) + inside bar (range inside yesterday).
    Consolidation at peak discount at support = coiling before bounce."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _inside_bar(tsla, i) else 0


def sig_s685_s648_rsi_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S685: S648 (WR=94%, compressed+disc) + RSI < 40.
    Best-WR base + deep RSI oversold = multi-layer depth confirmation."""
    if sig_s648_s631_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 40.0 else 0


def sig_s686_s526_no_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S686: S526 (MACD+VIX_cooldown, $1.08M, 92% WR) + no gap down.
    VIX cooling + MACD turning without gap-down panic = controlled recovery."""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_no_gap_down(tsla, i) else 0


def sig_s687_s631_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S687: S631 (10% disc + weekly support) + MACD histogram turning up.
    Peak discount at support + momentum turning = momentum confirmation."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


def sig_s688_s631_vix_pct50(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S688: S631 (10% disc + weekly support) + VIX pct>50 (any elevated fear).
    More trades than pct60: captures any above-median fear with 10%_disc at support."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.50) else 0


def sig_s689_s658_rsi_divergence(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S689: S658 (above200MA + weekly support, $1.32M, WR=89%) + TSLA RSI divergence.
    Long-term uptrend + structural support + TSLA-specific RSI weakness."""
    if sig_s658_s215_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_rsi_below_spy_rsi(rt, rs, i, spread=15.0) else 0


def sig_s690_s631_consec_down3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S690: S631 (10% disc + weekly support) + 3+ consecutive down days.
    Extended red streak at peak discount at support = momentum exhaustion setup."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=3) else 0


SIGNALS_P9S: List[Tuple] = [
    # Phase 9S — RSI divergence + DOW + inside bar + MACD (S681-S690)
    ('S681_s631_rsi_divergence',  sig_s681_s631_rsi_divergence,  10, 0.20, 50),
    ('S682_s631_monday',          sig_s682_s631_monday,          10, 0.20, 50),
    ('S683_s648_rsi_divergence',  sig_s683_s648_rsi_divergence,  10, 0.20, 50),
    ('S684_s631_inside_bar',      sig_s684_s631_inside_bar,      10, 0.20, 50),
    ('S685_s648_rsi_low',         sig_s685_s648_rsi_low,         10, 0.20, 50),
    ('S686_s526_no_gap_down',     sig_s686_s526_no_gap_down,     10, 0.20, 50),
    ('S687_s631_macd_up',         sig_s687_s631_macd_up,         10, 0.20, 50),
    ('S688_s631_vix_pct50',       sig_s688_s631_vix_pct50,       10, 0.20, 50),
    ('S689_s658_rsi_divergence',  sig_s689_s658_rsi_divergence,  10, 0.20, 50),
    ('S690_s631_consec_down3',    sig_s690_s631_consec_down3,    10, 0.20, 50),
]

# ── Phase 9T — S687 extensions + S700 milestone (extreme multi-filter) ────────
# S687 (MACD up + 10%disc + weekly, $1.46M, 8/9yr) is our newest strong signal.
# Extend with top proven filters. S700 = maximum multi-filter alignment.

def sig_s691_s687_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S691: S687 (MACD up + 10%disc + weekly, 8/9yr) + VIX pct>60.
    MACD momentum + peak discount + support + fear regime = 4-way alignment."""
    if sig_s687_s631_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s692_s687_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S692: S687 (MACD up + 10%disc + weekly) + ATR compression.
    MACD turning up while volatility still compressed = early-mover setup."""
    if sig_s687_s631_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s693_s687_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S693: S687 (MACD up + 10%disc + weekly) + TSLA lags SPY 5%+.
    MACD turning + peak discount + TSLA underperformance = timing + depth + specificity."""
    if sig_s687_s631_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s694_s687_rsi_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S694: S687 (MACD up + 10%disc + weekly) + RSI < 40.
    MACD turning up from RSI oversold territory = strongest momentum reversal."""
    if sig_s687_s631_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 40.0 else 0


def sig_s695_s687_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S695: S687 (MACD up + 10%disc + weekly) + volume dry-up.
    MACD turning + peak discount at support + sellers exhausted = ideal entry."""
    if sig_s687_s631_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s696_s631_rsi30(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S696: S631 (10%disc + weekly support) + RSI < 30 (extreme oversold).
    Peak discount at structural support with RSI in capitulation territory."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 30.0 else 0


def sig_s697_s526_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S697: S526 (MACD+VIX_cooldown, 92% WR) + ATR compression.
    VIX fear receding + MACD turning + volatility squeeze = maximum timing precision."""
    if sig_s526_s333_vix_cooldown(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s698_s687_above200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S698: S687 (MACD up + 10%disc + weekly) + above 200d MA.
    Best new signal restricted to long-term uptrend: MACD momentum in bull market."""
    if sig_s687_s631_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if float(tsla['close'].iloc[i]) > ma200 else 0


def sig_s699_s687_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S699: S687 (MACD up + 10%disc + weekly) + VIX pct>70 (high fear tier).
    MACD timing in extreme fear environment at peak discount = maximum opportunity."""
    if sig_s687_s631_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s700_milestone_max_alignment(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S700 MILESTONE: Maximum signal alignment — all top filters at once.
    Weekly support + 10%_52wk + MACD up + VIX pct>60 + lag5pct.
    Every dimension must agree: structural, momentum, fear, and relative value."""
    if sig_s687_s631_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _vix_elevated_pct(vix, i, window=252, pct=0.60):
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


SIGNALS_P9T: List[Tuple] = [
    # Phase 9T — S687 extensions + S700 milestone (S691-S700)
    ('S691_s687_vix_pct60',           sig_s691_s687_vix_pct60,           10, 0.20, 50),
    ('S692_s687_compressed',          sig_s692_s687_compressed,          10, 0.20, 50),
    ('S693_s687_lag5pct',             sig_s693_s687_lag5pct,             10, 0.20, 50),
    ('S694_s687_rsi_low',             sig_s694_s687_rsi_low,             10, 0.20, 50),
    ('S695_s687_vol_dryup',           sig_s695_s687_vol_dryup,           10, 0.20, 50),
    ('S696_s631_rsi30',               sig_s696_s631_rsi30,               10, 0.20, 50),
    ('S697_s526_compressed',          sig_s697_s526_compressed,          10, 0.20, 50),
    ('S698_s687_above200ma',          sig_s698_s687_above200ma,          10, 0.20, 50),
    ('S699_s687_vix_pct70',           sig_s699_s687_vix_pct70,           10, 0.20, 50),
    ('S700_milestone_max_alignment',  sig_s700_milestone_max_alignment,  10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T)


# ── Phase 9U — S631+VIX (no MACD), S691 precision, S641+VIX (S701-S710) ──────
# Key question: does removing MACD from S691 → S631+VIX_pct60 give MORE coverage
# at similar quality? Also: do S691 precision extensions push WR toward 100%?

def sig_s701_s631_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S701: S631 (10%disc + weekly support) + VIX pct>60.
    Direct: skip MACD filter, just combine peak discount + structural support + fear regime.
    Should be bigger than S691 since MACD removed."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s702_s631_vix_pct50(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S702: S631 (10%disc + weekly support) + VIX pct>50 (moderate fear).
    Wider fear filter — captures more trades at peak discount in elevated-VIX market."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.50) else 0


def sig_s703_s691_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S703: S691 (S687+VIX_pct60, WR=93%) + TSLA lags SPY 5%+.
    Add relative underperformance to already-high-quality S691: precision push."""
    if sig_s691_s687_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s704_s691_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S704: S691 (S687+VIX_pct60, WR=93%) + volume dry-up.
    Volume dry-up has produced 100% WR in every extension — test on S691."""
    if sig_s691_s687_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s705_s691_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S705: S691 (S687+VIX_pct60, WR=93%) + ATR compression.
    VIX fear + MACD turning + peak discount + compressed vol = perfect entry."""
    if sig_s691_s687_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s706_s701_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S706: S701 (S631+VIX_pct60, no MACD) + TSLA lags SPY 5%+.
    Triple filter: peak discount + fear + relative underperformance (no MACD)."""
    if sig_s701_s631_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s707_s701_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S707: S701 (S631+VIX_pct60, no MACD) + ATR compression.
    Peak discount + structural support + VIX fear + vol squeeze."""
    if sig_s701_s631_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s708_s700_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S708: S700 milestone (5-way alignment) + volume dry-up.
    Already WR=92% → add vol exhaustion → expect near 100% WR."""
    if sig_s700_milestone_max_alignment(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s709_s642_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S709: S642 (5%disc + weekly support, $1.77M IS) + VIX pct>60.
    Apply VIX fear filter to our highest-coverage IS signal. Loose discount = more trades."""
    if sig_s642_s215_52wk_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s710_s642_macd_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S710: S642 (5%disc + weekly) + MACD turning up + VIX pct>50.
    Maximum coverage variant: loose discount + MACD momentum + moderate fear."""
    if sig_s642_s215_52wk_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    if hist_now <= hist_prev:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.50) else 0


SIGNALS_P9U: List[Tuple] = [
    # Phase 9U — S631/S641+VIX (no MACD), S691 precision, S700 extensions (S701-S710)
    ('S701_s631_vix_pct60',      sig_s701_s631_vix_pct60,      10, 0.20, 50),
    ('S702_s631_vix_pct50',      sig_s702_s631_vix_pct50,      10, 0.20, 50),
    ('S703_s691_lag5pct',        sig_s703_s691_lag5pct,        10, 0.20, 50),
    ('S704_s691_vol_dryup',      sig_s704_s691_vol_dryup,      10, 0.20, 50),
    ('S705_s691_compressed',     sig_s705_s691_compressed,     10, 0.20, 50),
    ('S706_s701_lag5pct',        sig_s706_s701_lag5pct,        10, 0.20, 50),
    ('S707_s701_compressed',     sig_s707_s701_compressed,     10, 0.20, 50),
    ('S708_s700_vol_dryup',      sig_s708_s700_vol_dryup,      10, 0.20, 50),
    ('S709_s642_vix_pct60',      sig_s709_s642_vix_pct60,      10, 0.20, 50),
    ('S710_s642_macd_vix',       sig_s710_s642_macd_vix,       10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T + SIGNALS_P9U)


# ── Phase 9V — Fresh dimensions: NR4, consec_down, false-break, RSI tiers, capitulation (S711-S720) ──
# New price-action dimensions not yet tested at S631 level.
# Goal: find another independent dimension that complements 52wk discount + weekly support.

def _tsla_nr4(tsla, i) -> bool:
    """True if today's range is the narrowest of the last 4 days (NR4 setup)."""
    if i < 4:
        return False
    today_range = float(tsla['high'].iloc[i]) - float(tsla['low'].iloc[i])
    for j in range(i - 3, i):
        prior_range = float(tsla['high'].iloc[j]) - float(tsla['low'].iloc[j])
        if today_range >= prior_range:
            return False
    return True


def _tsla_consec_down(tsla, i, n: int = 2) -> bool:
    """True if TSLA closed down for n consecutive days ending at i-1 (entering today)."""
    if i < n + 1:
        return False
    for j in range(i - n, i):
        if float(tsla['close'].iloc[j]) >= float(tsla['close'].iloc[j - 1]):
            return False
    return True


def _tsla_below_prior_week_low(tsla, tw, i) -> bool:
    """True if today's close is below last week's weekly low (false breakdown setup)."""
    if i < 7 or len(tw) < 2:
        return False
    # Find today's date and get prior week bar
    today = tsla.index[i]
    # Get weekly bars up to (not including) current week
    prior_week_bars = tw[tw.index < today]
    if len(prior_week_bars) < 1:
        return False
    prior_week_low = float(prior_week_bars.iloc[-1]['low'])
    return float(tsla['close'].iloc[i]) < prior_week_low


def _tsla_high_volume(tsla, i, multiplier: float = 1.5) -> bool:
    """True if today's volume is at least multiplier× the 20-day average (capitulation surge)."""
    if i < 20:
        return False
    avg_vol = float(tsla['volume'].iloc[i - 20:i].mean())
    if avg_vol <= 0:
        return False
    return float(tsla['volume'].iloc[i]) >= multiplier * avg_vol


def sig_s711_s631_rsi35(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S711: S631 (10%disc + weekly support) + RSI < 35 (deep oversold).
    Tighter than S685 (RSI<40) — only the most oversold peak-discount entries."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 35.0 else 0


def sig_s712_s631_consec_down2(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S712: S631 (10%disc + weekly support) + 2 consecutive down days.
    Back-to-back selling at weekly support with peak discount = exhaustion signal."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=2) else 0


def sig_s713_s701_rsi40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S713: S701 (S631+VIX_pct60, $1.4M) + RSI < 40.
    VIX fear regime + peak discount + RSI oversold = 3-way oversold alignment."""
    if sig_s701_s631_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 40.0 else 0


def sig_s714_s631_nr4(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S714: S631 (10%disc + weekly support) + NR4 (narrowest range in 4 days).
    Volatility squeeze at peak discount level — energy coiling before bounce."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_nr4(tsla, i) else 0


def sig_s715_s631_high_vol(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S715: S631 (10%disc + weekly support) + high volume (1.5× avg capitulation).
    Capitulation surge at peak discount + structural support = exhaustion bottom."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_high_volume(tsla, i, multiplier=1.5) else 0


def sig_s716_s631_consec_down3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S716: S631 (10%disc + weekly support) + 3 consecutive down days.
    Three days of selling at peak discount support = momentum exhaustion."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=3) else 0


def sig_s717_s701_consec_down2(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S717: S701 (S631+VIX_pct60) + 2 consecutive down days.
    VIX fear + peak discount + back-to-back selling = maximum buy-the-dip setup."""
    if sig_s701_s631_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=2) else 0


def sig_s718_s631_high_vol2x(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S718: S631 (10%disc + weekly support) + high volume (2× avg, extreme capitulation).
    Even stronger volume surge — panic-level selling at structural support."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_high_volume(tsla, i, multiplier=2.0) else 0


def sig_s719_s691_rsi35(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S719: S691 (S687+VIX_pct60, WR=93%) + RSI < 35.
    4-way alignment: VIX fear + MACD turning + peak discount + RSI deep oversold."""
    if sig_s691_s687_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 35.0 else 0


def sig_s720_s631_below_pw_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S720: S631 (10%disc + weekly support) + below prior week's low (false breakdown).
    TSLA undercuts prior week low at peak discount = stop-run exhaustion setup."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_prior_week_low(tsla, tw, i) else 0


SIGNALS_P9V: List[Tuple] = [
    # Phase 9V — NR4, consec_down, high-vol, RSI tiers, false-break (S711-S720)
    ('S711_s631_rsi35',          sig_s711_s631_rsi35,          10, 0.20, 50),
    ('S712_s631_consec_down2',   sig_s712_s631_consec_down2,   10, 0.20, 50),
    ('S713_s701_rsi40',          sig_s713_s701_rsi40,          10, 0.20, 50),
    ('S714_s631_nr4',            sig_s714_s631_nr4,            10, 0.20, 50),
    ('S715_s631_high_vol',       sig_s715_s631_high_vol,       10, 0.20, 50),
    ('S716_s631_consec_down3',   sig_s716_s631_consec_down3,   10, 0.20, 50),
    ('S717_s701_consec_down2',   sig_s717_s701_consec_down2,   10, 0.20, 50),
    ('S718_s631_high_vol2x',     sig_s718_s631_high_vol2x,     10, 0.20, 50),
    ('S719_s691_rsi35',          sig_s719_s691_rsi35,          10, 0.20, 50),
    ('S720_s631_below_pw_low',   sig_s720_s631_below_pw_low,   10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T + SIGNALS_P9U
           + SIGNALS_P9V)


# ── Phase 9W — NR4 extensions + hammer candle + 20d discount (S721-S730) ──────
# Best from 9V: NR4 ($809K) and below_pw_low ($762K).
# Extend NR4 with proven filters. Also test bullish-hammer candle + 20-day discount.

def _tsla_bullish_hammer(tsla, i, body_ratio: float = 0.30) -> bool:
    """True if today is a bullish hammer: small body in upper portion, long lower shadow.
    Lower shadow >= 2× body, body in top 30% of bar range. Sign of rejection at lows."""
    if i < 1:
        return False
    o = float(tsla['open'].iloc[i])
    h = float(tsla['high'].iloc[i])
    lo = float(tsla['low'].iloc[i])
    c = float(tsla['close'].iloc[i])
    bar_range = h - lo
    if bar_range <= 0:
        return False
    body = abs(c - o)
    lower_shadow = min(o, c) - lo
    # Bullish: close >= open, body small, lower shadow >= 2× body
    if c < o:
        return False
    if body > body_ratio * bar_range:
        return False
    if lower_shadow < 2.0 * body and lower_shadow < 0.5 * bar_range:
        return False
    return True


def _tsla_below_20d_high_pct(tsla, i, pct: float = 0.08) -> bool:
    """True if TSLA is at least pct% below its 20-day high (recent momentum decline)."""
    if i < 20:
        return False
    high_20d = float(tsla['high'].iloc[i - 20:i].max())
    if high_20d <= 0:
        return False
    discount = (high_20d - float(tsla['close'].iloc[i])) / high_20d
    return discount >= pct


def sig_s721_s714_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S721: S714 (S631+NR4, $809K) + VIX pct>60.
    NR4 squeeze at peak discount + fear regime = maximum energy coil."""
    if sig_s714_s631_nr4(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s722_s714_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S722: S714 (S631+NR4, $809K) + MACD histogram turning up.
    NR4 energy coil at peak discount with momentum turning = precision reversal."""
    if sig_s714_s631_nr4(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


def sig_s723_s720_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S723: S720 (S631+below_pw_low, $762K) + VIX pct>60.
    False-breakdown below prior week's low during fear regime = high-conviction stop-run."""
    if sig_s720_s631_below_pw_low(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s724_s631_hammer(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S724: S631 (10%disc + weekly support) + bullish hammer candle.
    Hammer rejection candle at peak discount structural support = classic reversal."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_bullish_hammer(tsla, i) else 0


def sig_s725_s631_20d_disc8(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S725: S631 (10%disc + weekly support) + 8%+ below 20-day high.
    Short-term momentum drop onto structural support with peak discount = fresh selling."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_20d_high_pct(tsla, i, pct=0.08) else 0


def sig_s726_s215_20d_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S726: Weekly channel support + 10%+ below 20-day high (standalone, no 52wk).
    Does 20-day high discount add value without the 52-week anchor?"""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_20d_high_pct(tsla, i, pct=0.10) else 0


def sig_s727_s714_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S727: S714 (S631+NR4) + ATR compression.
    NR4 + compressed ATR = double-confirmed volatility squeeze at peak discount."""
    if sig_s714_s631_nr4(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s728_s701_nr4(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S728: S701 (S631+VIX_pct60, $1.4M) + NR4 squeeze.
    VIX fear + peak discount + weekly support + NR4 = 4-way energy compression."""
    if sig_s701_s631_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_nr4(tsla, i) else 0


def sig_s729_s631_20d_disc12(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S729: S631 (10%disc + weekly support) + 12%+ below 20-day high.
    Sharper short-term drop at peak discount — more urgent selling exhaustion."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_20d_high_pct(tsla, i, pct=0.12) else 0


def sig_s730_s701_hammer(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S730: S701 (S631+VIX_pct60, $1.4M) + bullish hammer candle.
    Hammer rejection in fear regime at peak discount = high conviction reversal day."""
    if sig_s701_s631_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_bullish_hammer(tsla, i) else 0


SIGNALS_P9W: List[Tuple] = [
    # Phase 9W — NR4 extensions + hammer candle + 20d discount (S721-S730)
    ('S721_s714_vix_pct60',      sig_s721_s714_vix_pct60,      10, 0.20, 50),
    ('S722_s714_macd_up',        sig_s722_s714_macd_up,        10, 0.20, 50),
    ('S723_s720_vix_pct60',      sig_s723_s720_vix_pct60,      10, 0.20, 50),
    ('S724_s631_hammer',         sig_s724_s631_hammer,         10, 0.20, 50),
    ('S725_s631_20d_disc8',      sig_s725_s631_20d_disc8,      10, 0.20, 50),
    ('S726_s215_20d_disc10',     sig_s726_s215_20d_disc10,     10, 0.20, 50),
    ('S727_s714_compressed',     sig_s727_s714_compressed,     10, 0.20, 50),
    ('S728_s701_nr4',            sig_s728_s701_nr4,            10, 0.20, 50),
    ('S729_s631_20d_disc12',     sig_s729_s631_20d_disc12,     10, 0.20, 50),
    ('S730_s701_hammer',         sig_s730_s701_hammer,         10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T + SIGNALS_P9U
           + SIGNALS_P9V + SIGNALS_P9W)


# ── Phase 9X — S726 extensions: threshold sweep + precision filters (S731-S740) ──
# S726 (weekly + 10% below 20d high, $1.68M, 9/9yr) = strong new dimension.
# Extend: threshold sweep (5-15%), VIX filter, MACD, compression, vol_dryup.

def sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S731: Weekly channel support + 5%+ below 20-day high (loose threshold).
    More trades at cost of some quality — test the coverage vs precision tradeoff."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_20d_high_pct(tsla, i, pct=0.05) else 0


def sig_s732_s215_20d_disc7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S732: Weekly channel support + 7%+ below 20-day high."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_20d_high_pct(tsla, i, pct=0.07) else 0


def sig_s733_s215_20d_disc15(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S733: Weekly channel support + 15%+ below 20-day high (tight threshold).
    Only deep sharp drops onto weekly support — high-urgency exhaustion."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_20d_high_pct(tsla, i, pct=0.15) else 0


def sig_s734_s726_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S734: S726 (weekly + 10% below 20d high) + VIX pct>60.
    Apply fear-regime filter to new strong signal — expect WR boost."""
    if sig_s726_s215_20d_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s735_s726_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S735: S726 (weekly + 10% below 20d high) + MACD turning up.
    Momentum starting to reverse after sharp 10%+ drop at weekly support."""
    if sig_s726_s215_20d_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


def sig_s736_s726_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S736: S726 (weekly + 10% below 20d high) + ATR compression.
    Volatility squeeze after sharp drop at support — classic energy-coiling setup."""
    if sig_s726_s215_20d_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s737_s726_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S737: S726 (weekly + 10% below 20d high) + volume dry-up.
    After sharp 10% drop at support, sellers exhausted (low vol) = best entry."""
    if sig_s726_s215_20d_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s738_s726_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S738: S726 (weekly + 10% below 20d high) + TSLA lags SPY 5%+.
    Sharp drop at support where TSLA is underperforming SPY = stock-specific capitulation."""
    if sig_s726_s215_20d_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s739_s726_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S739: S726 (weekly + 10% below 20d high) + 52-week high discount 10%+.
    Combine both discount anchors: short-term (20d) and long-term (52wk) alignment."""
    if sig_s726_s215_20d_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.10) else 0


def sig_s740_s726_vix_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S740: S726 (weekly + 10% below 20d high) + VIX pct>50 + MACD turning up.
    Triple alignment: short-term drop + fear regime + momentum reversal."""
    if sig_s726_s215_20d_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _vix_elevated_pct(vix, i, window=252, pct=0.50):
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


SIGNALS_P9X: List[Tuple] = [
    # Phase 9X — S726 extensions: threshold sweep + precision filters (S731-S740)
    ('S731_s215_20d_disc5',      sig_s731_s215_20d_disc5,      10, 0.20, 50),
    ('S732_s215_20d_disc7',      sig_s732_s215_20d_disc7,      10, 0.20, 50),
    ('S733_s215_20d_disc15',     sig_s733_s215_20d_disc15,     10, 0.20, 50),
    ('S734_s726_vix_pct60',      sig_s734_s726_vix_pct60,      10, 0.20, 50),
    ('S735_s726_macd_up',        sig_s735_s726_macd_up,        10, 0.20, 50),
    ('S736_s726_compressed',     sig_s736_s726_compressed,     10, 0.20, 50),
    ('S737_s726_vol_dryup',      sig_s737_s726_vol_dryup,      10, 0.20, 50),
    ('S738_s726_lag5pct',        sig_s738_s726_lag5pct,        10, 0.20, 50),
    ('S739_s726_52wk_disc10',    sig_s739_s726_52wk_disc10,    10, 0.20, 50),
    ('S740_s726_vix_macd',       sig_s740_s726_vix_macd,       10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T + SIGNALS_P9U
           + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X)


# ── Phase 9Y — S731 extensions + SPY-aligned bounce + 100d MA context (S741-S750) ──
# S731 (weekly + 5% below 20d high, $1.93M IS+2025) = strongest new signal.
# Extend with precision filters. Also test SPY weekly support alignment.

def _spy_below_20d_high_pct(spy, i, pct: float = 0.05) -> bool:
    """True if SPY is at least pct% below its 20-day high (market also pulling back)."""
    if i < 20:
        return False
    high_20d = float(spy['high'].iloc[i - 20:i].max())
    if high_20d <= 0:
        return False
    discount = (high_20d - float(spy['close'].iloc[i])) / high_20d
    return discount >= pct


def _tsla_above_100ma(tsla, i) -> bool:
    """True if TSLA close is above its 100-day SMA (medium-term uptrend context)."""
    if i < 100:
        return True  # Not enough data, pass through
    ma100 = float(tsla['close'].iloc[i - 100:i].mean())
    return float(tsla['close'].iloc[i]) > ma100


def sig_s741_s731_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S741: S731 (weekly + 5% below 20d high) + ATR compression.
    Best new signal + volatility squeeze = high-precision entry at 5% 20d discount."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s742_s731_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S742: S731 (weekly + 5% below 20d high) + volume dry-up.
    Expect 100% WR: sellers exhausted at 5% discount from recent high at weekly support."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s743_s731_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S743: S731 (weekly + 5% below 20d high) + MACD turning up.
    Momentum reversal confirmation after 5% drop at weekly support."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


def sig_s744_s731_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S744: S731 (weekly + 5% below 20d high) + TSLA lags SPY 5%+.
    Stock-specific selling: TSLA dropped 5% from 20d high AND underperformed SPY."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s745_s731_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S745: S731 (weekly + 5% below 20d high) + VIX pct>60.
    Does VIX fear filter help S731 (recall VIX hurt S726, but S731 has looser threshold)?"""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s746_s731_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S746: S731 (weekly + 5% below 20d high) + 52-week high discount 10%+.
    Combine both discount anchors at their optimal thresholds: 5%/20d + 10%/52wk."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.10) else 0


def sig_s747_s731_rsi40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S747: S731 (weekly + 5% below 20d high) + RSI < 40.
    RSI oversold at 5% 20d discount + weekly support — three-way oversold."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 40.0 else 0


def sig_s748_s731_spy_also_weak(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S748: S731 (weekly + 5% below 20d high) + SPY also 5%+ below its 20d high.
    Market-wide pullback amplifies TSLA at-support bounce probability."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _spy_below_20d_high_pct(spy, i, pct=0.05) else 0


def sig_s749_s731_above100ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S749: S731 (weekly + 5% below 20d high) + above 100d MA (bull-market context).
    5% pullback in longer-term uptrend at weekly support = higher-quality bounce."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_above_100ma(tsla, i) else 0


def sig_s750_s631_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S750: S631 (10%disc 52wk + weekly) + 5% below 20d high.
    Add 20d momentum dimension to our best 52wk signal: both discount anchors aligned."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_20d_high_pct(tsla, i, pct=0.05) else 0


SIGNALS_P9Y: List[Tuple] = [
    # Phase 9Y — S731 extensions + SPY alignment + 100d MA + S631 with 20d (S741-S750)
    ('S741_s731_compressed',     sig_s741_s731_compressed,     10, 0.20, 50),
    ('S742_s731_vol_dryup',      sig_s742_s731_vol_dryup,      10, 0.20, 50),
    ('S743_s731_macd_up',        sig_s743_s731_macd_up,        10, 0.20, 50),
    ('S744_s731_lag5pct',        sig_s744_s731_lag5pct,        10, 0.20, 50),
    ('S745_s731_vix_pct60',      sig_s745_s731_vix_pct60,      10, 0.20, 50),
    ('S746_s731_52wk_disc10',    sig_s746_s731_52wk_disc10,    10, 0.20, 50),
    ('S747_s731_rsi40',          sig_s747_s731_rsi40,          10, 0.20, 50),
    ('S748_s731_spy_also_weak',  sig_s748_s731_spy_also_weak,  10, 0.20, 50),
    ('S749_s731_above100ma',     sig_s749_s731_above100ma,     10, 0.20, 50),
    ('S750_s631_20d_disc5',      sig_s750_s631_20d_disc5,      10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y)


# ── Phase 9Z — S743/S741 extensions + 5-day decline dimension (S751-S760) ──────
# S743 (MACD+20d+weekly, 9/9yr, $1.32M) and S741 (compressed, WR=93%, $1.22M) are
# new strong signals. Extend both. Also test 5-day decline as new short-term momentum.

def _tsla_below_5d_high_pct(tsla, i, pct: float = 0.05) -> bool:
    """True if TSLA is at least pct% below its 5-day high (very recent drop)."""
    if i < 5:
        return False
    high_5d = float(tsla['high'].iloc[i - 5:i].max())
    if high_5d <= 0:
        return False
    discount = (high_5d - float(tsla['close'].iloc[i])) / high_5d
    return discount >= pct


def sig_s751_s743_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S751: S743 (S731+MACD_up, 9/9yr) + ATR compression.
    MACD momentum reversal + 20d discount at support + volatility squeeze."""
    if sig_s743_s731_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s752_s743_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S752: S743 (S731+MACD_up, 9/9yr) + VIX pct>60.
    Fear regime + MACD turning + 20d discount at support = 4-way alignment."""
    if sig_s743_s731_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s753_s743_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S753: S743 (S731+MACD_up, 9/9yr) + TSLA lags SPY 5%+.
    MACD momentum reversal where TSLA has also been underperforming SPY."""
    if sig_s743_s731_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s754_s743_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S754: S743 (S731+MACD_up, 9/9yr) + volume dry-up.
    MACD turning up on declining volume = institutional accumulation at support."""
    if sig_s743_s731_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s755_s741_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S755: S741 (S731+compressed, WR=93%) + TSLA lags SPY 5%+.
    Compressed+20d_disc at support where TSLA underperformed SPY = high conviction."""
    if sig_s741_s731_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s756_s741_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S756: S741 (S731+compressed, WR=93%) + volume dry-up.
    Double exhaustion: ATR contracted AND volume dried up = maximum seller fatigue."""
    if sig_s741_s731_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s757_s215_5d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S757: Weekly channel support + 5%+ below 5-day high (very recent momentum).
    Short-term (5-day) drop onto weekly support — captures fresh momentum exhaustion."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_5d_high_pct(tsla, i, pct=0.05) else 0


def sig_s758_s215_5d_disc3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S758: Weekly channel support + 3%+ below 5-day high (very short-term decline).
    Even looser 5-day threshold — captures mild pullbacks to weekly support."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_5d_high_pct(tsla, i, pct=0.03) else 0


def sig_s759_s731_5d_disc3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S759: S731 (5% below 20d high) + 3%+ below 5-day high (both momentum anchors).
    Combine 20-day (medium) and 5-day (short) discount at weekly support."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_5d_high_pct(tsla, i, pct=0.03) else 0


def sig_s760_s743_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S760: S743 (S731+MACD_up, 9/9yr) + 52-week high discount 10%+.
    All three discount anchors: 20d + 52wk + MACD timing at weekly support."""
    if sig_s743_s731_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.10) else 0


SIGNALS_P9Z: List[Tuple] = [
    # Phase 9Z — S743/S741 extensions + 5-day decline dimension (S751-S760)
    ('S751_s743_compressed',     sig_s751_s743_compressed,     10, 0.20, 50),
    ('S752_s743_vix_pct60',      sig_s752_s743_vix_pct60,      10, 0.20, 50),
    ('S753_s743_lag5pct',        sig_s753_s743_lag5pct,        10, 0.20, 50),
    ('S754_s743_vol_dryup',      sig_s754_s743_vol_dryup,      10, 0.20, 50),
    ('S755_s741_lag5pct',        sig_s755_s741_lag5pct,        10, 0.20, 50),
    ('S756_s741_vol_dryup',      sig_s756_s741_vol_dryup,      10, 0.20, 50),
    ('S757_s215_5d_disc5',       sig_s757_s215_5d_disc5,       10, 0.20, 50),
    ('S758_s215_5d_disc3',       sig_s758_s215_5d_disc3,       10, 0.20, 50),
    ('S759_s731_5d_disc3',       sig_s759_s731_5d_disc3,       10, 0.20, 50),
    ('S760_s743_52wk_disc10',    sig_s760_s743_52wk_disc10,    10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z)


# ── Phase 10A — S757 extensions + near-52wk-low + S755 wider coverage (S761-S770) ──
# Extend 5-day discount signal S757. Test 52-week LOW proximity.
# Try to get S755 (100% WR) to cover 9 years vs 8.

def _tsla_near_52wk_low(tsla, i, pct: float = 0.10) -> bool:
    """True if TSLA close is within pct% of its 52-week (252-day) low.
    Near annual low at weekly support = maximum buy-the-dip scenario."""
    lookback = min(252, i)
    if lookback < 20:
        return False
    low_52wk = float(tsla['low'].iloc[i - lookback:i].min())
    if low_52wk <= 0:
        return False
    proximity = (float(tsla['close'].iloc[i]) - low_52wk) / low_52wk
    return proximity <= pct


def sig_s761_s757_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S761: S757 (weekly + 5% below 5d high) + ATR compression.
    Short-term 5d momentum drop at weekly support + volatility squeeze."""
    if sig_s757_s215_5d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s762_s757_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S762: S757 (weekly + 5% below 5d high) + MACD turning up.
    Very fresh momentum (5d drop) + MACD reversal at weekly support."""
    if sig_s757_s215_5d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


def sig_s763_s757_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S763: S757 (weekly + 5% below 5d high) + volume dry-up.
    Fresh 5d drop on fading volume at weekly support = clean exhaustion setup."""
    if sig_s757_s215_5d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s764_s215_near_52wk_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S764: Weekly channel support + within 10% of 52-week low.
    TSLA near annual low at structural support = maximum value + exhaustion setup."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_near_52wk_low(tsla, i, pct=0.10) else 0


def sig_s765_s631_near_52wk_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S765: S631 (10%disc + weekly) + within 10% of 52-week low.
    Peak discount from high AND near annual low at weekly support."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_near_52wk_low(tsla, i, pct=0.10) else 0


def sig_s766_s215_near_52wk_low5(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S766: Weekly channel support + within 5% of 52-week low (very close to annual low).
    TSLA essentially AT its annual low at structural support = deep capitulation."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_near_52wk_low(tsla, i, pct=0.05) else 0


def sig_s767_s741_rsi35(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S767: S741 (S731+compressed, WR=93%) + RSI < 35 (deep oversold).
    Compressed + 20d discount at support AND RSI in capitulation territory."""
    if sig_s741_s731_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi_val = float(rt.iloc[i]) if i < len(rt) else 50.0
    return 1 if rsi_val < 35.0 else 0


def sig_s768_s741_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S768: S741 (S731+compressed, WR=93%) + VIX pct>60.
    S741 is already 93% WR — adding VIX might push toward 100%."""
    if sig_s741_s731_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s769_s757_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S769: S757 (weekly + 5% below 5d high) + 52-week high discount 10%+.
    Short-term (5d) drop combined with long-term (52wk) discount at weekly support."""
    if sig_s757_s215_5d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.10) else 0


def sig_s770_s757_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S770: S757 (weekly + 5% below 5d high) + TSLA lags SPY 5%+.
    Very fresh TSLA drop at support where TSLA is underperforming SPY."""
    if sig_s757_s215_5d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


SIGNALS_P10A: List[Tuple] = [
    # Phase 10A — S757 extensions + near-52wk-low + precision push (S761-S770)
    ('S761_s757_compressed',     sig_s761_s757_compressed,     10, 0.20, 50),
    ('S762_s757_macd_up',        sig_s762_s757_macd_up,        10, 0.20, 50),
    ('S763_s757_vol_dryup',      sig_s763_s757_vol_dryup,      10, 0.20, 50),
    ('S764_s215_near_52wk_low',  sig_s764_s215_near_52wk_low,  10, 0.20, 50),
    ('S765_s631_near_52wk_low',  sig_s765_s631_near_52wk_low,  10, 0.20, 50),
    ('S766_s215_near_52wk_low5', sig_s766_s215_near_52wk_low5, 10, 0.20, 50),
    ('S767_s741_rsi35',          sig_s767_s741_rsi35,          10, 0.20, 50),
    ('S768_s741_vix_pct60',      sig_s768_s741_vix_pct60,      10, 0.20, 50),
    ('S769_s757_52wk_disc10',    sig_s769_s757_52wk_disc10,    10, 0.20, 50),
    ('S770_s757_lag5pct',        sig_s770_s757_lag5pct,        10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A)


# ── Phase 10B — BB compression + 3-day RSI + 50MA context + S778 max-precision (S771-S780) ──
# Explore: Bollinger Band width compression, 3-day RSI extremes, 50d MA context,
# S778 = S753+VIX (4-way MACD combo), and hold-time extended variants.

def _bb_width(tsla, i, period: int = 20, n_std: float = 2.0) -> float:
    """Return Bollinger Band width = (upper - lower) / mid. Lower = more compressed."""
    if i < period:
        return 0.10
    closes = tsla['close'].iloc[i - period:i]
    mid = float(closes.mean())
    std = float(closes.std())
    if mid <= 0:
        return 0.10
    upper = mid + n_std * std
    lower = mid - n_std * std
    return (upper - lower) / mid


def _tsla_bb_compressed(tsla, i, width_threshold: float = 0.08) -> bool:
    """True if Bollinger Band width is below threshold (very compressed)."""
    return _bb_width(tsla, i) < width_threshold


def _rsi_n_day(prices, i, period: int = 3) -> float:
    """Simple n-period RSI calculation."""
    if i < period + 1:
        return 50.0
    gains, losses = [], []
    for j in range(i - period, i):
        change = float(prices.iloc[j]) - float(prices.iloc[j - 1])
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-change)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _tsla_3d_rsi_oversold(tsla, i, threshold: float = 20.0) -> bool:
    """True if 3-day RSI < threshold (extreme short-term oversold)."""
    rsi_3 = _rsi_n_day(tsla['close'], i, period=3)
    return rsi_3 < threshold


def _tsla_above_50ma(tsla, i) -> bool:
    """True if TSLA close is above its 50-day SMA."""
    if i < 50:
        return True
    ma50 = float(tsla['close'].iloc[i - 50:i].mean())
    return float(tsla['close'].iloc[i]) > ma50


def sig_s771_s215_bb_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S771: Weekly channel support + Bollinger Band compression (BB width < 8%).
    BB squeeze at weekly support = energy coiling before bounce."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_bb_compressed(tsla, i) else 0


def sig_s772_s631_bb_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S772: S631 (10%disc + weekly) + Bollinger Band compression.
    Peak discount at support with BB squeeze — compare to S648 (ATR compressed)."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_bb_compressed(tsla, i) else 0


def sig_s773_s731_bb_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S773: S731 (weekly + 5% below 20d high) + Bollinger Band compression.
    Compare BB compressed vs ATR compressed (S741) at 20d discount support."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_bb_compressed(tsla, i) else 0


def sig_s774_s631_3d_rsi20(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S774: S631 (10%disc + weekly) + 3-day RSI < 20 (extreme short-term oversold).
    Three consecutive down days have driven 3d RSI to extreme = momentum exhaustion."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_3d_rsi_oversold(tsla, i, threshold=20.0) else 0


def sig_s775_s731_3d_rsi20(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S775: S731 (weekly + 5% below 20d high) + 3-day RSI < 20.
    20d high discount at weekly support with extreme 3d RSI oversold."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_3d_rsi_oversold(tsla, i, threshold=20.0) else 0


def sig_s776_s741_3d_rsi20(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S776: S741 (S731+compressed, WR=93%) + 3-day RSI < 20.
    Already best WR signal + extreme 3d RSI = push toward 100%."""
    if sig_s741_s731_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_3d_rsi_oversold(tsla, i, threshold=20.0) else 0


def sig_s777_s731_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S777: S731 (weekly + 5% below 20d high) + below 50-day MA.
    TSLA below medium-term trend at 20d discount + weekly support = bear pullback setup."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


def sig_s778_s753_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S778: S753 (S743+lag5pct) + VIX pct>60. 4-way: weekly+20d+MACD+lag+VIX.
    Maximum MACD combo: all momentum, discount, fear, and relative dimensions aligned."""
    if sig_s753_s743_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s779_s741_consec_down2(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S779: S741 (S731+compressed, WR=93%) + 2 consecutive down days.
    ATR compressed + 2 down days entering = selling into tight range = snap-back."""
    if sig_s741_s731_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=2) else 0


def sig_s780_s631_3d_rsi30(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S780: S631 (10%disc + weekly) + 3-day RSI < 30.
    Wider 3d RSI threshold (30 vs 20) for more trades at peak discount."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_3d_rsi_oversold(tsla, i, threshold=30.0) else 0


SIGNALS_P10B: List[Tuple] = [
    # Phase 10B — BB compression + 3-day RSI + 50MA context (S771-S780)
    ('S771_s215_bb_compressed',  sig_s771_s215_bb_compressed,  10, 0.20, 50),
    ('S772_s631_bb_compressed',  sig_s772_s631_bb_compressed,  10, 0.20, 50),
    ('S773_s731_bb_compressed',  sig_s773_s731_bb_compressed,  10, 0.20, 50),
    ('S774_s631_3d_rsi20',       sig_s774_s631_3d_rsi20,       10, 0.20, 50),
    ('S775_s731_3d_rsi20',       sig_s775_s731_3d_rsi20,       10, 0.20, 50),
    ('S776_s741_3d_rsi20',       sig_s776_s741_3d_rsi20,       10, 0.20, 50),
    ('S777_s731_below50ma',      sig_s777_s731_below50ma,      10, 0.20, 50),
    ('S778_s753_vix_pct60',      sig_s778_s753_vix_pct60,      10, 0.20, 50),
    ('S779_s741_consec_down2',   sig_s779_s741_consec_down2,   10, 0.20, 50),
    ('S780_s631_3d_rsi30',       sig_s780_s631_3d_rsi30,       10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B)


# ── Phase 10C — S777 extensions + BB recalibration + S631+below50MA (S781-S790) ──
# S777 (below50MA + 20d discount, $1.85M+2025) is a new top signal.
# Extend with precision filters. Recalibrate BB (try 15-20% width). Test S631+below50MA.

def sig_s781_s777_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S781: S777 (S731+below50MA) + ATR compression.
    Below 50MA at 20d discount + weekly support + volatility squeeze = high precision."""
    if sig_s777_s731_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s782_s777_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S782: S777 (S731+below50MA) + volume dry-up.
    Bear context + 20d discount + support + seller exhaustion = maximum setup."""
    if sig_s777_s731_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s783_s777_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S783: S777 (S731+below50MA) + MACD turning up.
    Below-50MA pullback to support where MACD just starting to reverse."""
    if sig_s777_s731_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


def sig_s784_s777_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S784: S777 (S731+below50MA) + TSLA lags SPY 5%+.
    Bear context + TSLA-specific underperformance: stock-specific bear at support."""
    if sig_s777_s731_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s785_s777_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S785: S777 (S731+below50MA) + VIX pct>60.
    Bear context + fear regime + 20d discount at weekly support."""
    if sig_s777_s731_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s786_s631_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S786: S631 (10%disc + weekly) + below 50-day MA.
    Apply below-50MA to our 52wk-discount anchor — complements S777."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


def sig_s787_s215_bb_wide(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S787: Weekly channel support + BB width < 20% (recalibrated, wider threshold).
    Previous BB threshold (8%) was too tight. Try 20% to find compressed-ish setups."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_bb_compressed(tsla, i, width_threshold=0.20) else 0


def sig_s788_s631_bb_wide(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S788: S631 (10%disc + weekly) + BB width < 20%.
    Recalibrated BB compression at peak discount level."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_bb_compressed(tsla, i, width_threshold=0.20) else 0


def sig_s789_s731_bb_wide(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S789: S731 (weekly + 5% below 20d high) + BB width < 20%.
    Recalibrated BB at 20d discount — compare to S741 (ATR compressed)."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_bb_compressed(tsla, i, width_threshold=0.20) else 0


def sig_s790_s777_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S790: S777 (S731+below50MA) + 52-week high discount 10%+.
    Below 50MA + 20d disc + 52wk disc + weekly support = all discount anchors."""
    if sig_s777_s731_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_below_52wk_pct(tsla, i, pct=0.10) else 0


SIGNALS_P10C: List[Tuple] = [
    # Phase 10C — S777 extensions + BB recalibration + S631+below50MA (S781-S790)
    ('S781_s777_compressed',     sig_s781_s777_compressed,     10, 0.20, 50),
    ('S782_s777_vol_dryup',      sig_s782_s777_vol_dryup,      10, 0.20, 50),
    ('S783_s777_macd_up',        sig_s783_s777_macd_up,        10, 0.20, 50),
    ('S784_s777_lag5pct',        sig_s784_s777_lag5pct,        10, 0.20, 50),
    ('S785_s777_vix_pct60',      sig_s785_s777_vix_pct60,      10, 0.20, 50),
    ('S786_s631_below50ma',      sig_s786_s631_below50ma,      10, 0.20, 50),
    ('S787_s215_bb_wide',        sig_s787_s215_bb_wide,        10, 0.20, 50),
    ('S788_s631_bb_wide',        sig_s788_s631_bb_wide,        10, 0.20, 50),
    ('S789_s731_bb_wide',        sig_s789_s731_bb_wide,        10, 0.20, 50),
    ('S790_s777_52wk_disc10',    sig_s790_s777_52wk_disc10,    10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C)


# ── Phase 10D — DOW sweep + annual underperformance + S786 extensions (S791-S800) ──
# Test Friday/Wednesday entries, annual TSLA vs SPY underperformance, S786 quality filters.
# S800 milestone: try to find the single best multi-filter signal combination.

def _is_friday(tsla, i) -> bool:
    """Signal fires on a Friday."""
    if i >= len(tsla):
        return False
    return tsla.index[i].weekday() == 4  # 4 = Friday


def _is_wednesday(tsla, i) -> bool:
    """Signal fires on a Wednesday."""
    if i >= len(tsla):
        return False
    return tsla.index[i].weekday() == 2  # 2 = Wednesday


def _tsla_annual_underperformer(tsla, spy, i, lookback: int = 252, lag: float = 0.10) -> bool:
    """True if TSLA underperformed SPY by at least lag over the last year (252 days).
    Annual underperformance = TSLA-specific bear cycle, not just a brief dip."""
    if i < lookback + 1:
        return False
    tsla_ret = float(tsla['close'].iloc[i]) / float(tsla['close'].iloc[i - lookback]) - 1.0
    spy_ret  = float(spy['close'].iloc[i])  / float(spy['close'].iloc[i - lookback])  - 1.0
    return (tsla_ret - spy_ret) <= -lag


def _recent_support_test(tsla, tw, i, lookback_days: int = 25) -> bool:
    """True if TSLA tested weekly channel support at least once more in the last lookback_days.
    Second test of a support level is often stronger (double-bottom setup)."""
    if i < lookback_days + 1 or len(tw) < 2:
        return False
    today = tsla.index[i]
    recent_start = tsla.index[max(0, i - lookback_days)]
    # Get weekly bars from lookback window
    recent_weekly = tw[(tw.index >= recent_start) & (tw.index < today)]
    if len(recent_weekly) < 1:
        return False
    # Get last 3 weekly channel (approximate: use min of recent weekly closes as support proxy)
    recent_lows = [float(recent_weekly.iloc[j]['low']) for j in range(len(recent_weekly))]
    support_level = min(recent_lows) * 1.03  # 3% above the recent low = support zone
    current_close = float(tsla['close'].iloc[i])
    return current_close <= support_level


def sig_s791_s631_friday(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S791: S631 (10%disc + weekly support) + entry on Friday.
    Best DOW day for week-end bounce entries at peak discount level."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _is_friday(tsla, i) else 0


def sig_s792_s731_friday(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S792: S731 (weekly + 5% below 20d high) + entry on Friday.
    Friday entries at 20d discount support — test DOW effect on this signal."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _is_friday(tsla, i) else 0


def sig_s793_s631_wednesday(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S793: S631 (10%disc + weekly support) + entry on Wednesday.
    Mid-week entries at peak discount — often best for short hold times."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _is_wednesday(tsla, i) else 0


def sig_s794_s631_annual_underperf(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S794: S631 (10%disc + weekly) + TSLA underperformed SPY by 10%+ over last year.
    Annual underperformance = TSLA-specific bear cycle bouncing from structural support."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_annual_underperformer(tsla, spy, i, lag=0.10) else 0


def sig_s795_s731_annual_underperf(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S795: S731 (weekly + 5% below 20d high) + TSLA underperformed SPY by 10%+ over 1yr.
    Annual underperformance + recent momentum drop at support = deep value setup."""
    if sig_s731_s215_20d_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_annual_underperformer(tsla, spy, i, lag=0.10) else 0


def sig_s796_s786_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S796: S786 (S631+below50MA, $1.87M+2025) + ATR compression.
    Best 52wk+below50MA signal + volatility squeeze — expect WR=93%."""
    if sig_s786_s631_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s797_s786_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S797: S786 (S631+below50MA, $1.87M+2025) + volume dry-up.
    Expect 100% WR: bear context + peak discount at support + sellers exhausted."""
    if sig_s786_s631_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s798_s786_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S798: S786 (S631+below50MA, $1.87M+2025) + MACD turning up.
    Bear phase bounce + peak 52wk discount at support + MACD reversing."""
    if sig_s786_s631_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


def sig_s799_s786_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S799: S786 (S631+below50MA, $1.87M+2025) + TSLA lags SPY 5%+.
    Below 50MA + peak discount + TSLA-specific underperformance = 4-way signal."""
    if sig_s786_s631_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s800_milestone_below50ma_max(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S800 MILESTONE: Maximum below-50MA alignment.
    S786 (52wk+weekly+below50MA) + ATR compressed + TSLA lags SPY 5%.
    All bear-phase dimensions aligned: structural, discount, trend, momentum, relative."""
    if sig_s796_s786_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


SIGNALS_P10D: List[Tuple] = [
    # Phase 10D — DOW sweep + annual underperf + S786 extensions + S800 milestone (S791-S800)
    ('S791_s631_friday',            sig_s791_s631_friday,            10, 0.20, 50),
    ('S792_s731_friday',            sig_s792_s731_friday,            10, 0.20, 50),
    ('S793_s631_wednesday',         sig_s793_s631_wednesday,         10, 0.20, 50),
    ('S794_s631_annual_underperf',  sig_s794_s631_annual_underperf,  10, 0.20, 50),
    ('S795_s731_annual_underperf',  sig_s795_s731_annual_underperf,  10, 0.20, 50),
    ('S796_s786_compressed',        sig_s796_s786_compressed,        10, 0.20, 50),
    ('S797_s786_vol_dryup',         sig_s797_s786_vol_dryup,         10, 0.20, 50),
    ('S798_s786_macd_up',           sig_s798_s786_macd_up,           10, 0.20, 50),
    ('S799_s786_lag5pct',           sig_s799_s786_lag5pct,           10, 0.20, 50),
    ('S800_milestone_below50ma_max',sig_s800_milestone_below50ma_max,10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D)


# ── Phase 10E — Union signals + S796/S741 extensions + max-coverage push (S801-S810) ──
# Key question: does OR-union of two strong discount anchors beat either alone?
# Also: can we push WR=93% signals past 8/8yr coverage?

def sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S801: Weekly channel support + (52wk disc10 OR 5% below 20d high) = UNION.
    Either peak discount OR recent momentum drop at structural support.
    Tests if combining two strong dimensions in OR logic beats AND logic."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    has_52wk = _tsla_below_52wk_pct(tsla, i, pct=0.10)
    has_20d  = _tsla_below_20d_high_pct(tsla, i, pct=0.05)
    return 1 if (has_52wk or has_20d) else 0


def sig_s802_s215_disc_and(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S802: Weekly channel support + (52wk disc10 AND 5% below 20d high) = INTERSECTION.
    Both discount anchors must be true — compare to S739 (same signal, already tested)."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    has_52wk = _tsla_below_52wk_pct(tsla, i, pct=0.10)
    has_20d  = _tsla_below_20d_high_pct(tsla, i, pct=0.05)
    return 1 if (has_52wk and has_20d) else 0


def sig_s803_s796_vix_pct50(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S803: S796 (S786+compressed, WR=93%, 8/8yr) + VIX pct>50.
    Can VIX filter push WR=93% signal toward 100% while maintaining 8yr coverage?"""
    if sig_s796_s786_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.50) else 0


def sig_s804_s741_vix_pct50(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S804: S741 (S731+compressed, WR=93%, 7/7yr) + VIX pct>50.
    VIX filter on WR=93% 20d-disc compressed signal — try to reach 100% WR."""
    if sig_s741_s731_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.50) else 0


def sig_s805_s801_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S805: S801 (OR-union of disc) + ATR compression.
    Wider discount coverage (OR logic) + volatility squeeze for quality control."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s806_s801_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S806: S801 (OR-union of disc) + volume dry-up.
    Maximum coverage union signal + seller exhaustion = 100% WR expected."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s807_s786_compressed_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S807: S796 (S786+compressed) + VIX pct>60.
    Can VIX_pct60 push WR from 93% to 100% at S786 level?"""
    if sig_s796_s786_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s808_s801_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S808: S801 (OR-union, wider coverage) + MACD turning up.
    Max-coverage signal with momentum timing — test OR logic value."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


def sig_s809_s801_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S809: S801 (OR-union of both discounts) + below 50d MA.
    Widest discount coverage + bear-phase filter."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


def sig_s810_milestone_union_max(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S810 MILESTONE: Union discount + compressed + below50MA + lag.
    OR(52wk_disc + 20d_disc) + ATR compressed + below 50MA + TSLA lags SPY.
    Maximum coverage precision: wider entry net + quality filters."""
    if sig_s805_s801_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


SIGNALS_P10E: List[Tuple] = [
    # Phase 10E — Union signals + S796/S741 extensions (S801-S810)
    ('S801_s215_disc_or',         sig_s801_s215_disc_or,         10, 0.20, 50),
    ('S802_s215_disc_and',        sig_s802_s215_disc_and,        10, 0.20, 50),
    ('S803_s796_vix_pct50',       sig_s803_s796_vix_pct50,       10, 0.20, 50),
    ('S804_s741_vix_pct50',       sig_s804_s741_vix_pct50,       10, 0.20, 50),
    ('S805_s801_compressed',      sig_s805_s801_compressed,      10, 0.20, 50),
    ('S806_s801_vol_dryup',       sig_s806_s801_vol_dryup,       10, 0.20, 50),
    ('S807_s796_compressed_vix',  sig_s807_s786_compressed_vix,  10, 0.20, 50),
    ('S808_s801_macd_up',         sig_s808_s801_macd_up,         10, 0.20, 50),
    ('S809_s801_below50ma',       sig_s809_s801_below50ma,       10, 0.20, 50),
    ('S810_milestone_union_max',  sig_s810_milestone_union_max,  10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E)


# ── Phase 10F — OR-union extensions: MACD combos, precision stacks (S811-S820) ──────────

def sig_s811_s808_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S811: S808 (MACD+OR-union, 9/9yr) + below 50d MA.
    Best wide-coverage MACD signal filtered to bear-phase entries.
    Hypothesis: MACD momentum turn below 50MA = highest snap-back quality."""
    if sig_s808_s801_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


def sig_s812_s808_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S812: S808 (MACD+OR-union) + ATR compression.
    MACD turn at OR-union support during volatility squeeze.
    Hypothesis: compression pushes WR from 82% toward 93%+."""
    if sig_s808_s801_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s813_s809_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S813: S809 (OR-union+below50MA) + ATR compression.
    Bear-phase OR-union with volatility squeeze.
    Hypothesis: 3-filter precision on widest base → WR=93%+, 8yr+ coverage."""
    if sig_s809_s801_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S814: S801 (OR-union base) + TSLA lags SPY ≥5% over 20 days.
    Widest discount coverage + relative weakness = classic mean-reversion.
    Tests if lag5pct adds value beyond existing discount filters."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s815_s808_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S815: S808 (MACD+OR-union, 9/9yr) + TSLA lags SPY ≥5% over 20d.
    Momentum turn with relative weakness confirmation.
    Hypothesis: MACD + SPY lag = higher per-trade PnL, possible 9/9yr retention."""
    if sig_s808_s801_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s816_s805_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S816: S805 (OR-union+compressed) + volume dry-up.
    Double-squeeze: ATR compression + seller exhaustion.
    Hypothesis: OR-union+compressed+dryup = 100% WR (every compressed path hits 100% with dryup)."""
    if sig_s805_s801_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s817_s808_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S817: S808 (MACD+OR-union, 9/9yr) + VIX elevated pct>60.
    Does VIX filter boost the 9/9yr MACD OR-union signal quality?
    Hypothesis: fear-driven dip + MACD reversal = larger snap-back."""
    if sig_s808_s801_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s818_s801_below200ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S818: S801 (OR-union base) + TSLA below 200d MA (long-term downtrend).
    More extreme bear-phase filter than 50MA.
    Hypothesis: OR-union in confirmed bear trend = deeper discount, stronger reversal."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 200:
        return 1
    close = float(tsla['close'].iloc[i])
    ma200 = float(tsla['close'].iloc[i - 200:i].mean())
    return 1 if close < ma200 else 0


def sig_s819_s811_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S819: S811 (MACD+OR-union+below50MA) + ATR compression.
    Triple precision: MACD turn + bear phase + volatility squeeze.
    Hypothesis: best quality gate on 9/9yr base → near-100% WR."""
    if sig_s811_s808_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s820_milestone_s815_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S820 MILESTONE: S815 (MACD+OR-union+lag5pct) + below 50d MA.
    Three independent quality signals: MACD turn + SPY underperformance + bear phase.
    OR-union base provides widest trade coverage with max quality filtering."""
    if sig_s815_s808_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


SIGNALS_P10F: List[Tuple] = [
    # Phase 10F — OR-union extensions: MACD combos, precision stacks (S811-S820)
    ('S811_s808_below50ma',          sig_s811_s808_below50ma,          10, 0.20, 50),
    ('S812_s808_compressed',         sig_s812_s808_compressed,         10, 0.20, 50),
    ('S813_s809_compressed',         sig_s813_s809_compressed,         10, 0.20, 50),
    ('S814_s801_lag5pct',            sig_s814_s801_lag5pct,            10, 0.20, 50),
    ('S815_s808_lag5pct',            sig_s815_s808_lag5pct,            10, 0.20, 50),
    ('S816_s805_vol_dryup',          sig_s816_s805_vol_dryup,          10, 0.20, 50),
    ('S817_s808_vix_pct60',          sig_s817_s808_vix_pct60,          10, 0.20, 50),
    ('S818_s801_below200ma',         sig_s818_s801_below200ma,         10, 0.20, 50),
    ('S819_s811_compressed',         sig_s819_s811_compressed,         10, 0.20, 50),
    ('S820_milestone_s815_below50ma',sig_s820_milestone_s815_below50ma,10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F)


# ── Phase 10G — S814 extensions: lag5pct precision stacks (S821-S830) ──────────────────

def sig_s821_s814_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S821: S814 (OR-union+lag5pct, 9/9yr) + ATR compression.
    SPY-lag + discount union + volatility squeeze.
    Hypothesis: compression pushes WR from 85% to 93%+."""
    if sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s822_s814_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S822: S814 (OR-union+lag5pct, 9/9yr) + below 50d MA.
    SPY-lag discount signal filtered to bear-phase entries.
    Hypothesis: lag5pct + bear phase = strongest mean-reversion setup."""
    if sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


def sig_s823_s814_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S823: S814 (OR-union+lag5pct, 9/9yr) + VIX elevated pct>60.
    SPY-lag + market-wide fear = highest-quality discount entries.
    Hypothesis: VIX filter pushes WR from 85% toward 93%."""
    if sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s824_s814_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S824: S814 (OR-union+lag5pct, 9/9yr) + MACD turning up.
    Momentum confirmation on the lag+discount signal.
    Hypothesis: MACD turn + lag5pct = better timing, higher PF."""
    if sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


def sig_s825_s814_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S825: S814 (OR-union+lag5pct) + volume dry-up.
    Every lag5pct compressed path gives 100% WR — does dryup do the same here?
    Hypothesis: OR-union+lag5pct+dryup = 100% WR (sellers exhausted)."""
    if sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s826_s822_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S826: S822 (S814+below50MA) + ATR compression.
    Four-filter precision: OR-union + lag5pct + bear-phase + squeeze.
    Tests if 4-way filter still covers enough years to be useful."""
    if sig_s822_s814_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s827_s817_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S827: S817 (MACD+OR-union+VIX_pct60, WR=93%, 8/8yr) + lag5pct.
    Three quality filters (MACD + VIX) now with SPY-lag.
    Hypothesis: 4-filter stack on best precision path → near-100% WR."""
    if sig_s817_s808_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s828_s631_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S828: S631 (52wk-disc-only base) + lag5pct.
    Tests if lag5pct adds same value at 52wk-disc base as at OR-union base (S814).
    S814 uses S801 (OR of both discounts), S828 uses S631 (52wk only)."""
    if sig_s631_s215_52wk_disc10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s829_s801_lag10pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S829: S801 (OR-union base) + TSLA lags SPY ≥10% over 20 days.
    Stricter lag threshold than S814 (5% → 10%).
    Hypothesis: deeper relative weakness = even stronger mean-reversion."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.10) else 0


def sig_s830_milestone_s824_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S830 MILESTONE: S824 (S814+MACD) + ATR compression.
    OR-union + lag5pct + MACD turn + squeeze = four independent quality signals.
    Tests the limit of stacking: does 4-filter precision still cover 7+ years?"""
    if sig_s824_s814_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


SIGNALS_P10G: List[Tuple] = [
    # Phase 10G — S814 extensions: lag5pct precision stacks (S821-S830)
    ('S821_s814_compressed',       sig_s821_s814_compressed,       10, 0.20, 50),
    ('S822_s814_below50ma',        sig_s822_s814_below50ma,        10, 0.20, 50),
    ('S823_s814_vix_pct60',        sig_s823_s814_vix_pct60,        10, 0.20, 50),
    ('S824_s814_macd_up',          sig_s824_s814_macd_up,          10, 0.20, 50),
    ('S825_s814_vol_dryup',        sig_s825_s814_vol_dryup,        10, 0.20, 50),
    ('S826_s822_compressed',       sig_s826_s822_compressed,       10, 0.20, 50),
    ('S827_s817_lag5pct',          sig_s827_s817_lag5pct,          10, 0.20, 50),
    ('S828_s631_lag5pct',          sig_s828_s631_lag5pct,          10, 0.20, 50),
    ('S829_s801_lag10pct',         sig_s829_s801_lag10pct,         10, 0.20, 50),
    ('S830_milestone_s824_comp',   sig_s830_milestone_s824_compressed, 10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G)


# ── Phase 10H — Wider discount thresholds + DOW timing + OR-union combos (S831-S840) ──

def sig_s831_s801_disc15_or(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S831: Weekly + OR(52wk_disc15, 20d_disc8).
    Wider discount thresholds than S801 (10%/5%) — trades deeper pullbacks.
    Hypothesis: deeper discount = fewer but higher-quality trades."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    has_52wk = _tsla_below_52wk_pct(tsla, i, pct=0.15)
    has_20d  = _tsla_below_20d_high_pct(tsla, i, pct=0.08)
    return 1 if (has_52wk or has_20d) else 0


def sig_s832_s801_disc7_or(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S832: Weekly + OR(52wk_disc7, 20d_disc3).
    Tighter discount thresholds than S801 (10%/5%) — more trades, shallower pullbacks.
    Hypothesis: wider coverage = more trades but lower per-trade quality."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    has_52wk = _tsla_below_52wk_pct(tsla, i, pct=0.07)
    has_20d  = _tsla_below_20d_high_pct(tsla, i, pct=0.03)
    return 1 if (has_52wk or has_20d) else 0


def sig_s833_s801_monday(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S833: S801 (OR-union, best signal) filtered to Monday entries only.
    Monday = fresh week energy, most gap-fills happen Monday morning.
    Tests if DOW has meaningful impact on OR-union signal quality."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= len(tsla):
        return 0
    return 1 if tsla.index[i].weekday() == 0 else 0  # 0 = Monday


def sig_s834_s801_tuesday(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S834: S801 filtered to Tuesday entries (second-best DOW candidate).
    Tuesday = post-Monday reversal window in mean-reversion."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= len(tsla):
        return 0
    return 1 if tsla.index[i].weekday() == 1 else 0  # 1 = Tuesday


def sig_s835_s801_mon_tue(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S835: S801 filtered to Monday OR Tuesday entries.
    First half of week = best DOW window for mean-reversion.
    Hypothesis: Mon+Tue entries = strongest reversal window."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= len(tsla):
        return 0
    dow = tsla.index[i].weekday()
    return 1 if dow in (0, 1) else 0


def sig_s836_s814_macd_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S836: S824 (S814+MACD) + ATR compression = S830 explored path.
    Triple filter on 9/9yr lag5pct base: MACD turn + discount + squeeze.
    Tests if MACD+compressed adds to lag5pct beyond plain compressed."""
    if sig_s824_s814_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s837_s801_rsi30(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S837: S801 (OR-union) + RSI14 < 30 (oversold).
    Classic oversold filter on the best discount signal.
    Hypothesis: RSI oversold + deep discount at weekly support = strong reversal."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= len(rt) or pd.isna(rt.iloc[i]):
        return 1
    return 1 if rt.iloc[i] < 30.0 else 0


def sig_s838_s801_rsi40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S838: S801 (OR-union) + RSI14 < 40 (approaching oversold).
    Slightly looser RSI threshold — captures more trades than RSI<30.
    Tests optimal RSI threshold for OR-union discount signal."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= len(rt) or pd.isna(rt.iloc[i]):
        return 1
    return 1 if rt.iloc[i] < 40.0 else 0


def sig_s839_s814_rsi40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S839: S814 (OR-union+lag5pct, 9/9yr) + RSI14 < 40.
    Does RSI oversold add to the lag5pct signal?
    Hypothesis: lag5pct already implies RSI weakness — may be redundant."""
    if sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= len(rt) or pd.isna(rt.iloc[i]):
        return 1
    return 1 if rt.iloc[i] < 40.0 else 0


def sig_s840_milestone_s831_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S840 MILESTONE: S831 (wider OR-union 15%/8%) + ATR compression.
    Deeper discount thresholds + volatility squeeze.
    Tests if wider discount + compression finds higher-quality deep-value setups."""
    if sig_s831_s801_disc15_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


SIGNALS_P10H: List[Tuple] = [
    # Phase 10H — Discount threshold sweep + DOW + RSI (S831-S840)
    ('S831_s801_disc15_or',        sig_s831_s801_disc15_or,        10, 0.20, 50),
    ('S832_s801_disc7_or',         sig_s832_s801_disc7_or,         10, 0.20, 50),
    ('S833_s801_monday',           sig_s833_s801_monday,           10, 0.20, 50),
    ('S834_s801_tuesday',          sig_s834_s801_tuesday,          10, 0.20, 50),
    ('S835_s801_mon_tue',          sig_s835_s801_mon_tue,          10, 0.20, 50),
    ('S836_s814_macd_compressed',  sig_s836_s814_macd_compressed,  10, 0.20, 50),
    ('S837_s801_rsi30',            sig_s837_s801_rsi30,            10, 0.20, 50),
    ('S838_s801_rsi40',            sig_s838_s801_rsi40,            10, 0.20, 50),
    ('S839_s814_rsi40',            sig_s839_s814_rsi40,            10, 0.20, 50),
    ('S840_milestone_s831_comp',   sig_s840_milestone_s831_compressed, 10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H)


# ── Phase 10I — Hold sweep on top signals + new OR dimensions (S841-S850) ──────────────

def sig_s841_s801_hold5(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S841: S801 (OR-union, best) with 5-day max hold (vs default 10).
    Tests if shorter hold improves precision for the top signal.
    Hypothesis: OR-union bounces often reverse within 5 days."""
    return sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s842_s801_hold15(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S842: S801 (OR-union) with 15-day max hold.
    Tests if longer hold captures more of the bounce move.
    Hypothesis: bigger channel bounces take >10 days to play out."""
    return sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s843_s814_hold5(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S843: S814 (OR-union+lag5pct, 9/9yr) with 5-day hold.
    Shorter hold for the lag5pct signal — does lag signal resolve faster?"""
    return sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s844_s821_hold5(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S844: S821 (OR-union+lag5pct+compressed, 100% WR) with 5-day hold.
    Does the 100% WR signal resolve faster than 10 days?
    Hypothesis: compressed signals mean imminent move → 5d might capture it fully."""
    return sig_s821_s814_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s845_s801_or_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S845: OR(S801 discount union, TSLA-lags-SPY-5pct).
    Broadens the OR logic: buy if EITHER discount criteria OR relative weakness.
    Hypothesis: maximum coverage union = highest total P&L."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    has_disc = (_tsla_below_52wk_pct(tsla, i, pct=0.10) or
                _tsla_below_20d_high_pct(tsla, i, pct=0.05))
    has_lag  = _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05)
    return 1 if (has_disc or has_lag) else 0


def sig_s846_s801_and_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S846: AND(S801 discount union, TSLA-lags-SPY-5pct) = S814 re-verification.
    Both discount AND lag must be true — should equal S814 result ($1,377K).
    Confirms S814's implicit path."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    has_disc = (_tsla_below_52wk_pct(tsla, i, pct=0.10) or
                _tsla_below_20d_high_pct(tsla, i, pct=0.05))
    has_lag  = _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05)
    return 1 if (has_disc and has_lag) else 0


def sig_s847_s801_consec_down3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S847: S801 (OR-union) + 3 consecutive down days.
    Classic oversold filter: TSLA down 3 days in a row at weekly support.
    Hypothesis: multi-day selling exhaustion = snap-back catalyst."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=3) else 0


def sig_s848_s801_consec_down2(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S848: S801 (OR-union) + 2 consecutive down days.
    Slightly looser than 3-day; more trades but same directional exhaustion.
    Hypothesis: 2-day down more common, better coverage."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=2) else 0


def sig_s849_s814_consec_down2(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S849: S814 (OR-union+lag5pct, 9/9yr) + 2 consecutive down days.
    Selling exhaustion confirmation on the lag signal.
    Hypothesis: 2-day down + lagging SPY = maximum mean-reversion pressure."""
    if sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_consec_down(tsla, i, n=2) else 0


def sig_s850_milestone_s845_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S850 MILESTONE: S845 (OR-union OR lag5pct — widest base) + compressed.
    Maximum possible trade coverage + volatility squeeze quality gate.
    Hypothesis: widest valid OR-union + squeeze = highest P&L with 93%+ WR."""
    if sig_s845_s801_or_spy_lag(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


SIGNALS_P10I: List[Tuple] = [
    # Phase 10I — Hold sweep + OR expansions + consec down (S841-S850)
    ('S841_s801_hold5',            sig_s841_s801_hold5,            5,  0.20, 50),
    ('S842_s801_hold15',           sig_s842_s801_hold15,           15, 0.20, 50),
    ('S843_s814_hold5',            sig_s843_s814_hold5,            5,  0.20, 50),
    ('S844_s821_hold5',            sig_s844_s821_hold5,            5,  0.20, 50),
    ('S845_s801_or_spy_lag',       sig_s845_s801_or_spy_lag,       10, 0.20, 50),
    ('S846_s801_and_spy_lag',      sig_s846_s801_and_spy_lag,      10, 0.20, 50),
    ('S847_s801_consec_down3',     sig_s847_s801_consec_down3,     10, 0.20, 50),
    ('S848_s801_consec_down2',     sig_s848_s801_consec_down2,     10, 0.20, 50),
    ('S849_s814_consec_down2',     sig_s849_s814_consec_down2,     10, 0.20, 50),
    ('S850_milestone_s845_comp',   sig_s850_milestone_s845_compressed, 10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I)


# ── Phase 10J — Intraday close strength + VIX rolling window + seasonal (S851-S860) ───

def _tsla_strong_close(tsla, i, pct: float = 0.60) -> bool:
    """True if daily close is in top pct of day's high-low range (bullish close).
    pct=0.60 means close is at least 60% of the way from low to high."""
    h = float(tsla['high'].iloc[i])
    l = float(tsla['low'].iloc[i])
    c = float(tsla['close'].iloc[i])
    rng = h - l
    if rng <= 0:
        return True  # doji / flat = neutral, pass
    position = (c - l) / rng
    return position >= pct


def _tsla_weak_close(tsla, i, pct: float = 0.40) -> bool:
    """True if daily close is in bottom pct of day's high-low range (bearish close).
    pct=0.40 means close is in the lower 40% of the day's range."""
    h = float(tsla['high'].iloc[i])
    l = float(tsla['low'].iloc[i])
    c = float(tsla['close'].iloc[i])
    rng = h - l
    if rng <= 0:
        return True
    position = (c - l) / rng
    return position <= pct


def _vix_elevated_pct_63(vix, i, pct: float = 0.60) -> bool:
    """VIX elevated above pct percentile over rolling 63-day (~3 month) window.
    Shorter lookback = more responsive to recent fear vs 252-day annual window."""
    lookback = min(63, i)
    if lookback < 20:
        return False
    vix_vals = vix['close'].iloc[i - lookback:i]
    vix_now   = float(vix['close'].iloc[i])
    threshold = float(vix_vals.quantile(pct))
    return vix_now >= threshold


def sig_s851_s801_strong_close(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S851: S801 (OR-union) + strong close (close in top 60% of day's range).
    Intraday strength = buyers absorbing selling at weekly support.
    Hypothesis: strong close on signal day = conviction reversal, higher WR."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_strong_close(tsla, i, pct=0.60) else 0


def sig_s852_s801_weak_close(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S852: S801 (OR-union) + weak close (close in bottom 40% of range).
    Tests if entering on weak close (still selling) is better (buy the weakness).
    Hypothesis: weak close = not yet done selling, buy tomorrow's open instead."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_weak_close(tsla, i, pct=0.40) else 0


def sig_s853_s814_strong_close(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S853: S814 (OR-union+lag5pct, 9/9yr) + strong close.
    Intraday conviction on the lag5pct signal.
    Hypothesis: strong close + lag + discount = highest daily reversal quality."""
    if sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_strong_close(tsla, i, pct=0.60) else 0


def sig_s854_s801_vix_pct60_63d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S854: S801 (OR-union) + VIX pct>60 over 63-day rolling window.
    Shorter VIX lookback = more responsive to recent fear spikes.
    Compare to S817 (annual VIX pct60) — which window works better?"""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct_63(vix, i, pct=0.60) else 0


def sig_s855_s801_vix_pct80(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S855: S801 (OR-union) + VIX pct>80 over 252-day window.
    Stricter VIX threshold — only top-20% fear events.
    Hypothesis: extreme fear + discount = largest bounces."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.80) else 0


def sig_s856_s814_strong_close_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S856: S821 (S814+compressed) + strong close.
    100% WR compressed signal with intraday conviction.
    Hypothesis: strong close + compressed + lag + discount = maximum setup quality."""
    if sig_s821_s814_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_strong_close(tsla, i, pct=0.60) else 0


def sig_s857_s801_oct_nov(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S857: S801 filtered to October-November entries (seasonal weak period).
    Oct/Nov historically shows strong TSLA mean-reversion from channel support.
    Hypothesis: seasonal weakness + discount = better recovery."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= len(tsla):
        return 0
    month = tsla.index[i].month
    return 1 if month in (10, 11) else 0


def sig_s858_s801_q1_q4(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S858: S801 filtered to Q1 (Jan-Mar) OR Q4 (Oct-Dec) entries.
    Historically strongest TSLA reversal periods.
    Hypothesis: TSLA seasonal strength in Q1 + Q4 amplifies bounce quality."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= len(tsla):
        return 0
    month = tsla.index[i].month
    return 1 if month in (1, 2, 3, 10, 11, 12) else 0


def sig_s859_s801_q2_q3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S859: S801 filtered to Q2-Q3 (Apr-Sep) — typically TSLA's weaker half.
    Complement to S858 — tests if seasonal effects actually exist in signal.
    If S859 > S858, seasonal filter is counter-productive."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= len(tsla):
        return 0
    month = tsla.index[i].month
    return 1 if month in (4, 5, 6, 7, 8, 9) else 0


def sig_s860_milestone_s853_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S860 MILESTONE: S821 (100% WR base) + strong close.
    Compressed lag5pct signal with intraday buyer conviction.
    Maximum quality: OR-union + lag5pct + compressed + strong close = 4 filters."""
    if sig_s821_s814_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_strong_close(tsla, i, pct=0.60) else 0


SIGNALS_P10J: List[Tuple] = [
    # Phase 10J — Close strength + VIX windows + seasonal (S851-S860)
    ('S851_s801_strong_close',       sig_s851_s801_strong_close,       10, 0.20, 50),
    ('S852_s801_weak_close',         sig_s852_s801_weak_close,         10, 0.20, 50),
    ('S853_s814_strong_close',       sig_s853_s814_strong_close,       10, 0.20, 50),
    ('S854_s801_vix_pct60_63d',      sig_s854_s801_vix_pct60_63d,      10, 0.20, 50),
    ('S855_s801_vix_pct80',          sig_s855_s801_vix_pct80,          10, 0.20, 50),
    ('S856_s821_strong_close',       sig_s856_s814_strong_close_compressed, 10, 0.20, 50),
    ('S857_s801_oct_nov',            sig_s857_s801_oct_nov,            10, 0.20, 50),
    ('S858_s801_q1_q4',              sig_s858_s801_q1_q4,              10, 0.20, 50),
    ('S859_s801_q2_q3',              sig_s859_s801_q2_q3,              10, 0.20, 50),
    ('S860_milestone_s821_strclose', sig_s860_milestone_s853_compressed,   10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J)


# ── Phase 10K — Weekly channel tightness + SPY co-weakness + VIX momentum (S861-S870) ─

def _weekly_channel_pos(tw, daily_date, window: int = 20) -> float:
    """Fractional position in weekly channel (0.0 = at lower, 1.0 = at upper).
    Returns -1.0 if channel not detected or insufficient data."""
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < window:
        return -1.0
    ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
    if ch is None:
        return -1.0
    lower = ch.lower_line[-1]
    upper = ch.upper_line[-1]
    w = upper - lower
    if w <= 0:
        return -1.0
    price = float(tw['close'].iloc[wk_idx])
    return (price - lower) / w


def _vix_rising(vix, i, pct_rise: float = 0.05) -> bool:
    """True if today's VIX is at least pct_rise higher than yesterday's.
    Rising VIX = increasing fear = potentially better entry (fear is peaking)."""
    if i < 1:
        return False
    vix_now  = float(vix['close'].iloc[i])
    vix_prev = float(vix['close'].iloc[i - 1])
    if vix_prev <= 0:
        return False
    return (vix_now - vix_prev) / vix_prev >= pct_rise


def _tsla_5d_return(tsla, i) -> float:
    """Return TSLA 5-day price change as fraction (-0.10 = down 10%)."""
    if i < 5:
        return 0.0
    c_now  = float(tsla['close'].iloc[i])
    c_prev = float(tsla['close'].iloc[i - 5])
    if c_prev <= 0:
        return 0.0
    return (c_now - c_prev) / c_prev


def sig_s861_s801_tight_channel10(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S861: S801 (OR-union) + weekly channel position < 10% (very close to lower band).
    Tighter channel filter than S215 default 25% — only bottom-10% entries.
    Hypothesis: deeper in channel = stronger support test = higher quality bounce."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    pos = _weekly_channel_pos(tw, tsla.index[i], window=20)
    if pos < 0:
        return 1  # channel not detected — keep the trade
    return 1 if pos < 0.10 else 0


def sig_s862_s801_tight_channel15(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S862: S801 + weekly channel position < 15% (bottom-15% entries).
    Intermediate between 10% (S861) and 25% (S215 default).
    Tests optimal channel position threshold."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    pos = _weekly_channel_pos(tw, tsla.index[i], window=20)
    if pos < 0:
        return 1
    return 1 if pos < 0.15 else 0


def sig_s863_s814_tight_channel15(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S863: S814 (OR-union+lag5pct, 9/9yr) + weekly channel position < 15%.
    Tighter channel position on the best 9/9yr signal.
    Hypothesis: lag+discount+deep_channel = highest precision."""
    if sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    pos = _weekly_channel_pos(tw, tsla.index[i], window=20)
    if pos < 0:
        return 1
    return 1 if pos < 0.15 else 0


def sig_s864_s801_spy_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S864: S801 (OR-union) + SPY also 5%+ below its 20-day high.
    Market-wide weakness confirmation — both TSLA and SPY in pullback.
    Hypothesis: broad market + TSLA weakness = stronger reversal when support holds."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    spy_high_20d = float(spy['high'].iloc[i - 20:i].max())
    if spy_high_20d <= 0:
        return 1
    spy_disc = (spy_high_20d - float(spy['close'].iloc[i])) / spy_high_20d
    return 1 if spy_disc >= 0.05 else 0


def sig_s865_s801_vix_rising(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S865: S801 (OR-union) + VIX rising ≥5% from yesterday.
    Increasing fear = panic selling at support = better entry timing.
    Hypothesis: VIX spike day at support = selling climax."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_rising(vix, i, pct_rise=0.05) else 0


def sig_s866_s801_5d_momentum(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S866: S801 (OR-union) + TSLA down ≥5% over 5 days (momentum exhaustion).
    Short-term momentum oversold at weekly support.
    Hypothesis: 5-day drop + channel support = multi-day mean reversion."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    ret_5d = _tsla_5d_return(tsla, i)
    return 1 if ret_5d <= -0.05 else 0


def sig_s867_s814_spy_disc5(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S867: S814 (OR-union+lag5pct, 9/9yr) + SPY also 5%+ below 20d high.
    SPY co-weakness on the best 9/9yr signal.
    Hypothesis: when BOTH TSLA lags AND SPY is weak, reversals are strongest."""
    if sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    spy_high_20d = float(spy['high'].iloc[i - 20:i].max())
    if spy_high_20d <= 0:
        return 1
    spy_disc = (spy_high_20d - float(spy['close'].iloc[i])) / spy_high_20d
    return 1 if spy_disc >= 0.05 else 0


def sig_s868_s801_spy_disc3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S868: S801 + SPY also 3%+ below its 20-day high (looser SPY filter).
    Compare to S864 (5% threshold) — does SPY co-weakness matter at any threshold?
    If S868 < S801 and S864 < S801, SPY co-weakness = irrelevant."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 1
    spy_high_20d = float(spy['high'].iloc[i - 20:i].max())
    if spy_high_20d <= 0:
        return 1
    spy_disc = (spy_high_20d - float(spy['close'].iloc[i])) / spy_high_20d
    return 1 if spy_disc >= 0.03 else 0


def sig_s869_s801_3d_drop5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S869: S801 (OR-union) + TSLA dropped ≥5% in last 3 days.
    Very short-term momentum crash at weekly support.
    Hypothesis: 3-day 5%+ drop = acute panic at support = snap-back imminent."""
    if sig_s801_s215_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    c_now  = float(tsla['close'].iloc[i])
    c_prev = float(tsla['close'].iloc[i - 3])
    if c_prev <= 0:
        return 1
    ret_3d = (c_now - c_prev) / c_prev
    return 1 if ret_3d <= -0.05 else 0


def sig_s870_milestone_s861_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S870 MILESTONE: Tight channel (bottom 10%) + OR-union + lag5pct.
    Deepest channel position filter on the best coverage signal.
    Tests if 'very deep in channel' adds information beyond discount thresholds."""
    if sig_s814_s801_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    pos = _weekly_channel_pos(tw, tsla.index[i], window=20)
    if pos < 0:
        return 1
    return 1 if pos < 0.10 else 0


SIGNALS_P10K: List[Tuple] = [
    # Phase 10K — Channel tightness + SPY co-weakness + VIX momentum (S861-S870)
    ('S861_s801_tight_chan10',      sig_s861_s801_tight_channel10,  10, 0.20, 50),
    ('S862_s801_tight_chan15',      sig_s862_s801_tight_channel15,  10, 0.20, 50),
    ('S863_s814_tight_chan15',      sig_s863_s814_tight_channel15,  10, 0.20, 50),
    ('S864_s801_spy_disc5',         sig_s864_s801_spy_disc5,        10, 0.20, 50),
    ('S865_s801_vix_rising',        sig_s865_s801_vix_rising,       10, 0.20, 50),
    ('S866_s801_5d_momentum',       sig_s866_s801_5d_momentum,      10, 0.20, 50),
    ('S867_s814_spy_disc5',         sig_s867_s814_spy_disc5,        10, 0.20, 50),
    ('S868_s801_spy_disc3',         sig_s868_s801_spy_disc3,        10, 0.20, 50),
    ('S869_s801_3d_drop5pct',       sig_s869_s801_3d_drop5pct,      10, 0.20, 50),
    ('S870_milestone_s861_lag5pct', sig_s870_milestone_s861_lag5pct,10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K)


# ── Phase 10L — S861 extensions: tight channel + precision filters (S871-S880) ─────────

def sig_s871_s861_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S871: S861 (OR-union+tight_chan10, WR=90%) + ATR compression.
    Tight channel + volatility squeeze = maximum entry precision.
    Hypothesis: WR=90% compressed → 95%+, new highest-precision path."""
    if sig_s861_s801_tight_channel10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s872_s861_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S872: S861 (OR-union+tight_chan10) + TSLA lags SPY ≥5%.
    Tight channel position + relative weakness confirmation.
    Hypothesis: very deep in channel + lagging SPY = highest snap-back quality."""
    if sig_s861_s801_tight_channel10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s873_s861_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S873: S861 (tight_chan10, WR=90%) + VIX elevated pct>60.
    Fear filter on tight channel signal — already WR=90%, VIX pushes higher?
    Hypothesis: tight channel + fear context = 95%+ WR."""
    if sig_s861_s801_tight_channel10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s874_s861_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S874: S861 (tight_chan10) + below 50d MA.
    Bear phase + deep channel position = defensive oversold setup.
    Hypothesis: tight channel below 50MA = stronger mean reversion."""
    if sig_s861_s801_tight_channel10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


def sig_s875_s862_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S875: S862 (OR-union+tight_chan15, WR=85%) + ATR compression.
    Slightly wider channel threshold + squeeze.
    Compare to S871 (tight_chan10+compressed) — threshold sensitivity."""
    if sig_s862_s801_tight_channel15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s876_s862_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S876: S862 (tight_chan15) + lag5pct.
    Wider channel position threshold version of S872.
    Tests optimal channel tightness threshold for the lag5pct dimension."""
    if sig_s862_s801_tight_channel15(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s877_s871_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S877: S871 (tight_chan10+compressed) + lag5pct.
    Three quality dimensions: tight channel + squeeze + SPY lag.
    Hypothesis: triple filter on WR=90% base → near-100% WR."""
    if sig_s871_s861_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s878_s861_macd_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S878: S861 (tight_chan10, WR=90%) + MACD turning up.
    Timing confirmation on the tight channel signal.
    Hypothesis: WR=90% + MACD = 93%+, improved PF."""
    if sig_s861_s801_tight_channel10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist_now  = _macd_histogram(tsla, i)
    hist_prev = _macd_histogram(tsla, i - 1) if i > 0 else None
    if hist_now is None or hist_prev is None:
        return 1
    return 1 if hist_now > hist_prev else 0


def sig_s879_s861_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S879: S861 (tight_chan10, WR=90%) + volume dry-up.
    Seller exhaustion at tight channel support.
    Hypothesis: every tight_chan+dryup combination = 100% WR."""
    if sig_s861_s801_tight_channel10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s880_milestone_s871_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S880 MILESTONE: S871 (tight_chan10+compressed) + below50MA.
    Deep channel position + compressed + bear phase = triple filter precision.
    Tests if bear-phase filter adds to already-tight compressed channel signal."""
    if sig_s871_s861_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


SIGNALS_P10L: List[Tuple] = [
    # Phase 10L — S861 extensions: tight channel precision (S871-S880)
    ('S871_s861_compressed',        sig_s871_s861_compressed,        10, 0.20, 50),
    ('S872_s861_lag5pct',           sig_s872_s861_lag5pct,           10, 0.20, 50),
    ('S873_s861_vix_pct60',         sig_s873_s861_vix_pct60,         10, 0.20, 50),
    ('S874_s861_below50ma',         sig_s874_s861_below50ma,         10, 0.20, 50),
    ('S875_s862_compressed',        sig_s875_s862_compressed,        10, 0.20, 50),
    ('S876_s862_lag5pct',           sig_s876_s862_lag5pct,           10, 0.20, 50),
    ('S877_s871_lag5pct',           sig_s877_s871_lag5pct,           10, 0.20, 50),
    ('S878_s861_macd_up',           sig_s878_s861_macd_up,           10, 0.20, 50),
    ('S879_s861_vol_dryup',         sig_s879_s861_vol_dryup,         10, 0.20, 50),
    ('S880_milestone_s871_b50ma',   sig_s880_milestone_s871_below50ma, 10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L)


# ── Phase 10M — S875 extensions + OR-union+tight_chan combinations (S881-S890) ─────────

def sig_s881_s875_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S881: S875 (tight_chan15+compressed, WR=92%, 9/9yr) + lag5pct.
    Tests if lag5pct adds to the tight_chan15+compressed signal.
    Hypothesis: may converge to S821 (100% WR) via tight_chan+compressed subsumption."""
    if sig_s875_s862_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s882_s875_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S882: S875 (tight_chan15+compressed, WR=92%) + VIX pct>60.
    Fear filter on the 9/9yr precision signal.
    Hypothesis: VIX pushes WR from 92% to 96%+."""
    if sig_s875_s862_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s883_s875_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S883: S875 (tight_chan15+compressed) + below 50d MA.
    Bear phase filter on 9/9yr signal.
    Hypothesis: tight_chan15+compressed already includes below50MA trades."""
    if sig_s875_s862_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


def sig_s884_s875_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S884: S875 (tight_chan15+compressed) + volume dry-up.
    Triple filter: tight channel + compression + seller exhaustion.
    Hypothesis: 100% WR (every compressed+dryup = 100% WR path)."""
    if sig_s875_s862_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s885_s801_tight_or_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S885: S801 base + OR(tight_chan15, lag5pct).
    Broadest quality-gated OR-union: OR-union + (tight_channel OR lag5pct).
    Hypothesis: union of two quality filters = more trades with higher average quality."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    has_disc = (_tsla_below_52wk_pct(tsla, i, pct=0.10) or
                _tsla_below_20d_high_pct(tsla, i, pct=0.05))
    if not has_disc:
        return 0
    pos = _weekly_channel_pos(tw, tsla.index[i], window=20)
    tight_chan = (pos < 0 or pos < 0.15)  # true if tight or no channel
    has_lag   = _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05)
    return 1 if (tight_chan or has_lag) else 0


def sig_s886_s861_30w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S886: OR-union discount + tight channel based on 30-WEEK channel (vs 20-week).
    Longer channel window may identify more robust support levels.
    Hypothesis: 30-week channel lower band = stronger structural support."""
    if sig_s215_s214_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    has_disc = (_tsla_below_52wk_pct(tsla, i, pct=0.10) or
                _tsla_below_20d_high_pct(tsla, i, pct=0.05))
    if not has_disc:
        return 0
    pos = _weekly_channel_pos(tw, tsla.index[i], window=30)
    if pos < 0:
        return 1  # no 30-week channel — keep trade
    return 1 if pos < 0.10 else 0


def sig_s887_s215_30w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S887: Weekly 30-week channel lower 10% (no discount requirement).
    Tests if 30-week channel alone (without discount) has signal value.
    Baseline for comparing 30-week vs 20-week channel approaches."""
    if tw is None or len(tw) < 30:
        return 0
    daily_date = tsla.index[i]
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 30:
        return 0
    # VIX filter
    if i < 252:
        return 0
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    # 30-week channel, bottom 25%
    ch = _channel_at(tw.iloc[wk_idx - 30:wk_idx])
    if ch is None:
        return 0
    return 1 if _near_lower(tw['close'].iloc[wk_idx], ch, 0.25) else 0


def sig_s888_s887_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S888: S887 (30-week channel) + OR-union discount filter.
    30-week channel support + discount anchor.
    Hypothesis: longer structural channel + discount = stronger signal."""
    if sig_s887_s215_30w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    has_disc = (_tsla_below_52wk_pct(tsla, i, pct=0.10) or
                _tsla_below_20d_high_pct(tsla, i, pct=0.05))
    return 1 if has_disc else 0


def sig_s889_s888_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S889: S888 (30-week channel + OR-union discount) + lag5pct.
    Full quality stack on 30-week channel base.
    Hypothesis: 30-week structural support + discount + SPY lag = premium setup."""
    if sig_s888_s887_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s890_milestone_s888_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S890 MILESTONE: S888 (30-week channel + OR-union) + ATR compression.
    30-week structural support + discount + volatility squeeze.
    Tests if 30-week channel adds anything over 20-week when combined with compression."""
    if sig_s888_s887_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


SIGNALS_P10M: List[Tuple] = [
    # Phase 10M — S875 extensions + 30-week channel + OR combinations (S881-S890)
    ('S881_s875_lag5pct',           sig_s881_s875_lag5pct,           10, 0.20, 50),
    ('S882_s875_vix_pct60',         sig_s882_s875_vix_pct60,         10, 0.20, 50),
    ('S883_s875_below50ma',         sig_s883_s875_below50ma,         10, 0.20, 50),
    ('S884_s875_vol_dryup',         sig_s884_s875_vol_dryup,         10, 0.20, 50),
    ('S885_s801_tight_or_lag',      sig_s885_s801_tight_or_lag,      10, 0.20, 50),
    ('S886_s861_30w_chan',          sig_s886_s861_30w_channel,        10, 0.20, 50),
    ('S887_s215_30w_channel',       sig_s887_s215_30w_channel,        10, 0.20, 50),
    ('S888_s887_disc_or',           sig_s888_s887_disc_or,            10, 0.20, 50),
    ('S889_s888_lag5pct',           sig_s889_s888_lag5pct,            10, 0.20, 50),
    ('S890_milestone_s888_comp',    sig_s890_milestone_s888_compressed,10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M)


# ── Phase 10N — S890 extensions: 30-week channel precision family (S891-S900) ──────────

def sig_s891_s890_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S891: S890 (30w+OR-union+compressed, WR=94%) + lag5pct.
    Tests if lag5pct adds to already-WR=94% signal.
    Hypothesis: may converge (compressed already selects lag-quality trades)."""
    if sig_s890_milestone_s888_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s892_s890_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S892: S890 (30w+OR-union+compressed, WR=94%) + VIX pct>60.
    Fear filter on 94% WR signal — can it reach 100%?
    Hypothesis: VIX on WR=94% compressed = 100% WR target."""
    if sig_s890_milestone_s888_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s893_s890_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S893: S890 (30w+OR-union+compressed) + below 50d MA.
    Bear phase on WR=94% signal.
    Hypothesis: 30w+compressed trades already below50MA — likely convergence."""
    if sig_s890_milestone_s888_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


def sig_s894_s890_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S894: S890 (30w+OR-union+compressed) + volume dry-up.
    Every compressed+dryup path = 100% WR. Does 30w channel also follow this rule?
    Hypothesis: 30w+compressed+dryup = 100% WR."""
    if sig_s890_milestone_s888_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i) else 0


def sig_s895_s887_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S895: 30-week channel (base, no discount) + ATR compression.
    Tests if 30-week channel alone with compression has signal value.
    Compare to S890 (30w+OR-union+compressed) to measure discount contribution."""
    if sig_s887_s215_30w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s896_s887_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S896: 30-week channel + lag5pct (no compression).
    Tests if lag5pct is the key ingredient for 30w channel quality.
    Compare to S890: compressed vs lag5pct on 30w+OR-union base."""
    if sig_s888_s887_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s897_s888_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S897: S888 (30w+OR-union) + below 50d MA.
    Bear phase on 30-week channel signal (without compression).
    Compares to S893 (which adds compression first)."""
    if sig_s888_s887_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


def sig_s898_s887_vix18_50(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S898: 30-week channel lower band + VIX 18-50 regime (no discount).
    Same structure as S215 but with 30-week instead of 20-week channel.
    Tests if 30-week window alone (without OR-union discount) is a valid signal."""
    if tw is None or len(tw) < 30:
        return 0
    daily_date = tsla.index[i]
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 30:
        return 0
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    ch = _channel_at(tw.iloc[wk_idx - 30:wk_idx])
    if ch is None:
        return 0
    return 1 if _near_lower(tw['close'].iloc[wk_idx], ch, 0.10) else 0


def sig_s899_s898_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S899: S898 (30w tight chan10+VIX) + OR-union discount.
    Tight 30-week channel (bottom 10%) + VIX regime + discount confirmation.
    This is S886 restructured to use tighter channel threshold."""
    if sig_s898_s887_vix18_50(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    has_disc = (_tsla_below_52wk_pct(tsla, i, pct=0.10) or
                _tsla_below_20d_high_pct(tsla, i, pct=0.05))
    return 1 if has_disc else 0


def sig_s900_milestone_s899_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S900 MILESTONE: S899 (30w tight+VIX+discount) + ATR compression.
    Maximum precision: 30-week structural channel + tight position + VIX + discount + squeeze.
    Tests if adding VIX regime to S890 improves WR above 94%."""
    if sig_s899_s898_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


SIGNALS_P10N: List[Tuple] = [
    # Phase 10N — S890 extensions: 30-week channel precision family (S891-S900)
    ('S891_s890_lag5pct',           sig_s891_s890_lag5pct,           10, 0.20, 50),
    ('S892_s890_vix_pct60',         sig_s892_s890_vix_pct60,         10, 0.20, 50),
    ('S893_s890_below50ma',         sig_s893_s890_below50ma,         10, 0.20, 50),
    ('S894_s890_vol_dryup',         sig_s894_s890_vol_dryup,         10, 0.20, 50),
    ('S895_s887_compressed',        sig_s895_s887_compressed,        10, 0.20, 50),
    ('S896_s888_lag5pct',           sig_s896_s887_lag5pct,           10, 0.20, 50),
    ('S897_s888_below50ma',         sig_s897_s888_below50ma,         10, 0.20, 50),
    ('S898_s887_vix18_50',          sig_s898_s887_vix18_50,          10, 0.20, 50),
    ('S899_s898_disc_or',           sig_s899_s898_disc_or,           10, 0.20, 50),
    ('S900_milestone_s899_comp',    sig_s900_milestone_s899_compressed, 10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N)


# ── Phase 10O — 30w channel pure variations + 40w channel (S901-S910) ───────────────────

def _weekly_channel_lower_30w(tw, daily_date, frac: float = 0.25) -> bool:
    """True if weekly bar at daily_date is in bottom frac of 30-week channel."""
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 30:
        return False
    ch = _channel_at(tw.iloc[wk_idx - 30:wk_idx])
    if ch is None:
        return False
    return _near_lower(float(tw['close'].iloc[wk_idx]), ch, frac)


def sig_s901_s895_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S901: S895 (30w base+compressed, WR=94%=S890) + lag5pct.
    Same result as S891 — verifies convergence (S895=S890, so S895+lag=S891).
    Tests that S895 path is truly identical to S890 path."""
    if sig_s895_s887_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s902_s895_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S902: S895 (30w+compressed, WR=94%) + VIX pct>60.
    Same as S892 (S890+VIX) — verifies convergence.
    Both should give WR=91%, 7/7yr."""
    if sig_s895_s887_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s903_s215_40w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S903: 40-week channel lower 25% + VIX 18-50.
    Even longer structural channel — tests if 40-week channel finds better support.
    Hypothesis: 40-week channel lower band = multi-month structural floor."""
    if tw is None or len(tw) < 40:
        return 0
    daily_date = tsla.index[i]
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 40:
        return 0
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    ch = _channel_at(tw.iloc[wk_idx - 40:wk_idx])
    if ch is None:
        return 0
    return 1 if _near_lower(tw['close'].iloc[wk_idx], ch, 0.25) else 0


def sig_s904_s903_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S904: 40-week channel + ATR compression.
    40-week structural support + volatility squeeze.
    Hypothesis: even longer channel = more selective but higher quality per trade."""
    if sig_s903_s215_40w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s905_s903_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S905: 40-week channel + OR-union discount.
    Tests if OR-union discount adds to 40-week channel (compare to S895: adds nothing to 30w).
    Hypothesis: still redundant for 40-week (channel position is the dominant filter)."""
    if sig_s903_s215_40w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    has_disc = (_tsla_below_52wk_pct(tsla, i, pct=0.10) or
                _tsla_below_20d_high_pct(tsla, i, pct=0.05))
    return 1 if has_disc else 0


def sig_s906_s905_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S906: 40-week channel + OR-union discount + compressed.
    Direct comparison to S890 (30w version) — does 40-week improve further?
    Hypothesis: WR > 94%, P&L similar but fewer trades."""
    if sig_s905_s903_disc_or(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s907_s887_below50ma_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S907: 30w channel + below50MA + ATR compression.
    Tests if below50MA adds to the 30w+compressed signal (S895).
    Hypothesis: redundant (S893≈S890 showed below50MA doesn't change 30w compressed trades)."""
    if sig_s887_s215_30w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_above_50ma(tsla, i):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s908_s215_15w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S908: 15-week channel lower 25% + VIX 18-50.
    Shorter than S215 default 20-week — tests if shorter window is better.
    Hypothesis: 15-week is too noisy and gives fewer valid channel detections."""
    if tw is None or len(tw) < 15:
        return 0
    daily_date = tsla.index[i]
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 15:
        return 0
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    ch = _channel_at(tw.iloc[wk_idx - 15:wk_idx])
    if ch is None:
        return 0
    return 1 if _near_lower(tw['close'].iloc[wk_idx], ch, 0.25) else 0


def sig_s909_s908_disc_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S909: 15-week channel + OR-union discount + compressed.
    Direct comparison to S890 (30w) and S224 (20w).
    Tests if 15-week is weaker or stronger than 20/30 weeks."""
    if sig_s908_s215_15w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    has_disc = (_tsla_below_52wk_pct(tsla, i, pct=0.10) or
                _tsla_below_20d_high_pct(tsla, i, pct=0.05))
    if not has_disc:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s910_milestone_channel_sweep(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S910 MILESTONE: Best of all channel windows — OR(20w, 30w, 40w) channel lower 25%.
    Union of all channel window sizes + compressed.
    Tests if the union of all window lengths adds coverage beyond 30w alone."""
    if tw is None or len(tw) < 40:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    has_disc = (_tsla_below_52wk_pct(tsla, i, pct=0.10) or
                _tsla_below_20d_high_pct(tsla, i, pct=0.05))
    if not has_disc:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    # Any of the channel windows fires
    for window in (20, 30, 40):
        wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(float(tw['close'].iloc[wk_idx]), ch, 0.25):
                return 1
    return 0


SIGNALS_P10O: List[Tuple] = [
    # Phase 10O — 30w convergence + 40w channel + 15w channel sweep (S901-S910)
    ('S901_s895_lag5pct',          sig_s901_s895_lag5pct,          10, 0.20, 50),
    ('S902_s895_vix_pct60',        sig_s902_s895_vix_pct60,        10, 0.20, 50),
    ('S903_s215_40w_channel',      sig_s903_s215_40w_channel,       10, 0.20, 50),
    ('S904_s903_compressed',       sig_s904_s903_compressed,        10, 0.20, 50),
    ('S905_s903_disc_or',          sig_s905_s903_disc_or,           10, 0.20, 50),
    ('S906_s905_compressed',       sig_s906_s905_compressed,        10, 0.20, 50),
    ('S907_s887_b50ma_compressed', sig_s907_s887_below50ma_compressed, 10, 0.20, 50),
    ('S908_s215_15w_channel',      sig_s908_s215_15w_channel,       10, 0.20, 50),
    ('S909_s908_disc_compressed',  sig_s909_s908_disc_compressed,   10, 0.20, 50),
    ('S910_milestone_chan_sweep',   sig_s910_milestone_channel_sweep,10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O)


# ── Phase 10P — S910 extensions + dual-channel AND + 50w channel (S911-S920) ──

def sig_s911_s910_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S911: S910 (OR(20w/30w/40w)+compressed) + lag5pct.
    Channel union + TSLA lagging SPY ≥5% over 20 days.
    Hypothesis: 100% WR path — multi-scale structural + momentum lag."""
    if sig_s910_milestone_channel_sweep(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s912_s910_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S912: S910 (OR(20w/30w/40w)+compressed) + VIX pct>70.
    Elevated fear + multi-scale channel support + compression.
    Hypothesis: WR > 94%, fear confirms oversold at structural support."""
    if sig_s910_milestone_channel_sweep(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s913_dual_channel_and(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S913: Dual-channel AND — in lower 25% of BOTH 20w AND 30w channel.
    Structural confluence: price satisfying two channel definitions simultaneously.
    Hypothesis: tighter precision filter → WR > 94% vs either alone (WR=94%)."""
    if tw is None or len(tw) < 30:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 30:
        return 0
    close_w = float(tw['close'].iloc[wk_idx])
    # Both 20w AND 30w channel must see price in lower 25%
    ch20 = _channel_at(tw.iloc[wk_idx - 20:wk_idx])
    if ch20 is None or not _near_lower(close_w, ch20, 0.25):
        return 0
    ch30 = _channel_at(tw.iloc[wk_idx - 30:wk_idx])
    if ch30 is None or not _near_lower(close_w, ch30, 0.25):
        return 0
    return 1


def sig_s914_s913_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S914: Dual-channel AND + ATR compression.
    Triple structural gate: 20w channel + 30w channel + squeeze.
    Hypothesis: WR ≥ 95%, each gate independently valid."""
    if sig_s913_dual_channel_and(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s915_s887_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S915: 30w channel + volume dry-up (<70% of 20d avg).
    Volume dry-up at 30-week structural support — no panic selling, quiet accumulation.
    Compare: S495 (20w+dryup) moderate, S495+compressed = 100% WR on 20w."""
    if sig_s887_s215_30w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s916_s890_vol_dryup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S916: S890 (30w+compressed) + volume dry-up.
    Triple gate: 30w structural support + ATR squeeze + quiet volume.
    Hypothesis: 100% WR — the quietest compressed bounces at structural support always win."""
    if sig_s890_milestone_s888_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _volume_below_avg(tsla, i, mult=0.70) else 0


def sig_s917_s215_50w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S917: 50-week channel lower 25% + VIX 18-50.
    Longest structural channel tested — nearly 1 year of weekly data.
    Hypothesis: very rare, near-perfect structural support, WR > 30w."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 50:
        return 0
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    ch = _channel_at(tw.iloc[wk_idx - 50:wk_idx])
    if ch is None:
        return 0
    return 1 if _near_lower(float(tw['close'].iloc[wk_idx]), ch, 0.25) else 0


def sig_s918_s917_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S918: 50-week channel + ATR compression.
    Very long structural support + squeeze — rarest structural quality gate.
    Hypothesis: WR ≥ 94%, fewer trades than 30w (n ≈ 10-14)."""
    if sig_s917_s215_50w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s919_s910_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S919: S910 (channel union + compressed) + below 50MA.
    Tests if below50MA is redundant on channel union (already redundant on 30w alone).
    Hypothesis: S919 = S910 (all channel-union+compressed trades already below 50MA)."""
    if sig_s910_milestone_channel_sweep(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


def sig_s920_milestone_s914_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S920 MILESTONE: Dual-channel AND + compressed + lag5pct.
    Ultimate precision signal: 20w+30w channel both active + squeeze + lag.
    Hypothesis: 100% WR — hardest gate combination ever tested."""
    if sig_s914_s913_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


SIGNALS_P10P: List[Tuple] = [
    # Phase 10P — S910 extensions + dual-channel AND + 50w channel (S911-S920)
    ('S911_s910_lag5pct',          sig_s911_s910_lag5pct,           10, 0.20, 50),
    ('S912_s910_vix_pct70',        sig_s912_s910_vix_pct70,         10, 0.20, 50),
    ('S913_dual_channel_and',      sig_s913_dual_channel_and,       10, 0.20, 50),
    ('S914_s913_compressed',       sig_s914_s913_compressed,        10, 0.20, 50),
    ('S915_s887_vol_dryup',        sig_s915_s887_vol_dryup,         10, 0.20, 50),
    ('S916_s890_vol_dryup',        sig_s916_s890_vol_dryup,         10, 0.20, 50),
    ('S917_s215_50w_channel',      sig_s917_s215_50w_channel,       10, 0.20, 50),
    ('S918_s917_compressed',       sig_s918_s917_compressed,        10, 0.20, 50),
    ('S919_s910_below50ma',        sig_s919_s910_below50ma,         10, 0.20, 50),
    ('S920_milestone_dual_lag',    sig_s920_milestone_s914_lag5pct, 10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P)


# ── Phase 10Q — 25w/35w gap fill + tight 30w + triple-AND + 50w union (S921-S930) ──

def sig_s921_s918_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S921: S918 (50w+compressed) + lag5pct.
    50w structural support + squeeze + TSLA momentum lag.
    Hypothesis: 100% WR — matches pattern from all compressed+lag paths."""
    if sig_s918_s917_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s922_s918_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S922: S918 (50w+compressed) + VIX pct>60.
    50w structural support + squeeze + elevated VIX (any fear).
    Compare: S892 (30w+compressed+VIX_pct60) = WR=91%, 7/7yr."""
    if sig_s918_s917_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s923_s215_25w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S923: 25-week channel lower 25% + VIX 18-50.
    Gap fill between 20w (S215 base) and 30w (S887).
    Hypothesis: WR between 20w (noisy) and 30w (WR=94% base)."""
    if tw is None or len(tw) < 25:
        return 0
    daily_date = tsla.index[i]
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 25:
        return 0
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    ch = _channel_at(tw.iloc[wk_idx - 25:wk_idx])
    if ch is None:
        return 0
    return 1 if _near_lower(float(tw['close'].iloc[wk_idx]), ch, 0.25) else 0


def sig_s924_s923_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S924: 25-week channel + ATR compression.
    25w structural support + squeeze — gap fill between S224 (20w) and S890 (30w).
    Hypothesis: WR matches or exceeds 30w (WR=94%)."""
    if sig_s923_s215_25w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s925_s215_35w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S925: 35-week channel lower 25% + VIX 18-50.
    Gap fill between 30w (S887) and 40w (S903).
    Hypothesis: WR similar to 30w (both give WR=94% with compression)."""
    if tw is None or len(tw) < 35:
        return 0
    daily_date = tsla.index[i]
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 35:
        return 0
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    ch = _channel_at(tw.iloc[wk_idx - 35:wk_idx])
    if ch is None:
        return 0
    return 1 if _near_lower(float(tw['close'].iloc[wk_idx]), ch, 0.25) else 0


def sig_s926_s925_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S926: 35-week channel + ATR compression.
    35w structural support + squeeze — gap fill between S890 (30w) and S904 (40w).
    Hypothesis: WR=94%, trade count between S890 (n=18) and S904 (n=15)."""
    if sig_s925_s215_35w_channel(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s927_s887_tight10(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S927: 30w channel bottom 10% + VIX 18-50.
    Very tight position in 30-week channel (bottom 10% of band width).
    Compare: S861 (20w tight10) = WR=90%; 30w should be equal or better.
    Hypothesis: WR=90-95% on 30w tight position."""
    if tw is None or len(tw) < 30:
        return 0
    daily_date = tsla.index[i]
    pos = _weekly_channel_pos(tw, daily_date, window=30)
    if pos < 0:
        return 0
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    return 1 if pos < 0.10 else 0


def sig_s928_s927_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S928: 30w channel bottom 10% + ATR compression.
    Extreme position precision (bottom 10%) + squeeze — very tight gate.
    Compare: S875 (20w tight15+compressed) = WR=92%; S871 (20w tight10+compressed) = 100% WR, 5/5yr.
    Hypothesis: WR ≥ 94%, very few trades."""
    if sig_s927_s887_tight10(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S929: OR(20w/30w/40w/50w) channel + compressed — extended S910 to include 50w.
    Maximum coverage: any of 4 structural windows fires.
    Hypothesis: adds 1-2 more trades vs S910 (n=20) while maintaining WR=94%+."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s930_milestone_triple_and_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S930 MILESTONE: Triple-AND (20w+30w+40w all lower 25%) + compressed.
    Ultimate structural confluence: ALL THREE channel windows fire simultaneously.
    More selective than dual-AND (S913/S914). Hypothesis: WR=94%+, very few trades."""
    if tw is None or len(tw) < 40:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 40:
        return 0
    close_w = float(tw['close'].iloc[wk_idx])
    # ALL THREE windows must see price in lower 25%
    for window in (20, 30, 40):
        ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
        if ch is None or not _near_lower(close_w, ch, 0.25):
            return 0
    return 1


SIGNALS_P10Q: List[Tuple] = [
    # Phase 10Q — 25w/35w gap fill + tight 30w + triple-AND + 50w union (S921-S930)
    ('S921_s918_lag5pct',          sig_s921_s918_lag5pct,           10, 0.20, 50),
    ('S922_s918_vix_pct60',        sig_s922_s918_vix_pct60,         10, 0.20, 50),
    ('S923_s215_25w_channel',      sig_s923_s215_25w_channel,       10, 0.20, 50),
    ('S924_s923_compressed',       sig_s924_s923_compressed,        10, 0.20, 50),
    ('S925_s215_35w_channel',      sig_s925_s215_35w_channel,       10, 0.20, 50),
    ('S926_s925_compressed',       sig_s926_s925_compressed,        10, 0.20, 50),
    ('S927_s887_tight10',          sig_s927_s887_tight10,           10, 0.20, 50),
    ('S928_s927_compressed',       sig_s928_s927_compressed,        10, 0.20, 50),
    ('S929_s215_4chan_union',       sig_s929_s215_4chan_union,       10, 0.20, 50),
    ('S930_milestone_triple_and',  sig_s930_milestone_triple_and_compressed, 10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q)


# ── Phase 10R — S929 extensions + granular window sweeps (S931-S940) ──────────

def sig_s931_s929_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S931: S929 (OR(20/30/40/50w)+compressed) + lag5pct.
    4-chan union + momentum lag — testing 100% WR on the champion signal.
    Hypothesis: 100% WR, n ≈ 17-18, best P&L + WR combo."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s932_s929_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S932: S929 (4-chan union+compressed) + VIX pct>60.
    4-chan champion + elevated VIX (any fear).
    Hypothesis: WR > 94%, n < 21."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s933_s929_vix_pct70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S933: S929 (4-chan union+compressed) + VIX pct>70.
    4-chan champion + high VIX fear.
    Hypothesis: WR ≥ 95%, very selective."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.70) else 0


def sig_s934_s929_tighter20pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S934: 4-chan union + compressed, tighter position (bottom 20% of channel band).
    Tests if tightening from 25%→20% fraction improves WR on S929.
    Hypothesis: slightly fewer trades, maybe WR=95%+."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.20):
                return 1
    return 0


def sig_s935_s215_5chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S935: OR(20/25/30/35/40w) channel + compressed — 5 windows, no 50w.
    Granular coverage in the 20-40w range.
    Compare to S910 (20/30/40w, n=20) — adding 25w and 35w gaps."""
    if tw is None or len(tw) < 40:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 25, 30, 35, 40):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s936_s215_6chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S936: OR(20/25/30/35/40/50w) channel + compressed — 6 windows, maximum coverage.
    Full granular sweep from 20w to 50w.
    Hypothesis: adds further coverage vs S929 (4 windows), WR should be ≥ 94%."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 25, 30, 35, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s937_s924_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S937: S924 (25w+compressed) + lag5pct.
    25w structural support + squeeze + TSLA momentum lag.
    Hypothesis: 100% WR — same pattern as all compressed+lag paths."""
    if sig_s924_s923_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s938_s926_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S938: S926 (35w+compressed) + lag5pct.
    35w structural support + squeeze + TSLA momentum lag.
    Hypothesis: 100% WR — same convergence as 20w/30w/40w+lag5pct paths."""
    if sig_s926_s925_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s939_s929_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S939: S929 (4-chan union+compressed) + below 50MA.
    Convergence check: expect S939=S929 (below50MA redundant on all channel+compressed paths).
    Hypothesis: same n=21, confirming redundancy of 50MA filter."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 0 if _tsla_above_50ma(tsla, i) else 1


def sig_s940_milestone_s931_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S940 MILESTONE: S929 + lag5pct + VIX_pct60 — triple precision on champion signal.
    4-chan union + compressed + TSLA lag + elevated VIX.
    Hypothesis: 100% WR, very selective, n ≈ 10-13."""
    if sig_s931_s929_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


SIGNALS_P10R: List[Tuple] = [
    # Phase 10R — S929 extensions + granular window sweeps (S931-S940)
    ('S931_s929_lag5pct',          sig_s931_s929_lag5pct,           10, 0.20, 50),
    ('S932_s929_vix_pct60',        sig_s932_s929_vix_pct60,         10, 0.20, 50),
    ('S933_s929_vix_pct70',        sig_s933_s929_vix_pct70,         10, 0.20, 50),
    ('S934_s929_tighter20pct',     sig_s934_s929_tighter20pct,      10, 0.20, 50),
    ('S935_s215_5chan_union',       sig_s935_s215_5chan_union,        10, 0.20, 50),
    ('S936_s215_6chan_union',       sig_s936_s215_6chan_union,        10, 0.20, 50),
    ('S937_s924_lag5pct',          sig_s937_s924_lag5pct,           10, 0.20, 50),
    ('S938_s926_lag5pct',          sig_s938_s926_lag5pct,           10, 0.20, 50),
    ('S939_s929_below50ma',        sig_s939_s929_below50ma,         10, 0.20, 50),
    ('S940_milestone_s931_vix',    sig_s940_milestone_s931_vix,     10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R)


# ── Phase 10S — ATR ratio + lag threshold sweep + S929 position variants (S941-S950) ──

def sig_s941_s929_atr65(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S941: 4-chan union (S929 base logic) with tighter ATR ratio 0.65.
    Tighter compression threshold: atr_5 < 0.65 * atr_20 (vs 0.75 in S929).
    Hypothesis: fewer trades, possibly higher WR — more extreme squeeze required."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.65 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s942_s929_atr85(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S942: 4-chan union with looser ATR ratio 0.85.
    Looser compression: atr_5 < 0.85 * atr_20 — captures more mildly-compressed setups.
    Hypothesis: more trades, lower WR — tests optimal threshold."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.85 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s943_s890_lag3pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S943: S890 (30w+compressed) + lag3pct (3% underperformance — more permissive).
    Tests if lag threshold matters: lag3pct vs lag5pct (current S891).
    Hypothesis: more trades than S891 (lag5pct), similar or slightly lower WR."""
    if sig_s890_milestone_s888_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.03) else 0


def sig_s944_s890_lag8pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S944: S890 (30w+compressed) + lag8pct (8% underperformance — stricter).
    Tests stricter lag: lag8pct vs lag5pct (current S891).
    Hypothesis: fewer trades than S891, possibly higher or same WR."""
    if sig_s890_milestone_s888_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.08) else 0


def sig_s945_s929_lag3pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S945: S929 (4-chan union+compressed) + lag3pct.
    More permissive lag on champion signal.
    Hypothesis: n > 17 (vs S931 n=17 with lag5pct), WR might be slightly lower."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.03) else 0


def sig_s946_s929_lag8pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S946: S929 (4-chan union+compressed) + lag8pct.
    Stricter lag on champion signal.
    Hypothesis: n < 17 (vs S931 n=17), possibly 100% WR maintained."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.08) else 0


def sig_s947_s929_frac30pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S947: 4-chan union + compressed with looser position (bottom 30% of channel).
    More permissive position gate: 30% fraction vs 25% in S929.
    Hypothesis: more trades (n > 21), WR might dip slightly below 95%."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.30):
                return 1
    return 0


def sig_s948_s929_lag10pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S948: S929 (4-chan union+compressed) + lag10pct (10% underperformance — very strict).
    Maximum lag filter — TSLA dramatically underperforming SPY.
    Hypothesis: very few trades but perfect WR."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.10) else 0


def sig_s949_s929_lag5pct_vix50(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S949: S929 + lag5pct + VIX pct>50 (any mild elevation).
    More permissive VIX threshold (pct>50 = median) on S931.
    Hypothesis: n > S940 (pct>60), WR = 100%."""
    if sig_s931_s929_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.50) else 0


def sig_s950_milestone_s929_no_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S950 MILESTONE: 4-chan union + compressed with NO VIX filter.
    Tests S929 without the VIX 18-50 gate — does VIX filter help or hurt?
    Hypothesis: same or slightly more trades, VIX filter might be redundant."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


SIGNALS_P10S: List[Tuple] = [
    # Phase 10S — ATR ratio + lag threshold sweep + S929 position variants (S941-S950)
    ('S941_s929_atr65',            sig_s941_s929_atr65,             10, 0.20, 50),
    ('S942_s929_atr85',            sig_s942_s929_atr85,             10, 0.20, 50),
    ('S943_s890_lag3pct',          sig_s943_s890_lag3pct,           10, 0.20, 50),
    ('S944_s890_lag8pct',          sig_s944_s890_lag8pct,           10, 0.20, 50),
    ('S945_s929_lag3pct',          sig_s945_s929_lag3pct,           10, 0.20, 50),
    ('S946_s929_lag8pct',          sig_s946_s929_lag8pct,           10, 0.20, 50),
    ('S947_s929_frac30pct',        sig_s947_s929_frac30pct,         10, 0.20, 50),
    ('S948_s929_lag10pct',         sig_s948_s929_lag10pct,          10, 0.20, 50),
    ('S949_s929_lag5_vix50',       sig_s949_s929_lag5pct_vix50,     10, 0.20, 50),
    ('S950_milestone_s929_novix',  sig_s950_milestone_s929_no_vix,  10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R + SIGNALS_P10S)


# ── Phase 10T — lag lookback sweep + SPY context + selloff + range-contraction (S951-S960) ──

def sig_s951_s929_lag10d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S951: S929 + lag5pct with SHORTER lookback=10 days.
    Same 5% underperformance but measured over 10 days instead of 20.
    Hypothesis: more trades (shorter window less strict), WR might be lower than S931."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=10, lag=0.05) else 0


def sig_s952_s929_lag30d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S952: S929 + lag5pct with LONGER lookback=30 days.
    Same 5% underperformance but measured over 30 days — more persistent lag.
    Hypothesis: fewer trades, possibly higher WR than S931 (20d lookback)."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=30, lag=0.05) else 0


def sig_s953_s929_spy_above50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S953: S929 + SPY above 50MA (bullish macro context).
    Test if market uptrend context improves bounce quality.
    Hypothesis: neutral or anti-signal (best bounces happen in any market)."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    spy_close = float(spy['close'].iloc[i])
    if i < 50:
        return 1
    spy_ma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close > spy_ma50 else 0


def sig_s954_s929_spy_below50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S954: S929 + SPY below 50MA (bearish macro = more fear = better bounce?).
    Test if market downtrend context improves quality.
    Hypothesis: possibly anti-signal (SPY downtrend = market still falling)."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    spy_close = float(spy['close'].iloc[i])
    if i < 50:
        return 0
    spy_ma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if spy_close <= spy_ma50 else 0


def sig_s955_s929_tsla_3d_selloff(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S955: S929 + TSLA 3-day return < -5% (recent sharp selloff).
    Price at structural support after acute 3-day drop.
    Hypothesis: improves quality — acute panic before bounce = stronger reversal."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if c_3d <= 0:
        return 0
    return 1 if (c_now - c_3d) / c_3d < -0.05 else 0


def sig_s956_s929_tsla_5d_selloff(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S956: S929 + TSLA 5-day return < -8% (5-day sharp selloff).
    Price at structural support after 5-day drop exceeding 8%.
    Hypothesis: capitulation + structural support = high conviction bounce."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    ret = _tsla_5d_return(tsla, i)
    return 1 if ret < -0.08 else 0


def sig_s957_s929_range_contraction(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S957: S929 + 3-day bar range contraction (each day's H-L range narrows).
    Consecutive day range narrowing = coiling spring pattern.
    Hypothesis: more specific than ATR compression — 3 days of narrowing."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 1
    r0 = float(tsla['high'].iloc[i]) - float(tsla['low'].iloc[i])
    r1 = float(tsla['high'].iloc[i-1]) - float(tsla['low'].iloc[i-1])
    r2 = float(tsla['high'].iloc[i-2]) - float(tsla['low'].iloc[i-2])
    if r0 <= 0:
        return 1
    # Three consecutive narrowing bars
    return 1 if r0 < r1 < r2 else 0


def sig_s958_s950_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S958: S950 (4-chan union+compressed, NO VIX filter) + lag5pct.
    Tests if no-VIX champion + lag = same as S931 (with VIX filter + lag).
    Hypothesis: S958 ≥ S931 (no VIX filter lets in more lag5pct trades)."""
    if sig_s950_milestone_s929_no_vix(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s959_s929_spy_5d_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S959: S929 + SPY 5-day return positive (market recovering while TSLA lagged).
    Market bounce started, TSLA hasn't followed yet.
    Hypothesis: SPY already recovering = TSLA catch-up trade."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_5d = float(spy['close'].iloc[i - 5])
    if spy_5d <= 0:
        return 0
    return 1 if spy_now > spy_5d else 0


def sig_s960_milestone_s931_spy_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S960 MILESTONE: S931 (4-chan+compressed+lag5pct) + SPY 5-day positive.
    Ultimate combo: structural support + squeeze + lag + market recovering.
    Hypothesis: highest avg/trade — TSLA lagging recovering market at structural support."""
    if sig_s931_s929_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_5d = float(spy['close'].iloc[i - 5])
    if spy_5d <= 0:
        return 0
    return 1 if spy_now > spy_5d else 0


SIGNALS_P10T: List[Tuple] = [
    # Phase 10T — lag lookback sweep + SPY context + selloff + range-contraction (S951-S960)
    ('S951_s929_lag10d',           sig_s951_s929_lag10d,            10, 0.20, 50),
    ('S952_s929_lag30d',           sig_s952_s929_lag30d,            10, 0.20, 50),
    ('S953_s929_spy_above50ma',    sig_s953_s929_spy_above50ma,     10, 0.20, 50),
    ('S954_s929_spy_below50ma',    sig_s954_s929_spy_below50ma,     10, 0.20, 50),
    ('S955_s929_3d_selloff',       sig_s955_s929_tsla_3d_selloff,   10, 0.20, 50),
    ('S956_s929_5d_selloff8pct',   sig_s956_s929_tsla_5d_selloff,   10, 0.20, 50),
    ('S957_s929_range_contract',   sig_s957_s929_range_contraction, 10, 0.20, 50),
    ('S958_s950_lag5pct',          sig_s958_s950_lag5pct,           10, 0.20, 50),
    ('S959_s929_spy_5d_up',        sig_s959_s929_spy_5d_up,         10, 0.20, 50),
    ('S960_milestone_s931_spyup',  sig_s960_milestone_s931_spy_up,  10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R + SIGNALS_P10S + SIGNALS_P10T)


# ── Phase 10U — S955 extensions + SPY recovery + 20d low + selloff sweep (S961-S970) ──

def sig_s961_s955_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S961: S955 (S929 + 3d selloff) + lag5pct.
    Combines two 100% WR conditions: recent sharp selloff + momentum lag.
    Hypothesis: 100% WR, very few but ultra-high-quality trades."""
    if sig_s955_s929_tsla_3d_selloff(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s962_s955_vix_pct60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S962: S955 (S929 + 3d selloff) + VIX pct>60.
    Acute selloff at structural support + elevated VIX (external fear).
    Hypothesis: 100% WR, fewer trades than S955 (n=9)."""
    if sig_s955_s929_tsla_3d_selloff(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _vix_elevated_pct(vix, i, window=252, pct=0.60) else 0


def sig_s963_s929_3d_selloff3pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S963: S929 + 3-day return < -3% (more permissive selloff threshold).
    Compare to S955 (< -5%): does a shallower 3-day drop still indicate quality bounce?
    Hypothesis: more trades than S955, slightly lower WR."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if c_3d <= 0:
        return 0
    return 1 if (c_now - c_3d) / c_3d < -0.03 else 0


def sig_s964_s929_spy_5d_up2pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S964: S929 + SPY 5-day return > +2% (meaningful market recovery).
    Market meaningfully recovered from recent lows while TSLA still lagged.
    Compare to S959 (any positive 5d SPY return): does a 2%+ threshold matter?"""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_5d = float(spy['close'].iloc[i - 5])
    if spy_5d <= 0:
        return 0
    return 1 if (spy_now - spy_5d) / spy_5d > 0.02 else 0


def sig_s965_s931_3d_selloff(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S965: S931 (S929+lag5pct) + 3-day selloff < -5%.
    Triple condition: structural support + squeeze + lag + acute panic.
    Hypothesis: 100% WR, fewer than S955's 9 but highest individual trade quality."""
    if sig_s931_s929_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if c_3d <= 0:
        return 0
    return 1 if (c_now - c_3d) / c_3d < -0.05 else 0


def sig_s966_s929_new_10d_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S966: S929 + today's close is a new 10-day low.
    Fresh capitulation low at structural support = forced seller exhaustion.
    Hypothesis: WR ≥ 94%, fewer trades — new low = maximum fear at support."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 0
    c_now = float(tsla['close'].iloc[i])
    prior_min = float(tsla['close'].iloc[i - 10:i].min())
    return 1 if c_now <= prior_min else 0


def sig_s967_s931_new_10d_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S967: S931 (4-chan+compressed+lag5pct) + new 10-day low.
    Lag + new low at structural support = extremely oversold quality bounce.
    Hypothesis: 100% WR, very few but extraordinary setups."""
    if sig_s931_s929_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 0
    c_now = float(tsla['close'].iloc[i])
    prior_min = float(tsla['close'].iloc[i - 10:i].min())
    return 1 if c_now <= prior_min else 0


def sig_s968_s929_spy_20d_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S968: S929 + SPY 20-day return positive (longer market recovery).
    SPY has recovered over 20 days while TSLA hasn't — longer divergence.
    Hypothesis: WR ≥ 94%, covers more of the lag5pct scenarios."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_20d = float(spy['close'].iloc[i - 20])
    if spy_20d <= 0:
        return 0
    return 1 if spy_now > spy_20d else 0


def sig_s969_s929_yesterday_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S969: S929 + TSLA closed down yesterday (still selling pressure).
    Price at structural support after another down day = fresh seller capitulation.
    Hypothesis: more selective version of S929, higher WR."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 0
    c_yesterday = float(tsla['close'].iloc[i - 1])
    o_yesterday = float(tsla['open'].iloc[i - 1])
    return 1 if c_yesterday < o_yesterday else 0


def sig_s970_milestone_s929_2d_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S970 MILESTONE: S929 + 2 consecutive down days (yesterday AND day before).
    Two straight red days at structural support = persistent selling before bounce.
    Hypothesis: WR ≥ 95%, the 2-day selling exhaustion sets up the cleanest bounces."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 2:
        return 0
    c1 = float(tsla['close'].iloc[i - 1])
    o1 = float(tsla['open'].iloc[i - 1])
    c2 = float(tsla['close'].iloc[i - 2])
    o2 = float(tsla['open'].iloc[i - 2])
    return 1 if (c1 < o1 and c2 < o2) else 0


SIGNALS_P10U: List[Tuple] = [
    # Phase 10U — S955 extensions + SPY recovery + new 10d low + selloff sweep (S961-S970)
    ('S961_s955_lag5pct',          sig_s961_s955_lag5pct,           10, 0.20, 50),
    ('S962_s955_vix_pct60',        sig_s962_s955_vix_pct60,         10, 0.20, 50),
    ('S963_s929_3d_selloff3pct',   sig_s963_s929_3d_selloff3pct,    10, 0.20, 50),
    ('S964_s929_spy5d_up2pct',     sig_s964_s929_spy_5d_up2pct,     10, 0.20, 50),
    ('S965_s931_3d_selloff',       sig_s965_s931_3d_selloff,        10, 0.20, 50),
    ('S966_s929_new_10d_low',      sig_s966_s929_new_10d_low,       10, 0.20, 50),
    ('S967_s931_new_10d_low',      sig_s967_s931_new_10d_low,       10, 0.20, 50),
    ('S968_s929_spy_20d_up',       sig_s968_s929_spy_20d_up,        10, 0.20, 50),
    ('S969_s929_yesterday_down',   sig_s969_s929_yesterday_down,    10, 0.20, 50),
    ('S970_milestone_s929_2ddown', sig_s970_milestone_s929_2d_down, 10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R + SIGNALS_P10S + SIGNALS_P10T
           + SIGNALS_P10U)


# ── Phase 10V — S963/S968 extensions + combined conditions + SPY window sweep (S971-S980) ──

def sig_s971_s963_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S971: S963 (S929+3d-selloff3%) + lag5pct.
    3-day acute selloff + 4-chan structural support + squeeze + TSLA lagging.
    Hypothesis: 100% WR, fewer than S963 (n=11)."""
    if sig_s963_s929_3d_selloff3pct(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s972_s968_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S972: S968 (S929+SPY_20d_up) + lag5pct.
    Market recovering (SPY 20d up) + TSLA lagging SPY 5% over 20 days.
    Hypothesis: 100% WR — SPY up + TSLA lagging = maximally strong catch-up setup."""
    if sig_s968_s929_spy_20d_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s973_s929_selloff_and_spy_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S973: S929 + BOTH 3d selloff (-3%) AND SPY 20d up.
    Dual condition: TSLA dropped sharply while market has been rising — extreme divergence.
    Hypothesis: 100% WR, very few but highest individual quality."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # 3-day TSLA selloff
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if c_3d <= 0 or (c_now - c_3d) / c_3d >= -0.03:
        return 0
    # SPY 20-day positive
    if i < 20:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_20d = float(spy['close'].iloc[i - 20])
    if spy_20d <= 0 or spy_now <= spy_20d:
        return 0
    return 1


def sig_s974_s929_spy_10d_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S974: S929 + SPY 10-day return positive.
    Shorter SPY recovery window (10d vs 20d in S968).
    Hypothesis: slightly more trades than S968 (n=11), possibly lower WR."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_10d = float(spy['close'].iloc[i - 10])
    if spy_10d <= 0:
        return 0
    return 1 if spy_now > spy_10d else 0


def sig_s975_s929_spy_30d_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S975: S929 + SPY 30-day return positive.
    Longer SPY recovery (30d vs 20d in S968).
    Hypothesis: similar to S968 or slightly fewer trades."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 30:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_30d = float(spy['close'].iloc[i - 30])
    if spy_30d <= 0:
        return 0
    return 1 if spy_now > spy_30d else 0


def sig_s976_s929_spy_20d_up3pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S976: S929 + SPY 20d return > +3% (more meaningful market recovery).
    Compare to S968 (any positive 20d SPY): does requiring 3%+ matter?
    Hypothesis: fewer trades but same 100% WR."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_20d = float(spy['close'].iloc[i - 20])
    if spy_20d <= 0:
        return 0
    return 1 if (spy_now - spy_20d) / spy_20d > 0.03 else 0


def sig_s977_s963_spy_20d_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S977: S963 (3d selloff 3%) + SPY 20d up.
    Tests if S973 has different behavior vs S963+S968 separately.
    Essentially same as S973 (3d selloff AND SPY 20d up)."""
    return sig_s973_s929_selloff_and_spy_up(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s978_s968_3d_selloff(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S978: S968 (SPY 20d up) + 3d selloff 5%.
    Same as S973 but with stricter 5% selloff threshold.
    Tests: does the stricter selloff (5% vs 3%) reduce trades on S968?"""
    if sig_s968_s929_spy_20d_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if c_3d <= 0:
        return 0
    return 1 if (c_now - c_3d) / c_3d < -0.05 else 0


def sig_s979_s963_or_s968(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S979: S963 OR S968 — union of both 100% WR paths.
    Maximizes coverage: fires on 3d selloff OR SPY 20d up (or both).
    Hypothesis: n ≥ 11 (might overlap), WR should still be very high."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # Condition 1: 3-day TSLA selloff ≥ 3%
    has_selloff = False
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            has_selloff = True
    # Condition 2: SPY 20d up
    has_spy_up = False
    if i >= 20:
        spy_now = float(spy['close'].iloc[i])
        spy_20d = float(spy['close'].iloc[i - 20])
        if spy_20d > 0 and spy_now > spy_20d:
            has_spy_up = True
    return 1 if (has_selloff or has_spy_up) else 0


def sig_s980_milestone_s973_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S980 MILESTONE: S973 (3d selloff + SPY 20d up) + lag5pct.
    Triple timing confluence: acute panic + market recovering + TSLA lagging.
    Hypothesis: 100% WR, very few but highest conviction trades."""
    if sig_s973_s929_selloff_and_spy_up(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


SIGNALS_P10V: List[Tuple] = [
    # Phase 10V — S963/S968 extensions + combined + SPY window sweep (S971-S980)
    ('S971_s963_lag5pct',          sig_s971_s963_lag5pct,           10, 0.20, 50),
    ('S972_s968_lag5pct',          sig_s972_s968_lag5pct,           10, 0.20, 50),
    ('S973_s929_selloff_spy_up',   sig_s973_s929_selloff_and_spy_up,10, 0.20, 50),
    ('S974_s929_spy10d_up',        sig_s974_s929_spy_10d_up,        10, 0.20, 50),
    ('S975_s929_spy30d_up',        sig_s975_s929_spy_30d_up,        10, 0.20, 50),
    ('S976_s929_spy20d_up3pct',    sig_s976_s929_spy_20d_up3pct,    10, 0.20, 50),
    ('S977_s963_spy20d',           sig_s977_s963_spy_20d_up,        10, 0.20, 50),
    ('S978_s968_3d_selloff5pct',   sig_s978_s968_3d_selloff,        10, 0.20, 50),
    ('S979_s963_or_s968',          sig_s979_s963_or_s968,           10, 0.20, 50),
    ('S980_milestone_s973_lag',    sig_s980_milestone_s973_lag,     10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R + SIGNALS_P10S + SIGNALS_P10T
           + SIGNALS_P10U + SIGNALS_P10V)


# ── Phase 10W — Maximum 100% WR union + RSI + intraday close strength (S981-S990) ──

def sig_s981_s931_or_s979(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S981: S931 OR S979 — union of all known 100% WR conditions on S929.
    Fires if S929 AND (lag5pct OR 3d-selloff3% OR SPY-20d-up).
    Hypothesis: 100% WR with n=18-20 — the maximum 100% WR subset of S929."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # Condition 1: TSLA lagging SPY by ≥5% over 20 days
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    # Condition 2: 3-day TSLA selloff ≥ 3%
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    # Condition 3: SPY 20-day positive (market recovering)
    if i >= 20:
        spy_now = float(spy['close'].iloc[i])
        spy_20d = float(spy['close'].iloc[i - 20])
        if spy_20d > 0 and spy_now > spy_20d:
            return 1
    return 0


def sig_s982_s929_close_above_mid(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S982: S929 + close in upper 50% of day's range (not panic close).
    Intraday close strength: close >= midpoint means buyers absorbed selling.
    Hypothesis: slightly higher WR — strong close = buyers in control."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    h = float(tsla['high'].iloc[i])
    l = float(tsla['low'].iloc[i])
    c = float(tsla['close'].iloc[i])
    if h <= l:
        return 1
    return 1 if c >= (h + l) / 2 else 0


def sig_s983_s929_close_below_mid(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S983: S929 + close in lower 50% of day's range (weak/panic close).
    Tests if weak intraday close (still selling at end of day) matters.
    Hypothesis: anti-signal or neutral — might be better (shakeout before reversal)."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    h = float(tsla['high'].iloc[i])
    l = float(tsla['low'].iloc[i])
    c = float(tsla['close'].iloc[i])
    if h <= l:
        return 1
    return 1 if c < (h + l) / 2 else 0


def sig_s984_s929_rsi_lt30(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S984: S929 + RSI < 30 (deeply oversold).
    RSI is passed as 'rt' (pre-computed series).
    Hypothesis: RSI<30 is too deep oversold — might indicate broken trend, anti-signal."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi = float(rt.iloc[i])
    return 1 if rsi < 30 else 0


def sig_s985_s929_rsi_30_50(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S985: S929 + RSI between 30 and 50 (pullback but not extreme).
    Tests if moderate RSI oversold (not panic) is the quality zone.
    Hypothesis: WR > S929 — moderate pullback to structural support = best entries."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi = float(rt.iloc[i])
    return 1 if 30 <= rsi < 50 else 0


def sig_s986_s981_rsi_lt50(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S986: S981 (max 100% WR union) + RSI < 50 (not overbought).
    Tests if RSI gate further cleans the union path.
    Hypothesis: same count (all S981 trades likely have RSI<50 already)."""
    if sig_s981_s931_or_s979(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    rsi = float(rt.iloc[i])
    return 1 if rsi < 50 else 0


def sig_s987_s979_close_above_mid(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S987: S979 (100% WR union) + strong intraday close (close >= midpoint).
    Tests intraday close on the precision 100% WR path.
    Hypothesis: same or fewer trades, WR stays 100%."""
    if sig_s979_s963_or_s968(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    h = float(tsla['high'].iloc[i])
    l = float(tsla['low'].iloc[i])
    c = float(tsla['close'].iloc[i])
    if h <= l:
        return 1
    return 1 if c >= (h + l) / 2 else 0


def sig_s988_s929_weekly_low_week(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S988: S929 + price near weekly low (close within bottom 15% of current week's range).
    Within-week capitulation at structural support.
    Hypothesis: more precise timing — price at weekly lows within the setup."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # Find current week's high/low using daily data (look back up to 5 bars for Mon-Fri)
    # Get the Monday of the current week
    if i < 4:
        return 1
    daily_date = tsla.index[i]
    dow = daily_date.dayofweek  # 0=Mon, 4=Fri
    week_start = i - dow  # Go back to Monday
    if week_start < 0:
        week_start = 0
    week_high = float(tsla['high'].iloc[week_start:i + 1].max())
    week_low = float(tsla['low'].iloc[week_start:i + 1].min())
    c = float(tsla['close'].iloc[i])
    rng = week_high - week_low
    if rng <= 0:
        return 1
    # Close in bottom 15% of week's range
    return 1 if (c - week_low) / rng < 0.15 else 0


def sig_s989_s929_weekly_open_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S989: S929 + open is below prior day's close (gap down today).
    Gap-down open at structural support = fear-driven selling.
    Hypothesis: gap down = immediate capitulation signal, WR ≥ 95%."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 0
    o_today = float(tsla['open'].iloc[i])
    c_yesterday = float(tsla['close'].iloc[i - 1])
    return 1 if o_today < c_yesterday * 0.995 else 0  # gap down > 0.5%


def sig_s990_milestone_max_100pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S990 MILESTONE: Extended S981 adding gap-down condition.
    S929 + (lag5pct OR 3d-selloff OR SPY-20d-up OR gap-down).
    Hypothesis: 100% WR with n=19-20 — approaching maximum 100% WR set."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # Condition 1: lag5pct
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    # Condition 2: 3-day selloff ≥ 3%
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    # Condition 3: SPY 20-day positive
    if i >= 20:
        spy_now = float(spy['close'].iloc[i])
        spy_20d = float(spy['close'].iloc[i - 20])
        if spy_20d > 0 and spy_now > spy_20d:
            return 1
    # Condition 4: gap-down open (> 0.5% below prior close)
    if i >= 1:
        o_today = float(tsla['open'].iloc[i])
        c_yesterday = float(tsla['close'].iloc[i - 1])
        if o_today < c_yesterday * 0.995:
            return 1
    return 0


SIGNALS_P10W: List[Tuple] = [
    # Phase 10W — Maximum 100% WR union + RSI + intraday/weekly close (S981-S990)
    ('S981_s931_or_s979',          sig_s981_s931_or_s979,           10, 0.20, 50),
    ('S982_s929_close_above_mid',  sig_s982_s929_close_above_mid,   10, 0.20, 50),
    ('S983_s929_close_below_mid',  sig_s983_s929_close_below_mid,   10, 0.20, 50),
    ('S984_s929_rsi_lt30',         sig_s984_s929_rsi_lt30,          10, 0.20, 50),
    ('S985_s929_rsi_30_50',        sig_s985_s929_rsi_30_50,         10, 0.20, 50),
    ('S986_s981_rsi_lt50',         sig_s986_s981_rsi_lt50,          10, 0.20, 50),
    ('S987_s979_close_above_mid',  sig_s987_s979_close_above_mid,   10, 0.20, 50),
    ('S988_s929_weekly_low_week',  sig_s988_s929_weekly_low_week,   10, 0.20, 50),
    ('S989_s929_gap_down',         sig_s989_s929_weekly_open_down,  10, 0.20, 50),
    ('S990_milestone_max_100pct',  sig_s990_milestone_max_100pct,   10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R + SIGNALS_P10S + SIGNALS_P10T
           + SIGNALS_P10U + SIGNALS_P10V + SIGNALS_P10W)


# ── Phase 10X — Closing in on n=20 100% WR + SPY-10d variant + S999/S1000 milestones (S991-S1000) ──

def sig_s991_s929_or_lag_selloff_spy10d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S991: S929 + (lag5pct OR 3d-selloff OR SPY-10d-up).
    Variant of S981 using SPY-10d instead of SPY-20d.
    Hypothesis: catches a different subset — possibly finds the excluded winner."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    if i >= 10:
        spy_now = float(spy['close'].iloc[i])
        spy_10d = float(spy['close'].iloc[i - 10])
        if spy_10d > 0 and spy_now > spy_10d:
            return 1
    return 0


def sig_s992_s981_or_spy10d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S992: S981 extended with SPY-10d-up (4th condition).
    Adds SPY-10d positive on top of S981's 3 conditions.
    Hypothesis: catches the excluded winner → 100% WR, n=20."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    if i >= 10:
        spy_now = float(spy['close'].iloc[i])
        spy_10d = float(spy['close'].iloc[i - 10])
        if spy_10d > 0 and spy_now > spy_10d:
            return 1
    return 0


def sig_s993_s981_or_spy5d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S993: S981 + SPY-5d-up (short SPY window as 4th condition).
    Hypothesis: n=19 or 20, captures the excluded winner if SPY was up 5 days ago."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    return 0


def sig_s994_s981_or_rsi45(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S994: S981 + (RSI < 45 as 4th condition).
    Most 100% WR S929 trades have RSI<50. Adding RSI<45 catches excluded winner?
    Hypothesis: n=19 or 20, tests if RSI condition identifies the excluded winner."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    rsi = float(rt.iloc[i])
    return 1 if rsi < 45 else 0


def sig_s995_s981_or_vix_up5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S995: S981 + (VIX up 5%+ yesterday as 4th condition).
    VIX spike = sudden fear injection before structural bounce.
    Hypothesis: catches the excluded winner if VIX was spiking on that day."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    # VIX spike: today's VIX > yesterday's VIX by ≥5%
    if i >= 1:
        vix_now = float(vix['close'].iloc[i])
        vix_prev = float(vix['close'].iloc[i - 1])
        if vix_prev > 0 and (vix_now - vix_prev) / vix_prev >= 0.05:
            return 1
    return 0


def sig_s996_s981_or_tsla_5d_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S996: S981 + (TSLA 5d return < 0 as 4th condition).
    TSLA has been declining over the past 5 days — catch the recent downtrend before bounce.
    Hypothesis: catches the excluded winner which had a 5-day downtrend."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    ret5 = _tsla_5d_return(tsla, i)
    return 1 if ret5 < 0 else 0


def sig_s997_s929_tsla_5d_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S997: S929 + TSLA 5-day return < 0 (any recent decline before bounce).
    Most structural bounces have had some recent price decline.
    Hypothesis: similar to S929 but excludes the rare 'price at support but still climbing' case."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    ret5 = _tsla_5d_return(tsla, i)
    return 1 if ret5 < 0 else 0


def sig_s998_s929_spy_positive_tsla_negative_5d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S998: S929 + SPY 5d positive AND TSLA 5d negative.
    Classic relative weakness: market up, TSLA down over 5 days.
    Hypothesis: WR ≥ 95%, captures most of S931's setups."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_5d = float(spy['close'].iloc[i - 5])
    tsla_now = float(tsla['close'].iloc[i])
    tsla_5d = float(tsla['close'].iloc[i - 5])
    if spy_5d <= 0 or tsla_5d <= 0:
        return 0
    spy_up = spy_now > spy_5d
    tsla_down = tsla_now < tsla_5d
    return 1 if (spy_up and tsla_down) else 0


def sig_s999_milestone_s981_detail_run(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S999 MILESTONE: S981 — exact copy for --detail analysis run.
    Run with --detail S999 to see trade-by-trade breakdown of the max 100% WR signal."""
    return sig_s981_s931_or_s979(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s1000_milestone_grand(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1000 GRAND MILESTONE: S929 with all positive conditions ORed.
    Union: lag5pct OR 3d-selloff OR SPY-20d-up OR SPY-10d-up OR TSLA-5d-down.
    Attempts to identify the excluded winner and reach n=20 at 100% WR."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # Condition 1: TSLA lagging SPY 5% over 20 days
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    # Condition 2: 3-day TSLA selloff ≥ 3%
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    # Condition 3: SPY 20-day positive
    if i >= 20:
        spy_now = float(spy['close'].iloc[i])
        spy_20d = float(spy['close'].iloc[i - 20])
        if spy_20d > 0 and spy_now > spy_20d:
            return 1
    # Condition 4: TSLA 5-day return negative (simple recent decline)
    ret5 = _tsla_5d_return(tsla, i)
    if ret5 < 0:
        return 1
    return 0


SIGNALS_P10X: List[Tuple] = [
    # Phase 10X — n=20 search + SPY variants + S999/S1000 grand milestones (S991-S1000)
    ('S991_s929_or_lag_sell_spy10', sig_s991_s929_or_lag_selloff_spy10d, 10, 0.20, 50),
    ('S992_s981_or_spy10d',         sig_s992_s981_or_spy10d,            10, 0.20, 50),
    ('S993_s981_or_spy5d',          sig_s993_s981_or_spy5d,             10, 0.20, 50),
    ('S994_s981_or_rsi45',          sig_s994_s981_or_rsi45,             10, 0.20, 50),
    ('S995_s981_or_vix_up5pct',     sig_s995_s981_or_vix_up5pct,       10, 0.20, 50),
    ('S996_s981_or_tsla5d_down',    sig_s996_s981_or_tsla_5d_down,     10, 0.20, 50),
    ('S997_s929_tsla5d_down',       sig_s997_s929_tsla_5d_down,        10, 0.20, 50),
    ('S998_s929_spy_up_tsla_down',  sig_s998_s929_spy_positive_tsla_negative_5d, 10, 0.20, 50),
    ('S999_milestone_s981_detail',  sig_s999_milestone_s981_detail_run, 10, 0.20, 50),
    ('S1000_grand_milestone',       sig_s1000_milestone_grand,          10, 0.20, 50),
]

# ── Phase 10Y — Extended channel windows + MACD/RSI filters + champion extensions ──

def sig_s1001_s929_60w_union(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1001: OR(20/30/40/50/60w) + compressed + VIX 18-50.
    Extends S929 with 60-week (15-month) channel lower support.
    Hypothesis: longer channel windows add new structural touches not captured by 50w."""
    if tw is None or len(tw) < 60:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50, 60):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s1002_s929_80w_union(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1002: OR(20/30/40/50/60/80w) + compressed + VIX 18-50.
    Further extends S1001 with 80-week (20-month) channel support.
    Hypothesis: 80w channel catches multi-year structural lows."""
    if tw is None or len(tw) < 80:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50, 60, 80):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s1003_s929_macd_negative(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1003: S929 base + MACD histogram < 0 (falling momentum at channel support).
    MACD below signal line = selling pressure persists, but structural support = contrarian entry.
    Hypothesis: 'catching falling knife at channel' has high WR when compressed."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None or hist >= 0:
        return 0
    return 1


def sig_s1004_s929_rsi_45(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1004: S929 base + TSLA RSI(14) < 45 (oversold at channel lower bound).
    RSI < 45 confirms price has been weak enough to reach channel support.
    Hypothesis: filters out the rare cases where TSLA drifts to channel while RSI still elevated."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if pd.isna(t_rsi) or t_rsi >= 45:
        return 0
    return 1


def sig_s1005_s929_tsla_10d_pullback(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1005: S929 + TSLA 10-day return < -3% (medium-term pullback to channel).
    Between 3d-selloff (S963: -3% in 3d) and 20d-lag (S968).
    Hypothesis: 10d pullback is a sweet spot between too-fast and too-slow decline."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_10d = float(tsla['close'].iloc[i - 10])
    if c_10d <= 0:
        return 0
    return 1 if (c_now - c_10d) / c_10d < -0.03 else 0


def sig_s1006_s1001_60w_plus_or_conditions(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1006: OR(20/30/40/50/60w) base (S1001) + S993 four OR-conditions.
    The champion's quality filter applied to the extended 60w channel base.
    Hypothesis: adding 60w channel gives new trades that also pass the S993 conditions."""
    if sig_s1001_s929_60w_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # Condition 1: TSLA lagging SPY 5% over 20 days
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    # Condition 2: 3-day TSLA selloff ≥ 3%
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    # Condition 3: SPY 20-day positive
    if i >= 20:
        spy_now = float(spy['close'].iloc[i])
        spy_20d = float(spy['close'].iloc[i - 20])
        if spy_20d > 0 and spy_now > spy_20d:
            return 1
    # Condition 4: SPY 5-day positive
    if i >= 5:
        spy_now = float(spy['close'].iloc[i])
        spy_5d = float(spy['close'].iloc[i - 5])
        if spy_5d > 0 and spy_now > spy_5d:
            return 1
    return 0


def sig_s1007_s993_or_macd_neg(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1007: S993 champion + MACD < 0 as 5th OR condition.
    Attempts to extend champion by catching compressed bounces where momentum still falling.
    Hypothesis: adds 1-2 trades, check if maintains 100% WR."""
    # First check if S993 already fires
    if sig_s993_s981_or_spy5d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    # 5th OR: S929 base + MACD < 0
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None or hist >= 0:
        return 0
    return 1


def sig_s1008_s929_spy_10d_strong(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1008: S929 + SPY 10-day return > 3% (strong SPY recovery over 10 days).
    Tighter recovery confirmation than SPY-20d-up (S968): SPY must be UP 3%+ in 10d.
    Hypothesis: strong SPY recovery = fewer but higher-quality TSLA catch-up setups."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 10:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_10d = float(spy['close'].iloc[i - 10])
    if spy_10d <= 0:
        return 0
    return 1 if (spy_now - spy_10d) / spy_10d > 0.03 else 0


def sig_s1009_s929_rsi_40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1009: S929 base + TSLA RSI(14) < 40 (stricter oversold than S1004's <45).
    Classic deep-oversold confirmation at structural channel support.
    Hypothesis: RSI < 40 = true oversold, should have even higher WR than S1004."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if pd.isna(t_rsi) or t_rsi >= 40:
        return 0
    return 1


def sig_s1010_s993_or_tsla10d_or_rsi45(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1010: S993 champion + TSLA 10d return < -3% OR TSLA RSI < 45 as additional OR conditions.
    Attempts to extend champion to n=21 while maintaining 100% WR.
    These two new paths target medium-term pullbacks and oversold readings on S929 base."""
    # First check if S993 already fires
    if sig_s993_s981_or_spy5d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # 5th OR: TSLA 10d return < -3%
    if i >= 10:
        c_now = float(tsla['close'].iloc[i])
        c_10d = float(tsla['close'].iloc[i - 10])
        if c_10d > 0 and (c_now - c_10d) / c_10d < -0.03:
            return 1
    # 6th OR: TSLA RSI < 45
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 45:
        return 1
    return 0


SIGNALS_P10Y: List[Tuple] = [
    # Phase 10Y — Extended channel windows + MACD/RSI filters + champion extensions (S1001-S1010)
    ('S1001_s929_60w_union',           sig_s1001_s929_60w_union,                10, 0.20, 50),
    ('S1002_s929_80w_union',           sig_s1002_s929_80w_union,                10, 0.20, 50),
    ('S1003_s929_macd_negative',       sig_s1003_s929_macd_negative,            10, 0.20, 50),
    ('S1004_s929_rsi_45',              sig_s1004_s929_rsi_45,                   10, 0.20, 50),
    ('S1005_s929_tsla_10d_pullback',   sig_s1005_s929_tsla_10d_pullback,        10, 0.20, 50),
    ('S1006_s1001_60w_or_conditions',  sig_s1006_s1001_60w_plus_or_conditions,  10, 0.20, 50),
    ('S1007_s993_or_macd_neg',         sig_s1007_s993_or_macd_neg,              10, 0.20, 50),
    ('S1008_s929_spy_10d_strong',      sig_s1008_s929_spy_10d_strong,           10, 0.20, 50),
    ('S1009_s929_rsi_40',              sig_s1009_s929_rsi_40,                   10, 0.20, 50),
    ('S1010_s993_or_tsla10d_rsi45',    sig_s1010_s993_or_tsla10d_or_rsi45,     10, 0.20, 50),
]

# ── Phase 10Z — Precision unions, VIX sweep, channel tightening, spy sweet spot ──

def sig_s1011_s929_macd_or_rsi40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1011: S929 base + (MACD<0 OR TSLA RSI<40).
    Union of S1003 and S1009 — two independent 100% WR precision paths.
    Hypothesis: their union should also be 100% WR with more trades than each alone."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1012_s929_macd_and_rsi45(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1012: S929 + MACD<0 AND TSLA RSI<45 (double confirmation: falling momentum + oversold).
    Intersection of two filters — should give highest WR at cost of fewer trades."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None or hist >= 0:
        return 0
    t_rsi = rt.iloc[i]
    if pd.isna(t_rsi) or t_rsi >= 45:
        return 0
    return 1


def sig_s1013_s929_spy5d_mild(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1013: S929 + SPY 5-day return 0 to +2% (mild recovery, not strong).
    Phase 10Y showed SPY 5d positive good (S993) but SPY 10d > 3% bad (S1008: WR=70%).
    Hypothesis: 'mild SPY recovery' sweet spot = 0-2% over 5d. Captures quality setups."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_5d = float(spy['close'].iloc[i - 5])
    if spy_5d <= 0:
        return 0
    ret5 = (spy_now - spy_5d) / spy_5d
    # Mild: SPY up 0-2% over 5d (not strongly positive)
    return 1 if (0 < ret5 <= 0.02) else 0


def sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1014: OR(20/30/40/50w) + compressed + VIX 15-50 (relax VIX lower from 18 to 15).
    S929 uses VIX 18-50. Trades where VIX=15-17 were excluded. Are they good?
    Hypothesis: calm VIX 15-18 bounces may still work at structural channel support."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (15 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s1015_s929_tight_channel(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1015: OR(20/30/40/50w) + VIX 18-50 + compressed + bottom 15% channel position.
    S929 requires bottom 25%. Tightening to 15% = price very close to channel lower.
    Hypothesis: being in bottom 15% = even stronger structural support → higher WR."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.15):  # 15% vs S929's 25%
                return 1
    return 0


def sig_s1016_s929_lag_and_macd_neg(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1016: S929 + lag5pct AND MACD<0 (both: TSLA lagged AND momentum still falling).
    Combining the strongest 100% WR single condition (lag5pct=S931) with MACD confirmation.
    Hypothesis: lagging + falling momentum = most reliable catch-up setup."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None or hist >= 0:
        return 0
    return 1


def sig_s1017_s993_or_rsi40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1017: S993 champion + TSLA RSI<40 as 5th OR condition.
    Attempts to extend champion from n=20 to n=21 while maintaining 100% WR.
    The deep-oversold path (S1009: 100% WR, n=9) may add trades not in S993."""
    if sig_s993_s981_or_spy5d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    # 5th OR: S929 base + RSI < 40
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1018_s929_tsla_7d_pullback(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1018: S929 + TSLA 7-day return < -3% (medium-term pullback, 7d window).
    Between 3d-selloff (S963: 3 days) and 10d-pullback (S1005: 10 days).
    Hypothesis: 7d window may hit a sweet spot between too-short and too-long lookback."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 7:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_7d = float(tsla['close'].iloc[i - 7])
    if c_7d <= 0:
        return 0
    return 1 if (c_now - c_7d) / c_7d < -0.03 else 0


def sig_s1019_s929_spy_7d_positive(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1019: S929 + SPY 7-day return positive (recovery over 7 days).
    Between SPY-5d-up (S993 uses this) and SPY-10d-up (S991).
    Hypothesis: 7d window may behave similarly to 5d or 10d — interesting to compare."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 7:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_7d = float(spy['close'].iloc[i - 7])
    if spy_7d <= 0:
        return 0
    return 1 if spy_now > spy_7d else 0


def sig_s1020_s993_and_macd_or_rsi(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1020: S993 champion AND (MACD<0 OR RSI<40) — the champion's highest-confidence subset.
    Applies the precision filter AFTER S993's quality gate. Should give maximum WR.
    Hypothesis: best trades = structural support + quality condition + technical confirmation."""
    if sig_s993_s981_or_spy5d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # Must also pass: MACD<0 OR RSI<40
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


SIGNALS_P10Z: List[Tuple] = [
    # Phase 10Z — Precision unions, VIX sweep, channel tightening, spy sweet spot (S1011-S1020)
    ('S1011_s929_macd_or_rsi40',       sig_s1011_s929_macd_or_rsi40,            10, 0.20, 50),
    ('S1012_s929_macd_and_rsi45',      sig_s1012_s929_macd_and_rsi45,           10, 0.20, 50),
    ('S1013_s929_spy5d_mild',          sig_s1013_s929_spy5d_mild,               10, 0.20, 50),
    ('S1014_s929_vix15_base',          sig_s1014_s929_vix15_base,               10, 0.20, 50),
    ('S1015_s929_tight_chan15pct',      sig_s1015_s929_tight_channel,            10, 0.20, 50),
    ('S1016_s929_lag_and_macd_neg',    sig_s1016_s929_lag_and_macd_neg,         10, 0.20, 50),
    ('S1017_s993_or_rsi40',            sig_s1017_s993_or_rsi40,                 10, 0.20, 50),
    ('S1018_s929_tsla_7d_pullback',    sig_s1018_s929_tsla_7d_pullback,         10, 0.20, 50),
    ('S1019_s929_spy_7d_positive',     sig_s1019_s929_spy_7d_positive,          10, 0.20, 50),
    ('S1020_s993_and_macd_or_rsi',     sig_s1020_s993_and_macd_or_rsi,          10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R + SIGNALS_P10S + SIGNALS_P10T
           + SIGNALS_P10U + SIGNALS_P10V + SIGNALS_P10W + SIGNALS_P10X + SIGNALS_P10Y
           + SIGNALS_P10Z)


# ── Phase 11A — VIX 15-50 base sweep + quality filters (S1021-S1030) ─────────

def sig_s1021_s1014_plus_or_conditions(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1021: VIX 15-50 base (S1014) + S993 four OR-conditions.
    Champion quality gate applied to extended VIX range.
    Hypothesis: captures 20 S929 trades + some VIX 15-18 trades that pass conditions → n=22+."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # S993 OR-conditions (4 paths)
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    return 0


def sig_s1022_vix17_base(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1022: OR(20/30/40/50w) + compressed + VIX 17-50 (threshold 17 instead of 18).
    Sweeps VIX lower bound: 15=S1014, 16=S1023, 17=here, 18=S929.
    Hypothesis: VIX 17 may give better risk/reward than 15-16."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (17 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s1023_vix16_base(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1023: OR(20/30/40/50w) + compressed + VIX 16-50 (threshold 16).
    VIX threshold sweep between 15 and 17."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (16 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s1024_s1014_lag5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1024: VIX 15-50 base + lag5pct (TSLA lagging SPY 5%+ over 20d).
    Applies the strongest 100% WR single condition to the extended VIX base.
    Hypothesis: VIX 15-18 lag trades may be winners → could extend to n=19+."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05) else 0


def sig_s1025_s1014_3d_selloff(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1025: VIX 15-50 base + 3-day TSLA return < -3% (short-term selloff).
    Tests the S963 condition on the extended VIX 15-50 base."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if c_3d <= 0:
        return 0
    return 1 if (c_now - c_3d) / c_3d < -0.03 else 0


def sig_s1026_s1014_spy5d_positive(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1026: VIX 15-50 base + SPY 5-day positive (key S993 condition).
    Tests how the SPY-5d filter performs on the extended VIX base."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_5d = float(spy['close'].iloc[i - 5])
    if spy_5d <= 0:
        return 0
    return 1 if spy_now > spy_5d else 0


def sig_s1027_vix12_base(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1027: OR(20/30/40/50w) + compressed + VIX 12-50 (extreme lower, very calm market).
    Tests whether bounces in extremely calm VIX environments have any edge."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (12 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s1028_s1021_or_spy7d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1028: S1021 (VIX 15-50 + S993 conditions) + SPY 7d positive as 5th OR.
    Extends S1021 to catch additional trades via the SPY 7-day window."""
    if sig_s1021_s1014_plus_or_conditions(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 7:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_7d = float(spy['close'].iloc[i - 7])
    if spy_7d <= 0:
        return 0
    return 1 if spy_now > spy_7d else 0


def sig_s1029_s1014_macd_neg(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1029: VIX 15-50 base + MACD<0 (precision filter on extended VIX range).
    Tests whether falling-momentum bounces at VIX 15-50 maintain 100% WR."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None or hist >= 0:
        return 0
    return 1


def sig_s1030_s1014_rsi40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1030: VIX 15-50 base + TSLA RSI<40 (deep oversold at extended VIX base).
    Tests whether the deep-oversold path works at lower VIX levels."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if pd.isna(t_rsi) or t_rsi >= 40:
        return 0
    return 1


SIGNALS_P11A: List[Tuple] = [
    # Phase 11A — VIX 15-50 base sweep + quality filters (S1021-S1030)
    ('S1021_s1014_vix15_or_conditions', sig_s1021_s1014_plus_or_conditions,     10, 0.20, 50),
    ('S1022_vix17_base',                sig_s1022_vix17_base,                   10, 0.20, 50),
    ('S1023_vix16_base',                sig_s1023_vix16_base,                   10, 0.20, 50),
    ('S1024_s1014_lag5pct',             sig_s1024_s1014_lag5pct,                10, 0.20, 50),
    ('S1025_s1014_3d_selloff',          sig_s1025_s1014_3d_selloff,             10, 0.20, 50),
    ('S1026_s1014_spy5d_positive',      sig_s1026_s1014_spy5d_positive,         10, 0.20, 50),
    ('S1027_vix12_base',                sig_s1027_vix12_base,                   10, 0.20, 50),
    ('S1028_s1021_or_spy7d',            sig_s1028_s1021_or_spy7d,               10, 0.20, 50),
    ('S1029_s1014_macd_neg',            sig_s1029_s1014_macd_neg,               10, 0.20, 50),
    ('S1030_s1014_rsi40',               sig_s1030_s1014_rsi40,                  10, 0.20, 50),
]


# ── Phase 11B — Union of VIX-15 precision paths (S1031-S1040) ─────────────────

def sig_s1031_vix15_selloff_or_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1031: VIX 15-50 base + (3d-selloff OR MACD<0).
    Union of S1025 (100% WR, n=12) and S1029 (100% WR, n=9).
    Hypothesis: both paths independently 100% WR → union also 100% WR, n=14-16."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    return 0


def sig_s1032_vix15_selloff_or_rsi40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1032: VIX 15-50 base + (3d-selloff OR RSI<40).
    Union of S1025 (100% WR, n=12) and S1030 (100% WR, n=11).
    Hypothesis: union should give n=16-18 at 100% WR."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    return 0


def sig_s1033_vix15_macd_or_rsi40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1033: VIX 15-50 base + (MACD<0 OR RSI<40).
    Union of S1029 (100% WR, n=9) and S1030 (100% WR, n=11).
    Extends S1011 (VIX 18-50 + MACD/RSI, n=11) with VIX 15-18 trades."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1034_vix15_3way_union(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1034: VIX 15-50 + (3d-selloff OR MACD<0 OR RSI<40) — 3-way union.
    Union of all three 100% WR VIX-15 precision paths.
    Hypothesis: maximum precision coverage on VIX 15-50 base."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    return 0


def sig_s1035_s1027_vix12_or_conditions(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1035: VIX 12-50 base (S1027) + S993 OR-conditions.
    Maximum VIX coverage with champion quality filter.
    Hypothesis: extends S1021 to even lower VIX, tests how far down VIX can go."""
    if sig_s1027_vix12_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    return 0


def sig_s1036_s1021_and_precision(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1036: S1021 (VIX 15-50 + S993 conditions) AND (MACD<0 OR RSI<40).
    Applies precision gate to S1021 to remove 2 losers from VIX 15-18 zone.
    Hypothesis: the 2 S1021 losers are NOT in MACD<0 or RSI<40 → applying this removes them."""
    if sig_s1021_s1014_plus_or_conditions(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1037_vix15_tsla5d_neg(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1037: VIX 15-50 base + TSLA 5-day return negative.
    Looser version of 3d-selloff: any 5d decline (no minimum threshold).
    Hypothesis: 5d negative (S997 analog) on VIX 15-50 base."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    ret5 = _tsla_5d_return(tsla, i)
    return 1 if ret5 < 0 else 0


def sig_s1038_vix15_selloff_or_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1038: VIX 15-50 base + (3d-selloff OR lag5pct).
    Tests whether adding lag5pct to the 3d-selloff union maintains or hurts WR."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    return 0


def sig_s1039_vix15_selloff_or_spy20d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1039: VIX 15-50 base + (3d-selloff OR SPY 20-day positive).
    Tests the longer SPY recovery window on VIX 15-50 base with selloff union."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
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
    return 0


def sig_s1040_vix15_4way_precision(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1040: VIX 15-50 + (3d-selloff OR MACD<0 OR RSI<40 OR lag5pct) — 4-way precision union.
    Extends S1034 with lag5pct. Tests maximum precision on VIX 15-50 base."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    return 0


SIGNALS_P11B: List[Tuple] = [
    # Phase 11B — Union of VIX-15 precision paths (S1031-S1040)
    ('S1031_vix15_selloff_or_macd',    sig_s1031_vix15_selloff_or_macd,         10, 0.20, 50),
    ('S1032_vix15_selloff_or_rsi40',   sig_s1032_vix15_selloff_or_rsi40,        10, 0.20, 50),
    ('S1033_vix15_macd_or_rsi40',      sig_s1033_vix15_macd_or_rsi40,           10, 0.20, 50),
    ('S1034_vix15_3way_union',         sig_s1034_vix15_3way_union,              10, 0.20, 50),
    ('S1035_s1027_vix12_or_cond',      sig_s1035_s1027_vix12_or_conditions,     10, 0.20, 50),
    ('S1036_s1021_and_precision',      sig_s1036_s1021_and_precision,           10, 0.20, 50),
    ('S1037_vix15_tsla5d_neg',         sig_s1037_vix15_tsla5d_neg,              10, 0.20, 50),
    ('S1038_vix15_selloff_or_lag',     sig_s1038_vix15_selloff_or_lag,          10, 0.20, 50),
    ('S1039_vix15_selloff_or_spy20d',  sig_s1039_vix15_selloff_or_spy20d,       10, 0.20, 50),
    ('S1040_vix15_4way_precision',     sig_s1040_vix15_4way_precision,          10, 0.20, 50),
]


# ── Phase 11C — Super-champion (S993 OR S1034) + SPY extensions (S1041-S1050) ─

def sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1041: S993 champion OR S1034 VIX-15 champion.
    Combines both 100% WR families: VIX 18-50 quality-gated + VIX 15-50 precision.
    Hypothesis: super-champion with n=20+? at 100% WR, maximum P&L."""
    if sig_s993_s981_or_spy5d(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return sig_s1034_vix15_3way_union(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s1042_s1034_or_spy5d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1042: S1034 base + SPY 5-day positive as 4th OR condition.
    Tests whether SPY-5d (S993's key condition) maintains 100% WR on VIX 15-50 base."""
    if sig_s1034_vix15_3way_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_5d = float(spy['close'].iloc[i - 5])
    if spy_5d <= 0:
        return 0
    return 1 if spy_now > spy_5d else 0


def sig_s1043_s1034_or_spy20d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1043: S1034 base + SPY 20-day positive as 4th OR condition.
    Tests whether SPY-20d-up maintains 100% WR on VIX 15-50 base."""
    if sig_s1034_vix15_3way_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_20d = float(spy['close'].iloc[i - 20])
    if spy_20d <= 0:
        return 0
    return 1 if spy_now > spy_20d else 0


def sig_s1044_s1034_or_spy5d_and_20d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1044: S1034 + (SPY 5d-up OR SPY 20d-up) as additional OR conditions.
    Adds both SPY conditions to S1034. Hypothesis: n=20+ if SPY conditions well-behaved."""
    if sig_s1034_vix15_3way_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 5:
        spy_now = float(spy['close'].iloc[i])
        spy_5d = float(spy['close'].iloc[i - 5])
        if spy_5d > 0 and spy_now > spy_5d:
            return 1
    if i >= 20:
        spy_now = float(spy['close'].iloc[i])
        spy_20d = float(spy['close'].iloc[i - 20])
        if spy_20d > 0 and spy_now > spy_20d:
            return 1
    return 0


def sig_s1045_vix15_5way_union(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1045: VIX 15-50 + (3d-selloff OR MACD<0 OR RSI<40 OR SPY-5d-up OR SPY-20d-up).
    5-way union on VIX 15-50 base — S993's conditions combined with S1034's precision.
    Hypothesis: maximum coverage on VIX 15-50 base, check WR degradation."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    if i >= 5:
        spy_now = float(spy['close'].iloc[i])
        spy_5d = float(spy['close'].iloc[i - 5])
        if spy_5d > 0 and spy_now > spy_5d:
            return 1
    if i >= 20:
        spy_now = float(spy['close'].iloc[i])
        spy_20d = float(spy['close'].iloc[i - 20])
        if spy_20d > 0 and spy_now > spy_20d:
            return 1
    return 0


def sig_s1046_vix15_selloff_and_precision(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1046: VIX 15-50 + 3d-selloff AND (MACD<0 OR RSI<40) — intersection.
    Tightest subset: short-term selloff confirmed by technical deterioration.
    Hypothesis: strictest filter, highest WR, lowest n."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if c_3d <= 0 or (c_now - c_3d) / c_3d >= -0.03:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1047_vix15_3way_or_lag8pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1047: S1034 (3-way) + lag8pct as 4th OR (stronger lag threshold than 5%).
    Phase 10Y: lag5pct breaks 100% WR. Tests if lag8% (more selective) works."""
    if sig_s1034_vix15_3way_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.08) else 0


def sig_s1048_vix15_selloff_and_rsi(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1048: VIX 15-50 + 3d-selloff AND RSI<40 (double-filter: selloff + deep oversold).
    Strictest intersection: both conditions simultaneously true."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if pd.isna(t_rsi) or t_rsi >= 40:
        return 0
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if c_3d <= 0:
        return 0
    return 1 if (c_now - c_3d) / c_3d < -0.03 else 0


def sig_s1049_s1041_or_rsi45(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1049: Super-champion S1041 + RSI<45 as additional OR.
    Attempts to extend S1041 beyond S993's 20 with oversold mid-threshold."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 45:
        return 1
    return 0


def sig_s1050_grand_vix15_champion(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1050: Grand VIX-15 champion — S1034 + (SPY-5d-up OR SPY-20d-up OR lag8pct).
    6-way union on VIX 15-50: precision paths + SPY conditions + strong lag.
    Grand milestone for the VIX 15-50 family exploration."""
    if sig_s1034_vix15_3way_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    # SPY 5d up
    if i >= 5:
        spy_now = float(spy['close'].iloc[i])
        spy_5d = float(spy['close'].iloc[i - 5])
        if spy_5d > 0 and spy_now > spy_5d:
            return 1
    # SPY 20d up
    if i >= 20:
        spy_now = float(spy['close'].iloc[i])
        spy_20d = float(spy['close'].iloc[i - 20])
        if spy_20d > 0 and spy_now > spy_20d:
            return 1
    # lag 8% (stronger than 5%)
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.08):
        return 1
    return 0


SIGNALS_P11C: List[Tuple] = [
    # Phase 11C — Super-champion (S993 OR S1034) + SPY extensions (S1041-S1050)
    ('S1041_s993_or_s1034',            sig_s1041_s993_or_s1034,                 10, 0.20, 50),
    ('S1042_s1034_or_spy5d',           sig_s1042_s1034_or_spy5d,                10, 0.20, 50),
    ('S1043_s1034_or_spy20d',          sig_s1043_s1034_or_spy20d,               10, 0.20, 50),
    ('S1044_s1034_or_spy5d_20d',       sig_s1044_s1034_or_spy5d_and_20d,        10, 0.20, 50),
    ('S1045_vix15_5way_union',         sig_s1045_vix15_5way_union,              10, 0.20, 50),
    ('S1046_vix15_selloff_and_prec',   sig_s1046_vix15_selloff_and_precision,   10, 0.20, 50),
    ('S1047_vix15_3way_or_lag8pct',    sig_s1047_vix15_3way_or_lag8pct,         10, 0.20, 50),
    ('S1048_vix15_selloff_and_rsi',    sig_s1048_vix15_selloff_and_rsi,         10, 0.20, 50),
    ('S1049_s1041_or_rsi45',           sig_s1049_s1041_or_rsi45,                10, 0.20, 50),
    ('S1050_grand_vix15_champion',     sig_s1050_grand_vix15_champion,          10, 0.20, 50),
]


# ── Phase 11D — Elevated VIX zone (20-50) + 5d-selloff + champion extensions (S1051-S1060) ──

def sig_s1051_vix20_base(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1051: OR(20/30/40/50w) + compressed + VIX 20-50 (elevated VIX only).
    Hypothesis: moderate-high fear regime = cleanest bounces → WR > S929's 95%?"""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (20 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s1052_vix25_base(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1052: OR(20/30/40/50w) + compressed + VIX 25-50 (high VIX only).
    Tests whether the highest fear regime (VIX 25+) gives the cleanest bounces."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (25 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s1053_vix20_or_conditions(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1053: VIX 20-50 base + S993 OR-conditions (champion quality gate on elevated VIX).
    Hypothesis: elevated VIX + quality gate = maximum quality bounces."""
    if sig_s1051_vix20_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    return 0


def sig_s1054_vix20_3way_precision(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1054: VIX 20-50 + (3d-selloff OR MACD<0 OR RSI<40) — S1034 conditions on elevated VIX.
    Hypothesis: elevated VIX + 3-way precision = 100% WR, more trades than S1034."""
    if sig_s1051_vix20_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    return 0


def sig_s1055_vix20_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1055: VIX 20-50 base OR S1034 (VIX 15-50 precision) — alternative super-champion.
    Tests whether VIX 20-50 base (unfiltered) union with S1034 gives 100% WR."""
    if sig_s1051_vix20_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return sig_s1034_vix15_3way_union(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s1056_s1041_or_5d_selloff(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1056: S1041 super-champion + 5d-selloff (TSLA 5d < -3%) as additional OR.
    5d-selloff vs S1041: attempts to extend champion by catching medium-term selloffs."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_5d = float(tsla['close'].iloc[i - 5])
    if c_5d <= 0:
        return 0
    return 1 if (c_now - c_5d) / c_5d < -0.03 else 0


def sig_s1057_vix15_5d_selloff(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1057: VIX 15-50 base + TSLA 5-day return < -3% (5d-selloff condition).
    Tests 5d window between 3d (S1025) and 10d (S1005). 3d=100% WR, 10d=92% WR."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_5d = float(tsla['close'].iloc[i - 5])
    if c_5d <= 0:
        return 0
    return 1 if (c_now - c_5d) / c_5d < -0.03 else 0


def sig_s1058_vix15_3d_or_5d_selloff(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1058: VIX 15-50 + (3d-selloff OR 5d-selloff) — extended selloff window.
    Union of 3d and 5d thresholds: catches more selling events."""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    if i >= 5:
        c_now = float(tsla['close'].iloc[i])
        c_5d = float(tsla['close'].iloc[i - 5])
        if c_5d > 0 and (c_now - c_5d) / c_5d < -0.03:
            return 1
    return 0


def sig_s1059_vix15_3d_5d_macd_rsi(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1059: VIX 15-50 + (3d-selloff OR 5d-selloff OR MACD<0 OR RSI<40) — 4-way with 5d.
    Extends S1034 by adding 5d-selloff as 4th OR. Does it maintain 100% WR?"""
    if sig_s1014_s929_vix15_base(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    if i >= 5:
        c_now = float(tsla['close'].iloc[i])
        c_5d = float(tsla['close'].iloc[i - 5])
        if c_5d > 0 and (c_now - c_5d) / c_5d < -0.03:
            return 1
    return 0


def sig_s1060_s1041_or_vix20_precision(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1060: S1041 OR VIX 20-50 precision (S1054) — super-champion with elevated VIX path.
    Attempts to add elevated-VIX precise bounces to S1041's universe."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return sig_s1054_vix20_3way_precision(i, tsla, spy, vix, tw, sw, rt, rs, w)


SIGNALS_P11D: List[Tuple] = [
    # Phase 11D — Elevated VIX zone (20-50) + 5d-selloff + champion extensions (S1051-S1060)
    ('S1051_vix20_base',               sig_s1051_vix20_base,                    10, 0.20, 50),
    ('S1052_vix25_base',               sig_s1052_vix25_base,                    10, 0.20, 50),
    ('S1053_vix20_or_conditions',      sig_s1053_vix20_or_conditions,           10, 0.20, 50),
    ('S1054_vix20_3way_precision',     sig_s1054_vix20_3way_precision,          10, 0.20, 50),
    ('S1055_vix20_or_s1034',           sig_s1055_vix20_or_s1034,                10, 0.20, 50),
    ('S1056_s1041_or_5d_selloff',      sig_s1056_s1041_or_5d_selloff,           10, 0.20, 50),
    ('S1057_vix15_5d_selloff',         sig_s1057_vix15_5d_selloff,              10, 0.20, 50),
    ('S1058_vix15_3d_or_5d_selloff',   sig_s1058_vix15_3d_or_5d_selloff,       10, 0.20, 50),
    ('S1059_vix15_3d_5d_macd_rsi',     sig_s1059_vix15_3d_5d_macd_rsi,         10, 0.20, 50),
    ('S1060_s1041_or_vix20_precision', sig_s1060_s1041_or_vix20_precision,      10, 0.20, 50),
]


# ── Phase 11E — VIX upper bound + short channels + extreme fear (S1061-S1070) ─

def sig_s1061_vix18_70_or_conditions(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1061: OR(20/30/40/50w) + compressed + VIX 18-70 + S993 OR-conditions.
    Extends S993's VIX upper bound from 50 to 70 to capture extreme fear bounces.
    COVID March-April 2020 had VIX 40-82 — these were massive bounce opportunities."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 70):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                break
    else:
        return 0
    # S993 OR-conditions
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    return 0


def sig_s1062_vix15_70_3way(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1062: OR(20/30/40/50w) + compressed + VIX 15-70 + S1034 precision conditions.
    Full extension: lower VIX from 18 to 15, upper from 50 to 70.
    Hypothesis: captures both low-VIX precision trades AND high-VIX panic bounces."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (15 <= vix_now <= 70):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found_channel = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                found_channel = True
                break
    if not found_channel:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    return 0


def sig_s1063_s1041_vix70(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1063: S1041 super-champion with VIX upper bound extended to 70.
    = (S993 conditions OR S1034 conditions) with VIX 15-70 instead of 15-50.
    The 3 additional VIX 50-70 trades from 2020-2022 should be massive winners."""
    # S993 extended to VIX 70
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (15 <= vix_now <= 70):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found_channel = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                found_channel = True
                break
    if not found_channel:
        return 0
    # S993 OR-conditions
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    if i >= 3:
        c_now_v = float(tsla['close'].iloc[i])
        c_3d_v = float(tsla['close'].iloc[i - 3])
        if c_3d_v > 0 and (c_now_v - c_3d_v) / c_3d_v < -0.03:
            return 1
    if i >= 20:
        spy_now_v = float(spy['close'].iloc[i])
        spy_20d_v = float(spy['close'].iloc[i - 20])
        if spy_20d_v > 0 and spy_now_v > spy_20d_v:
            return 1
    if i >= 5:
        spy_now_v = float(spy['close'].iloc[i])
        spy_5d_v = float(spy['close'].iloc[i - 5])
        if spy_5d_v > 0 and spy_now_v > spy_5d_v:
            return 1
    # S1034 precision conditions (MACD/RSI already checked above for 3d-selloff)
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1064_vix18_100_or_cond(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1064: OR(20/30/40/50w) + compressed + VIX 18-100 + S993 OR-conditions.
    No upper bound on VIX — catches all fear regimes including COVID extreme (VIX>80)."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if vix_now < 18:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                found = True
                break
    if not found:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    if i >= 3:
        c_now2 = float(tsla['close'].iloc[i])
        c_3d2 = float(tsla['close'].iloc[i - 3])
        if c_3d2 > 0 and (c_now2 - c_3d2) / c_3d2 < -0.03:
            return 1
    if i >= 20:
        s_now2 = float(spy['close'].iloc[i])
        s_20d2 = float(spy['close'].iloc[i - 20])
        if s_20d2 > 0 and s_now2 > s_20d2:
            return 1
    if i >= 5:
        s_now2 = float(spy['close'].iloc[i])
        s_5d2 = float(spy['close'].iloc[i - 5])
        if s_5d2 > 0 and s_now2 > s_5d2:
            return 1
    return 0


def sig_s1065_s929_12w_added(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1065: OR(12/20/30/40/50w) + compressed + VIX 18-50 (add 12-week channel).
    12-week channel = 3-month support. Tests if short-cycle channel adds good trades."""
    if tw is None or len(tw) < 20:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (12, 20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s1066_s929_15w_added(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1066: OR(15/20/30/40/50w) + compressed + VIX 18-50 (add 15-week channel).
    15-week = 3.75-month support. Between 12w and 20w."""
    if tw is None or len(tw) < 20:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (15, 20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s1067_s1041_vix70_super(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1067: S1041 (VIX 15-50) OR VIX 50-70 precision (new extreme fear zone).
    Extends S1041 by adding high-VIX (50-70) compressed bounces with precision filter."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    # New: VIX 50-70 precision bounces
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (50 <= vix_now <= 70):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s1068_vix50plus_base(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1068: OR(20/30/40/50w) + compressed + VIX 50+ only (pure extreme fear).
    Isolates the panic-bounce regime: VIX > 50 = market crisis (COVID, 2022 peak).
    Hypothesis: extreme panic bounces are highly reliable at structural support."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if vix_now < 50:
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                return 1
    return 0


def sig_s1069_s1065_plus_or_cond(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1069: OR(12/20/30/40/50w) + compressed + VIX 18-50 + S993 OR-conditions.
    12w channel added to S993 family — what trades does 12w add?"""
    if sig_s1065_s929_12w_added(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    if i >= 3:
        c_now3 = float(tsla['close'].iloc[i])
        c_3d3 = float(tsla['close'].iloc[i - 3])
        if c_3d3 > 0 and (c_now3 - c_3d3) / c_3d3 < -0.03:
            return 1
    if i >= 20:
        s_now3 = float(spy['close'].iloc[i])
        s_20d3 = float(spy['close'].iloc[i - 20])
        if s_20d3 > 0 and s_now3 > s_20d3:
            return 1
    if i >= 5:
        s_now3 = float(spy['close'].iloc[i])
        s_5d3 = float(spy['close'].iloc[i - 5])
        if s_5d3 > 0 and s_now3 > s_5d3:
            return 1
    return 0


def sig_s1070_s1041_or_vix70_3way(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1070: S1041 champion OR S1062 (VIX 15-70 + 3-way precision).
    Grand extension: S1041 universe + any VIX 50-70 precision bounces."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return sig_s1062_vix15_70_3way(i, tsla, spy, vix, tw, sw, rt, rs, w)


SIGNALS_P11E: List[Tuple] = [
    # Phase 11E — VIX upper bound + short channels + extreme fear (S1061-S1070)
    ('S1061_vix18_70_or_cond',         sig_s1061_vix18_70_or_conditions,        10, 0.20, 50),
    ('S1062_vix15_70_3way',            sig_s1062_vix15_70_3way,                 10, 0.20, 50),
    ('S1063_s1041_vix70',              sig_s1063_s1041_vix70,                   10, 0.20, 50),
    ('S1064_vix18_100_or_cond',        sig_s1064_vix18_100_or_cond,             10, 0.20, 50),
    ('S1065_s929_12w_added',           sig_s1065_s929_12w_added,                10, 0.20, 50),
    ('S1066_s929_15w_added',           sig_s1066_s929_15w_added,                10, 0.20, 50),
    ('S1067_s1041_vix70_super',        sig_s1067_s1041_vix70_super,             10, 0.20, 50),
    ('S1068_vix50plus_base',           sig_s1068_vix50plus_base,                10, 0.20, 50),
    ('S1069_s1065_12w_or_cond',        sig_s1069_s1065_plus_or_cond,            10, 0.20, 50),
    ('S1070_s1041_or_vix70_3way',      sig_s1070_s1041_or_vix70_3way,           10, 0.20, 50),
]

# ── Phase 11F: VIX 50-70 precision filters to recover 100% WR on S1063 ────────
# S1063 has n=25 WR=92% ($2,064K) — 2 losers from VIX 50-70 zone.
# Goal: filter those losers with momentum/oscillator conditions → n=24+ 100% WR.

def sig_s1071_vix70_3dselloff(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1071: S1041 OR (VIX 50-70 + compressed + channel + 3d-selloff).
    VIX 50-70 arm requires 3-day TSLA selloff >3% to confirm panic entry."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (50 <= vix_now <= 70):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found_channel = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                found_channel = True
                break
    if not found_channel:
        return 0
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    return 0


def sig_s1072_vix70_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1072: S1041 OR (VIX 50-70 + compressed + channel + MACD<0).
    VIX 50-70 arm requires bearish MACD histogram to confirm downtrend."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (50 <= vix_now <= 70):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found_channel = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                found_channel = True
                break
    if not found_channel:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    return 0


def sig_s1073_vix70_rsi40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1073: S1041 OR (VIX 50-70 + compressed + channel + RSI<40).
    VIX 50-70 arm requires oversold RSI<40 to confirm panic-oversold entry."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (50 <= vix_now <= 70):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found_channel = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                found_channel = True
                break
    if not found_channel:
        return 0
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1074_vix70_3way(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1074: S1041 OR (VIX 50-70 + compressed + channel + (3d-selloff OR MACD<0 OR RSI<40)).
    Full 3-way precision filter on VIX 50-70 arm — same filters as S1034's VIX-15 arm."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (50 <= vix_now <= 70):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found_channel = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                found_channel = True
                break
    if not found_channel:
        return 0
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1075_vix70_lag5(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1075: S1041 OR (VIX 50-70 + compressed + channel + lag5pct).
    VIX 50-70 arm requires TSLA lagging SPY by 5% (relative weakness filter)."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (50 <= vix_now <= 70):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found_channel = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                found_channel = True
                break
    if not found_channel:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    return 0


def sig_s1076_vix45_60(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1076: S1041 OR (VIX 45-60 + compressed + channel + any-OR-cond).
    Extends S1041 into the VIX 45-60 elevated fear zone with full OR conditions."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (45 <= vix_now <= 60):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found_channel = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                found_channel = True
                break
    if not found_channel:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1077_s1063_vix55(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1077: S1063 with VIX upper bound tightened to 55 (not 70).
    Test if the 2 S1063 losers are in VIX 55-70 zone — cutting to 55 may recover 100% WR."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (15 <= vix_now <= 55):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found_channel = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                found_channel = True
                break
    if not found_channel:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    if i >= 3:
        c_now_v = float(tsla['close'].iloc[i])
        c_3d_v = float(tsla['close'].iloc[i - 3])
        if c_3d_v > 0 and (c_now_v - c_3d_v) / c_3d_v < -0.03:
            return 1
    if i >= 20:
        spy_now_v = float(spy['close'].iloc[i])
        spy_20d_v = float(spy['close'].iloc[i - 20])
        if spy_20d_v > 0 and spy_now_v > spy_20d_v:
            return 1
    if i >= 5:
        spy_now_v = float(spy['close'].iloc[i])
        spy_5d_v = float(spy['close'].iloc[i - 5])
        if spy_5d_v > 0 and spy_now_v > spy_5d_v:
            return 1
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1078_s1063_vix52(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1078: S1063 with VIX upper bound tightened to 52 (just above S1041's 50).
    Adds minimal VIX 50-52 extension — tests if losers are above 52."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (15 <= vix_now <= 52):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found_channel = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                found_channel = True
                break
    if not found_channel:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    if i >= 3:
        c_now_v = float(tsla['close'].iloc[i])
        c_3d_v = float(tsla['close'].iloc[i - 3])
        if c_3d_v > 0 and (c_now_v - c_3d_v) / c_3d_v < -0.03:
            return 1
    if i >= 20:
        spy_now_v = float(spy['close'].iloc[i])
        spy_20d_v = float(spy['close'].iloc[i - 20])
        if spy_20d_v > 0 and spy_now_v > spy_20d_v:
            return 1
    if i >= 5:
        spy_now_v = float(spy['close'].iloc[i])
        spy_5d_v = float(spy['close'].iloc[i - 5])
        if spy_5d_v > 0 and spy_now_v > spy_5d_v:
            return 1
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1079_s1041_atr65(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1079: S1041 with ATR threshold tightened to 0.65 (vs 0.75 in S1041).
    Stricter compression filter — eliminates more marginal trades to reduce risk."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (15 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.65 * atr_20:  # stricter: 0.65 vs 0.75
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found_channel = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                found_channel = True
                break
    if not found_channel:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1080_s1041_chan20pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1080: S1041 with channel position tightened to lower 20% (vs 25% in S1041).
    More precise support entry — trades only deepest channel penetration zones."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (15 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found_channel = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.20):  # 20% not 25%
                found_channel = True
                break
    if not found_channel:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


SIGNALS_P11F: List[Tuple] = [
    ('S1071_vix70_3dselloff',          sig_s1071_vix70_3dselloff,               10, 0.20, 50),
    ('S1072_vix70_macd',               sig_s1072_vix70_macd,                    10, 0.20, 50),
    ('S1073_vix70_rsi40',              sig_s1073_vix70_rsi40,                   10, 0.20, 50),
    ('S1074_vix70_3way',               sig_s1074_vix70_3way,                    10, 0.20, 50),
    ('S1075_vix70_lag5',               sig_s1075_vix70_lag5,                    10, 0.20, 50),
    ('S1076_vix45_60_fullcond',        sig_s1076_vix45_60,                      10, 0.20, 50),
    ('S1077_s1063_vix55',              sig_s1077_s1063_vix55,                   10, 0.20, 50),
    ('S1078_s1063_vix52',              sig_s1078_s1063_vix52,                   10, 0.20, 50),
    ('S1079_s1041_atr65',              sig_s1079_s1041_atr65,                   10, 0.20, 50),
    ('S1080_s1041_chan20pct',          sig_s1080_s1041_chan20pct,               10, 0.20, 50),
]

# ── Phase 11G: Intersection precision + alternative entry filters ──────────────
# Champion S1041 (n=23, 100% WR, $2,037K) uses OR-conditions. Testing:
# (a) AND-intersections of S1041's OR arms → ultra-precision, fewer trades, higher avg
# (b) New candidate filters: 5d selloff, deeper selloff, RSI sweeps, TSLA/SPY SMA
# (c) Selloff depth variants: 3d>5%, 5d>3%, 7d>5%

def sig_s1081_s929_and_selloff_lag(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1081: S929 base AND (3d-selloff AND lag5pct) — intersection of 2 OR conditions.
    Requires BOTH 3-day selloff AND TSLA lagging SPY — most precise dual-condition entry."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if not (c_3d > 0 and (c_now - c_3d) / c_3d < -0.03):
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 1
    return 0


def sig_s1082_s929_and_selloff_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1082: S929 base AND (3d-selloff AND MACD<0) — selloff confirmed by MACD downtrend."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if not (c_3d > 0 and (c_now - c_3d) / c_3d < -0.03):
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    return 0


def sig_s1083_s929_and_lag_rsi40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1083: S929 base AND (lag5pct AND RSI<40) — relative weakness confirmed by oversold RSI."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 0
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    return 0


def sig_s1084_s929_5d_selloff(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1084: S929 base + 5-day selloff >3% (extended selloff vs S993's 3-day).
    Tests if a longer selloff window identifies more reliable entry points."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 5:
        c_now = float(tsla['close'].iloc[i])
        c_5d = float(tsla['close'].iloc[i - 5])
        if c_5d > 0 and (c_now - c_5d) / c_5d < -0.03:
            return 1
    return 0


def sig_s1085_s929_3d_selloff5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1085: S929 base + 3d selloff >5% (deeper drop threshold vs 3% in S993).
    More severe 3-day drop may select panic-bottom entries of higher quality."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.05:
            return 1
    return 0


def sig_s1086_s929_7d_selloff5pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1086: S929 base + 7d selloff >5% — week-long significant drawdown."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 7:
        c_now = float(tsla['close'].iloc[i])
        c_7d = float(tsla['close'].iloc[i - 7])
        if c_7d > 0 and (c_now - c_7d) / c_7d < -0.05:
            return 1
    return 0


def sig_s1087_s929_rsi45(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1087: S929 base + RSI<45 (looser than RSI<40 in S1030).
    RSI 40-45 zone captures near-oversold entries — may add 2-3 trades from 2017/2023."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 45:
        return 1
    return 0


def sig_s1088_s929_rsi35(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1088: S929 base + RSI<35 — strictly oversold, tighter than RSI<40.
    Tests if the RSI<40 trades in S1030 are concentrated in the RSI<35 zone."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 35:
        return 1
    return 0


def sig_s1089_s1041_rsi45(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1089: S1041 OR (S929 VIX 18-50 + RSI 40-45) — add RSI 40-45 as new OR arm.
    S1041 already uses RSI<40 via S1034; this adds looser RSI 40-45 coverage."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and 40 <= t_rsi < 45:
        return 1
    return 0


def sig_s1090_s1041_5d_selloff(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1090: S1041 OR (S929 VIX 18-50 + 5d-selloff) — add 5d-selloff as new OR arm.
    Tests if a 5-day selloff (vs 3d in S993) captures distinct high-quality trades."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 5:
        c_now = float(tsla['close'].iloc[i])
        c_5d = float(tsla['close'].iloc[i - 5])
        if c_5d > 0 and (c_now - c_5d) / c_5d < -0.03:
            return 1
    return 0


SIGNALS_P11G: List[Tuple] = [
    ('S1081_s929_and_selloff_lag',     sig_s1081_s929_and_selloff_lag,          10, 0.20, 50),
    ('S1082_s929_and_selloff_macd',    sig_s1082_s929_and_selloff_macd,         10, 0.20, 50),
    ('S1083_s929_and_lag_rsi40',       sig_s1083_s929_and_lag_rsi40,            10, 0.20, 50),
    ('S1084_s929_5d_selloff',          sig_s1084_s929_5d_selloff,               10, 0.20, 50),
    ('S1085_s929_3d_selloff5pct',      sig_s1085_s929_3d_selloff5pct,           10, 0.20, 50),
    ('S1086_s929_7d_selloff5pct',      sig_s1086_s929_7d_selloff5pct,           10, 0.20, 50),
    ('S1087_s929_rsi45',               sig_s1087_s929_rsi45,                    10, 0.20, 50),
    ('S1088_s929_rsi35',               sig_s1088_s929_rsi35,                    10, 0.20, 50),
    ('S1089_s1041_rsi45',              sig_s1089_s1041_rsi45,                   10, 0.20, 50),
    ('S1090_s1041_5d_selloff',         sig_s1090_s1041_5d_selloff,              10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R + SIGNALS_P10S + SIGNALS_P10T
           + SIGNALS_P10U + SIGNALS_P10V + SIGNALS_P10W + SIGNALS_P10X + SIGNALS_P10Y
           + SIGNALS_P10Z + SIGNALS_P11A + SIGNALS_P11B + SIGNALS_P11C + SIGNALS_P11D + SIGNALS_P11E
           + SIGNALS_P11F + SIGNALS_P11G)

# ── Phase 11H: SPY RSI conditions + 10d TSLA pullback + new OR arms ────────────
# Testing the `rs` variable (SPY RSI 14-period daily) — never used before.
# Goal: find new 100% WR OR arms to add to S1041 (currently n=23 ceiling).
# Key hypothesis: SPY RSI < 50 + TSLA RSI < 40 = market + stock both oversold.

def sig_s1091_s929_spy_rsi45(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1091: S929 base + SPY RSI<45 — whole market somewhat oversold."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not pd.isna(rs.iloc[i]) and rs.iloc[i] < 45:
        return 1
    return 0


def sig_s1092_s929_spy_rsi40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1092: S929 base + SPY RSI<40 — whole market very oversold (market bottom)."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not pd.isna(rs.iloc[i]) and rs.iloc[i] < 40:
        return 1
    return 0


def sig_s1093_s929_tsla_rsi40_spy_above40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1093: S929 + (TSLA RSI<40 AND SPY RSI>40) — TSLA-specific weakness.
    TSLA oversold while market still neutral/healthy = highest quality catch-up."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    tsla_rsi = rt.iloc[i]
    spy_rsi = rs.iloc[i]
    if pd.isna(tsla_rsi) or pd.isna(spy_rsi):
        return 0
    if tsla_rsi < 40 and spy_rsi > 40:
        return 1
    return 0


def sig_s1094_s929_10d_selloff3pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1094: S929 base + 10-day TSLA pullback >3% — medium-term extended decline."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 10:
        c_now = float(tsla['close'].iloc[i])
        c_10d = float(tsla['close'].iloc[i - 10])
        if c_10d > 0 and (c_now - c_10d) / c_10d < -0.03:
            return 1
    return 0


def sig_s1095_s929_spy5d_down1pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1095: S929 + SPY 5d down >1% — short-term market pullback during TSLA bounce zone."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 5:
        spy_now = float(spy['close'].iloc[i])
        spy_5d = float(spy['close'].iloc[i - 5])
        if spy_5d > 0 and (spy_now - spy_5d) / spy_5d < -0.01:
            return 1
    return 0


def sig_s1096_s929_spy3d_down1pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1096: S929 + SPY 3d down >1% — very short-term market pullback."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 3:
        spy_now = float(spy['close'].iloc[i])
        spy_3d = float(spy['close'].iloc[i - 3])
        if spy_3d > 0 and (spy_now - spy_3d) / spy_3d < -0.01:
            return 1
    return 0


def sig_s1097_s1041_spy_rsi45(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1097: S1041 OR (S929 VIX 18-50 + SPY RSI<45) — add SPY RSI arm to champion.
    SPY somewhat oversold but not necessarily in fear zone may add 2017/2023 trades."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not pd.isna(rs.iloc[i]) and rs.iloc[i] < 45:
        return 1
    return 0


def sig_s1098_s929_macd_spy_rsi50(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1098: S929 + (MACD<0 AND SPY RSI<50) — TSLA downtrend + market below midpoint.
    Dual technical confirmation: TSLA bearish momentum + SPY not in overbought."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is None or hist >= 0:
        return 0
    if not pd.isna(rs.iloc[i]) and rs.iloc[i] < 50:
        return 1
    return 0


def sig_s1099_s1041_10d_pullback(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1099: S1041 OR (S929 VIX 18-50 + 10d TSLA down >3%) — add 10d pullback arm.
    Tests whether a medium-term 10-day pullback captures new trades beyond S1041."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i >= 10:
        c_now = float(tsla['close'].iloc[i])
        c_10d = float(tsla['close'].iloc[i - 10])
        if c_10d > 0 and (c_now - c_10d) / c_10d < -0.03:
            return 1
    return 0


def sig_s1100_s1034_vix18(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1100: S1034 variant with VIX 18-50 (not 15-50) — test the 15-18 contribution.
    S1034 = VIX 15-50 + (3d-selloff OR MACD<0 OR RSI<40). VIX 18-50 version = S1034 minus VIX 15-18."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.25):
                break
    else:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi < 40:
        return 1
    if i >= 3:
        c_now = float(tsla['close'].iloc[i])
        c_3d = float(tsla['close'].iloc[i - 3])
        if c_3d > 0 and (c_now - c_3d) / c_3d < -0.03:
            return 1
    return 0


SIGNALS_P11H: List[Tuple] = [
    ('S1091_s929_spy_rsi45',           sig_s1091_s929_spy_rsi45,               10, 0.20, 50),
    ('S1092_s929_spy_rsi40',           sig_s1092_s929_spy_rsi40,               10, 0.20, 50),
    ('S1093_s929_tsla40_spy_above40',  sig_s1093_s929_tsla_rsi40_spy_above40,  10, 0.20, 50),
    ('S1094_s929_10d_selloff3pct',     sig_s1094_s929_10d_selloff3pct,         10, 0.20, 50),
    ('S1095_s929_spy5d_down1pct',      sig_s1095_s929_spy5d_down1pct,          10, 0.20, 50),
    ('S1096_s929_spy3d_down1pct',      sig_s1096_s929_spy3d_down1pct,          10, 0.20, 50),
    ('S1097_s1041_spy_rsi45',          sig_s1097_s1041_spy_rsi45,              10, 0.20, 50),
    ('S1098_s929_macd_spy_rsi50',      sig_s1098_s929_macd_spy_rsi50,          10, 0.20, 50),
    ('S1099_s1041_10d_pullback',       sig_s1099_s1041_10d_pullback,           10, 0.20, 50),
    ('S1100_s1034_vix18',              sig_s1100_s1034_vix18,                  10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R + SIGNALS_P10S + SIGNALS_P10T
           + SIGNALS_P10U + SIGNALS_P10V + SIGNALS_P10W + SIGNALS_P10X + SIGNALS_P10Y
           + SIGNALS_P10Z + SIGNALS_P11A + SIGNALS_P11B + SIGNALS_P11C + SIGNALS_P11D + SIGNALS_P11E
           + SIGNALS_P11F + SIGNALS_P11G + SIGNALS_P11H)

# ── Phase 11I: TSLA-specific weakness + SPY weekly channel + lag depth ─────────
# New exploration: TSLA falling while SPY flat/up = idiosyncratic selling = highest quality.
# SPY weekly channel: if SPY also at weekly support, strong macro + micro confluence.
# Lag depth: 7% and 10% underperformance thresholds (deeper divergence = stronger signal).

def sig_s1101_s929_tsla3d_spy_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1101: S929 + (TSLA 3d down >3% while SPY 3d UP) — idiosyncratic weakness.
    TSLA falls on stock-specific news/pressure while market holds up = highest quality catch-up."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if not (c_3d > 0 and (c_now - c_3d) / c_3d < -0.03):
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_3d = float(spy['close'].iloc[i - 3])
    if spy_3d > 0 and spy_now >= spy_3d:
        return 1
    return 0


def sig_s1102_s929_lag7pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1102: S929 + lag7pct (TSLA lags SPY by 7% over 20d) — deeper underperformance."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.07):
        return 1
    return 0


def sig_s1103_s929_lag10pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1103: S929 + lag10pct (TSLA lags SPY by 10% over 20d) — extreme underperformance."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.10):
        return 1
    return 0


def sig_s1104_s929_lag5pct_10d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1104: S929 + lag5pct with 10-day lookback (shorter lag window, more recent divergence)."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=10, lag=0.05):
        return 1
    return 0


def sig_s1105_s929_lag5pct_30d(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1105: S929 + lag5pct with 30-day lookback (longer lag window, sustained divergence)."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=30, lag=0.05):
        return 1
    return 0


def _spy_near_lower(sw_df, wk_idx, window, frac=0.25):
    """Check if SPY weekly close is in lower `frac` of its N-week channel."""
    if wk_idx < window:
        return False
    ch = _channel_at(sw_df.iloc[wk_idx - window:wk_idx])
    if ch is None:
        return False
    close_sw = float(sw_df['close'].iloc[wk_idx])
    return _near_lower(close_sw, ch, frac)


def sig_s1106_s929_spy_weekly_chan(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1106: S929 base + SPY weekly also in lower 25% of 20-week channel.
    Both TSLA and SPY at weekly structural support = strongest macro+micro confluence."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if sw is None or len(sw) < 20:
        return 0
    daily_date = tsla.index[i]
    wk_idx = sw.index.searchsorted(daily_date, side='right') - 1
    if _spy_near_lower(sw, wk_idx, 20, 0.25):
        return 1
    return 0


def sig_s1107_s1041_or_spy_weekly(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1107: S1041 OR (S929 + SPY weekly lower channel 20w) — new SPY weekly arm.
    Tests if adding SPY-at-support as OR arm captures new 100% WR trades."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s1106_s929_spy_weekly_chan(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return 0


def sig_s1108_s929_tsla3d_spy_flat(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1108: S929 + (TSLA 3d down >3% AND SPY 3d flat ±1%) — TSLA-only decline.
    More permissive: SPY flat (not necessarily up) while TSLA falls."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if not (c_3d > 0 and (c_now - c_3d) / c_3d < -0.03):
        return 0
    spy_now = float(spy['close'].iloc[i])
    spy_3d = float(spy['close'].iloc[i - 3])
    if spy_3d > 0 and abs((spy_now - spy_3d) / spy_3d) < 0.01:
        return 1
    return 0


def sig_s1109_s1041_or_lag7pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1109: S1041 OR (S929 + lag7pct) — add 7% lag arm to S1041 champion.
    Deeper relative weakness (7% vs 5% in S993) may identify new distinct bounces."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.07):
        return 1
    return 0


def sig_s1110_s1041_or_tsla_spy_diverge(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1110: S1041 OR (S929 + TSLA 3d down >3% while SPY UP).
    Adds idiosyncratic-weakness arm: TSLA-specific panic while market holds."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return sig_s1101_s929_tsla3d_spy_up(i, tsla, spy, vix, tw, sw, rt, rs, w)


SIGNALS_P11I: List[Tuple] = [
    ('S1101_s929_tsla3d_spy_up',       sig_s1101_s929_tsla3d_spy_up,           10, 0.20, 50),
    ('S1102_s929_lag7pct',             sig_s1102_s929_lag7pct,                  10, 0.20, 50),
    ('S1103_s929_lag10pct',            sig_s1103_s929_lag10pct,                 10, 0.20, 50),
    ('S1104_s929_lag5pct_10d',         sig_s1104_s929_lag5pct_10d,             10, 0.20, 50),
    ('S1105_s929_lag5pct_30d',         sig_s1105_s929_lag5pct_30d,             10, 0.20, 50),
    ('S1106_s929_spy_weekly_chan',      sig_s1106_s929_spy_weekly_chan,          10, 0.20, 50),
    ('S1107_s1041_or_spy_weekly',      sig_s1107_s1041_or_spy_weekly,           10, 0.20, 50),
    ('S1108_s929_tsla3d_spy_flat',     sig_s1108_s929_tsla3d_spy_flat,         10, 0.20, 50),
    ('S1109_s1041_or_lag7pct',         sig_s1109_s1041_or_lag7pct,             10, 0.20, 50),
    ('S1110_s1041_or_tsla_spy_div',    sig_s1110_s1041_or_tsla_spy_diverge,    10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R + SIGNALS_P10S + SIGNALS_P10T
           + SIGNALS_P10U + SIGNALS_P10V + SIGNALS_P10W + SIGNALS_P10X + SIGNALS_P10Y
           + SIGNALS_P10Z + SIGNALS_P11A + SIGNALS_P11B + SIGNALS_P11C + SIGNALS_P11D + SIGNALS_P11E
           + SIGNALS_P11F + SIGNALS_P11G + SIGNALS_P11H + SIGNALS_P11I)

# ── Phase 11J: Volume capitulation + Bollinger Bands (new frontiers) ───────────
# Volume: entry-day high volume = capitulation panic selling → strong bounce candidate.
# Bollinger Bands: TSLA below lower BB = statistically oversold on absolute basis.
# These are the two major technical dimensions not yet explored.

def _volume_ratio(tsla_df, i, lookback=20):
    """Volume ratio: current day's volume vs 20-day average volume."""
    if i < lookback:
        return None
    vol_now = float(tsla_df['volume'].iloc[i])
    vol_avg = float(tsla_df['volume'].iloc[i - lookback:i].mean())
    if vol_avg <= 0:
        return None
    return vol_now / vol_avg


def _below_lower_bb(tsla_df, i, period=20, std_mult=2.0):
    """Return True if TSLA close is below lower Bollinger Band (period, std_mult)."""
    if i < period:
        return False
    closes = tsla_df['close'].iloc[i - period:i + 1]
    mean = float(closes.mean())
    std = float(closes.std())
    lower_bb = mean - std_mult * std
    close_now = float(tsla_df['close'].iloc[i])
    return close_now < lower_bb


def sig_s1111_s929_vol15x(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1111: S929 base + high volume (>1.5x 20d avg) — capitulation selling spike.
    Entry-day elevated volume suggests panic selling at channel support = quality bounce."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vr = _volume_ratio(tsla, i, lookback=20)
    if vr is not None and vr > 1.5:
        return 1
    return 0


def sig_s1112_s929_vol20x(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1112: S929 base + very high volume (>2.0x 20d avg) — extreme capitulation."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vr = _volume_ratio(tsla, i, lookback=20)
    if vr is not None and vr > 2.0:
        return 1
    return 0


def sig_s1113_s929_vol12x(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1113: S929 base + moderate volume (>1.2x 20d avg) — above-average selling."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    vr = _volume_ratio(tsla, i, lookback=20)
    if vr is not None and vr > 1.2:
        return 1
    return 0


def sig_s1114_s929_below_bb(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1114: S929 base + TSLA below lower Bollinger Band (20d, 2σ).
    BB breach = statistically oversold condition on absolute price basis."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _below_lower_bb(tsla, i, period=20, std_mult=2.0):
        return 1
    return 0


def sig_s1115_s929_below_bb15(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1115: S929 base + TSLA below 1.5σ Bollinger Band (looser, more signals)."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _below_lower_bb(tsla, i, period=20, std_mult=1.5):
        return 1
    return 0


def sig_s1116_s1041_or_vol20x(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1116: S1041 OR (S929 + extreme volume >2x) — add capitulation arm to champion.
    Tests if extreme volume panic entries capture new 100% WR trades beyond S1041."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return sig_s1112_s929_vol20x(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s1117_s1041_or_below_bb(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1117: S1041 OR (S929 + below lower BB) — add Bollinger Band arm to champion.
    Absolute oversold (BB) vs relative oversold (RSI<40) may capture different bounces."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return sig_s1114_s929_below_bb(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s1118_s929_selloff_vol15x(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1118: S929 + (3d-selloff >3% AND volume >1.5x) — panic selloff with volume confirmation.
    Requires BOTH momentum (3d selloff) AND volume (capitulation) for dual confirmation."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    c_now = float(tsla['close'].iloc[i])
    c_3d = float(tsla['close'].iloc[i - 3])
    if not (c_3d > 0 and (c_now - c_3d) / c_3d < -0.03):
        return 0
    vr = _volume_ratio(tsla, i, lookback=20)
    if vr is not None and vr > 1.5:
        return 1
    return 0


def sig_s1119_s929_bb_vol15x(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1119: S929 + (below BB AND volume >1.5x) — BB breach with capitulation volume.
    Both technically oversold AND high selling volume = strongest panic bottom signal."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _below_lower_bb(tsla, i, period=20, std_mult=2.0):
        return 0
    vr = _volume_ratio(tsla, i, lookback=20)
    if vr is not None and vr > 1.5:
        return 1
    return 0


def sig_s1120_s1041_or_bb_vol(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1120: S1041 OR (S929 + below BB + volume >1.5x) — combined capitulation arm.
    Tests if BB+volume panic bottom adds new 100% WR trades to S1041 champion."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return sig_s1119_s929_bb_vol15x(i, tsla, spy, vix, tw, sw, rt, rs, w)


SIGNALS_P11J: List[Tuple] = [
    ('S1111_s929_vol15x',              sig_s1111_s929_vol15x,                  10, 0.20, 50),
    ('S1112_s929_vol20x',              sig_s1112_s929_vol20x,                  10, 0.20, 50),
    ('S1113_s929_vol12x',              sig_s1113_s929_vol12x,                  10, 0.20, 50),
    ('S1114_s929_below_bb',            sig_s1114_s929_below_bb,                10, 0.20, 50),
    ('S1115_s929_below_bb15',          sig_s1115_s929_below_bb15,              10, 0.20, 50),
    ('S1116_s1041_or_vol20x',          sig_s1116_s1041_or_vol20x,              10, 0.20, 50),
    ('S1117_s1041_or_below_bb',        sig_s1117_s1041_or_below_bb,            10, 0.20, 50),
    ('S1118_s929_selloff_vol15x',      sig_s1118_s929_selloff_vol15x,          10, 0.20, 50),
    ('S1119_s929_bb_vol15x',           sig_s1119_s929_bb_vol15x,               10, 0.20, 50),
    ('S1120_s1041_or_bb_vol',          sig_s1120_s1041_or_bb_vol,              10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R + SIGNALS_P10S + SIGNALS_P10T
           + SIGNALS_P10U + SIGNALS_P10V + SIGNALS_P10W + SIGNALS_P10X + SIGNALS_P10Y
           + SIGNALS_P10Z + SIGNALS_P11A + SIGNALS_P11B + SIGNALS_P11C + SIGNALS_P11D + SIGNALS_P11E
           + SIGNALS_P11F + SIGNALS_P11G + SIGNALS_P11H + SIGNALS_P11I + SIGNALS_P11J)

# ── Phase 11K: Final sweep — channel depth, MACD cross, weekly reversal ────────
# Last unexplored corners: looser channel position (30%), MACD bullish crossover,
# weekly bar recovery (prev week lower than current), and S1041 sub-signal analysis.

def sig_s1121_s929_chan30pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1121: S929 base with looser channel position 30% (vs 25% in S929/S1041).
    More trades from deeper channel zone but possibly lower quality."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.30):  # 30% not 25%
                return 1
    return 0


def sig_s1122_s993_chan30pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1122: S993 conditions with channel 30% (looser) — tests quality at wider zone."""
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.30):
                found = True
                break
    if not found:
        return 0
    if _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
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
    return 0


def _macd_cross_up(tsla_df, i, fast=12, slow=26, signal=9):
    """Return True if MACD histogram crosses from negative to positive at bar i."""
    if i < slow + signal + 1:
        return False
    # Current histogram
    hist_now = _macd_histogram(tsla_df, i)
    # Previous bar histogram
    if i - 1 < slow + signal:
        return False
    # Compute histogram at i-1 directly
    closes_prev = tsla_df['close'].iloc[:i]
    if len(closes_prev) < slow + signal:
        return False
    try:
        ema_fast_prev = closes_prev.ewm(span=fast, adjust=False).mean().iloc[-1]
        ema_slow_prev = closes_prev.ewm(span=slow, adjust=False).mean().iloc[-1]
        macd_line_prev = ema_fast_prev - ema_slow_prev
        # Signal line
        macd_series_prev = (closes_prev.ewm(span=fast, adjust=False).mean()
                            - closes_prev.ewm(span=slow, adjust=False).mean())
        sig_prev = macd_series_prev.ewm(span=signal, adjust=False).mean().iloc[-1]
        hist_prev = float(macd_line_prev - sig_prev)
    except Exception:
        return False
    return hist_prev < 0 and hist_now is not None and hist_now >= 0


def sig_s1123_s929_macd_crossup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1123: S929 + MACD histogram bullish cross (negative → positive today).
    MACD crossover = momentum shift from bearish to bullish AT channel support."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _macd_cross_up(tsla, i):
        return 1
    return 0


def sig_s1124_s929_rsi_cross30(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1124: S929 + RSI crosses back above 30 (was oversold, now recovering).
    RSI oversold recovery = high quality reversal signal at channel support."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 0
    rsi_now = rt.iloc[i]
    rsi_prev = rt.iloc[i - 1]
    if pd.isna(rsi_now) or pd.isna(rsi_prev):
        return 0
    if rsi_prev < 30 and rsi_now >= 30:
        return 1
    return 0


def sig_s1125_s929_weekly_reversal(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1125: S929 + weekly close rising (this week higher than last week).
    Weekly upward reversal at support = confirmed bounce beginning."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if tw is None or len(tw) < 2:
        return 0
    daily_date = tsla.index[i]
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    if wk_idx < 1:
        return 0
    close_now_w = float(tw['close'].iloc[wk_idx])
    close_prev_w = float(tw['close'].iloc[wk_idx - 1])
    if close_prev_w > 0 and close_now_w > close_prev_w:
        return 1
    return 0


def sig_s1126_s929_3d_recovery(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1126: S929 + TSLA close today > 3-day low (recovering off recent bottom).
    Price bouncing off 3-day low = intraday/short-term reversal at support."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    close_now = float(tsla['close'].iloc[i])
    low_3d = float(tsla['low'].iloc[i - 3:i + 1].min())
    open_now = float(tsla['open'].iloc[i])
    # Today closed above the 3-day low AND closed up from open
    if close_now > low_3d and close_now > open_now:
        return 1
    return 0


def sig_s1127_s1041_or_chan30_cond(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1127: S1041 OR (S929 30% channel + precise conditions) — looser channel with precision.
    Test if the 30% channel zone with MACD+RSI filters adds 100% WR new trades."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    if tw is None or len(tw) < 50:
        return 0
    daily_date = tsla.index[i]
    vix_now = float(vix['close'].iloc[i])
    if not (18 <= vix_now <= 50):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 0
    _, atr_5, _, atr_20 = c
    if atr_5 >= 0.75 * atr_20:
        return 0
    wk_idx = tw.index.searchsorted(daily_date, side='right') - 1
    close_w = float(tw['close'].iloc[wk_idx])
    found = False
    for window in (20, 30, 40, 50):
        if wk_idx >= window:
            ch = _channel_at(tw.iloc[wk_idx - window:wk_idx])
            if ch is not None and _near_lower(close_w, ch, 0.30):
                found = True
                break
    if not found:
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        t_rsi = rt.iloc[i]
        if not pd.isna(t_rsi) and t_rsi < 40:
            return 1
    return 0


def sig_s1128_s929_and_lag_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1128: S929 AND (lag5pct AND MACD<0) — 3-way AND intersection.
    Most restrictive: S929 base + both lag5pct AND MACD bearish simultaneously."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _tsla_lagging_spy(tsla, spy, i, lookback=20, lag=0.05):
        return 0
    hist = _macd_histogram(tsla, i)
    if hist is not None and hist < 0:
        return 1
    return 0


def sig_s1129_s929_rsi30_cross(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1129: S929 + RSI was <30 anytime in last 5 days (extreme oversold recovery).
    Not requiring the EXACT cross day, but the recent state of extreme oversold."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 0
    recent_rsi = rt.iloc[i - 5:i + 1]
    if not recent_rsi.isna().all() and (recent_rsi < 30).any():
        return 1
    return 0


def sig_s1130_s929_macd_cross_and_rsi(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1130: S929 + (MACD cross up OR RSI was <30 in last 5d) — reversal signals.
    Either MACD turns bullish OR RSI was extremely oversold recently."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _macd_cross_up(tsla, i):
        return 1
    if i >= 5:
        recent_rsi = rt.iloc[i - 5:i + 1]
        if not recent_rsi.isna().all() and (recent_rsi < 30).any():
            return 1
    return 0


SIGNALS_P11K: List[Tuple] = [
    ('S1121_s929_chan30pct',            sig_s1121_s929_chan30pct,               10, 0.20, 50),
    ('S1122_s993_chan30pct',            sig_s1122_s993_chan30pct,               10, 0.20, 50),
    ('S1123_s929_macd_crossup',         sig_s1123_s929_macd_crossup,            10, 0.20, 50),
    ('S1124_s929_rsi_cross30',          sig_s1124_s929_rsi_cross30,             10, 0.20, 50),
    ('S1125_s929_weekly_reversal',      sig_s1125_s929_weekly_reversal,         10, 0.20, 50),
    ('S1126_s929_3d_recovery',          sig_s1126_s929_3d_recovery,             10, 0.20, 50),
    ('S1127_s1041_or_chan30_cond',      sig_s1127_s1041_or_chan30_cond,         10, 0.20, 50),
    ('S1128_s929_and_lag_macd',         sig_s1128_s929_and_lag_macd,            10, 0.20, 50),
    ('S1129_s929_rsi30_recent',         sig_s1129_s929_rsi30_cross,             10, 0.20, 50),
    ('S1130_s929_macd_cross_rsi',       sig_s1130_s929_macd_cross_and_rsi,     10, 0.20, 50),
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R + SIGNALS_P10S + SIGNALS_P10T
           + SIGNALS_P10U + SIGNALS_P10V + SIGNALS_P10W + SIGNALS_P10X + SIGNALS_P10Y
           + SIGNALS_P10Z + SIGNALS_P11A + SIGNALS_P11B + SIGNALS_P11C + SIGNALS_P11D + SIGNALS_P11E
           + SIGNALS_P11F + SIGNALS_P11G + SIGNALS_P11H + SIGNALS_P11I + SIGNALS_P11J + SIGNALS_P11K)

# ── Phase 11L: Final segmentation — DOW, 200d SMA trend, VIX slope ─────────────
# Goal: Understand what makes S1041's 23 trades high quality.
# These filters segment existing S1041 trades — and test a few new OR arms.

def _tsla_above_200sma(tsla_df, i):
    """True if TSLA close is above its 200-day simple moving average."""
    if i < 200:
        return False
    sma200 = float(tsla_df['close'].iloc[i - 200:i].mean())
    return float(tsla_df['close'].iloc[i]) > sma200


def _vix_declining(vix_df, i, fast=5, slow=10):
    """True if short-term VIX avg < longer-term VIX avg (fear abating)."""
    if i < slow:
        return False
    vix_fast = float(vix_df['close'].iloc[i - fast:i].mean())
    vix_slow = float(vix_df['close'].iloc[i - slow:i].mean())
    return vix_fast < vix_slow


def sig_s1141_s929_above_200sma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1141: S929 + TSLA above 200d SMA (bull market structural support).
    Entries when TSLA is in uptrend = catching dip in bull run = higher quality."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_above_200sma(tsla, i):
        return 1
    return 0


def sig_s1142_s929_below_200sma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1142: S929 + TSLA below 200d SMA (bear/weak trend structural bounce).
    Tests if channel bounces work in downtrends too."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _tsla_above_200sma(tsla, i):
        return 1
    return 0


def sig_s1143_s1041_above_200sma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1143: S1041 filtered to TSLA above 200d SMA only — premium bull-trend trades."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _tsla_above_200sma(tsla, i):
        return 1
    return 0


def sig_s1144_s1041_below_200sma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1144: S1041 filtered to TSLA below 200d SMA — bear/weak trend support test."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _tsla_above_200sma(tsla, i):
        return 1
    return 0


def sig_s1145_s929_vix_declining(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1145: S929 + VIX declining (5d avg < 10d avg) — fear abating = dip with tailwind."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _vix_declining(vix, i):
        return 1
    return 0


def sig_s1146_s929_vix_rising(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1146: S929 + VIX rising (5d avg > 10d avg) — fear building = buying into fear."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _vix_declining(vix, i):
        return 1
    return 0


def sig_s1147_s929_monday(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1147: S929 + Monday entry (day 0) — start-of-week channel support bounce."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if tsla.index[i].weekday() == 0:  # Monday
        return 1
    return 0


def sig_s1148_s929_tue_wed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1148: S929 + Tuesday or Wednesday entry — mid-week support bounces."""
    if sig_s929_s215_4chan_union(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if tsla.index[i].weekday() in (1, 2):  # Tuesday, Wednesday
        return 1
    return 0


def sig_s1149_s1041_vix_declining(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1149: S1041 filtered to VIX declining days only — 'fear cooling' premium entries."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if _vix_declining(vix, i):
        return 1
    return 0


def sig_s1150_s1041_vix_rising(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1150: S1041 filtered to VIX rising days — 'fear building' entry test."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if not _vix_declining(vix, i):
        return 1
    return 0


SIGNALS_P11L: List[Tuple] = [
    ('S1141_s929_above_200sma',        sig_s1141_s929_above_200sma,            10, 0.20, 50),
    ('S1142_s929_below_200sma',        sig_s1142_s929_below_200sma,            10, 0.20, 50),
    ('S1143_s1041_above_200sma',       sig_s1143_s1041_above_200sma,           10, 0.20, 50),
    ('S1144_s1041_below_200sma',       sig_s1144_s1041_below_200sma,           10, 0.20, 50),
    ('S1145_s929_vix_declining',       sig_s1145_s929_vix_declining,           10, 0.20, 50),
    ('S1146_s929_vix_rising',          sig_s1146_s929_vix_rising,              10, 0.20, 50),
    ('S1147_s929_monday',              sig_s1147_s929_monday,                  10, 0.20, 50),
    ('S1148_s929_tue_wed',             sig_s1148_s929_tue_wed,                 10, 0.20, 50),
    ('S1149_s1041_vix_declining',      sig_s1149_s1041_vix_declining,          10, 0.20, 50),
    ('S1150_s1041_vix_rising',         sig_s1150_s1041_vix_rising,             10, 0.20, 50),
]

# ── Phase 11M — Parameter sensitivity + seasonal + micro-structure on S1041 ───
# Explore: hold period (7d/5d), stop loss (10%/15%/25%), Q4+Q1 seasonal,
# NR7 (narrowest 7-bar range = maximum compression), gap-down entry.
# S1151-S1160. Champion baseline: S1041 (n=23, 100% WR, $2,037K, hold=10, stop=20%).

def sig_s1156_s1041_q4q1(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1156: S1041 + Q4/Q1 seasonal (Oct-Mar). TSLA fear months historically strongest.
    Hypothesis: channel support bounces in Oct-Mar (year-end + new-year) have higher quality."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    month = tsla.index[i].month
    return 1 if month in (10, 11, 12, 1, 2, 3) else 0


def sig_s1157_s1041_q1(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1157: S1041 + Q1 only (Jan-Mar). Post-Jan-effect, tax-loss reversals.
    Tests if early-year entries outperform."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    month = tsla.index[i].month
    return 1 if month in (1, 2, 3) else 0


def sig_s1158_s1041_q4(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1158: S1041 + Q4 only (Oct-Dec). Year-end window-dressing and rebalancing.
    Tests if Oct-Dec has asymmetric bounce quality vs rest of year."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    month = tsla.index[i].month
    return 1 if month in (10, 11, 12) else 0


def sig_s1159_s1041_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1159: S1041 + NR7 (today's H-L range is smallest of last 7 bars).
    NR7 = Toby Crabel pattern: maximum range contraction = spring fully coiled.
    Combined with S1041's ATR_compressed → ultra-precision compression entry."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 7:
        return 0
    today_range = float(tsla['high'].iloc[i]) - float(tsla['low'].iloc[i])
    past_ranges = [float(tsla['high'].iloc[i - j]) - float(tsla['low'].iloc[i - j])
                   for j in range(1, 7)]
    return 1 if today_range <= min(past_ranges) else 0


def sig_s1160_s1041_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1160: S1041 + gap-down entry (today open < yesterday close).
    Entry on a gap-down bar = entering into weakness = higher urgency signal.
    Hypothesis: gap-downs at channel support have faster mean-reversion."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 0
    today_open = float(tsla['open'].iloc[i])
    prev_close = float(tsla['close'].iloc[i - 1])
    return 1 if today_open < prev_close else 0


SIGNALS_P11M: List[Tuple] = [
    # -- Parameter variations (same signal function, different hold/stop in tuple) --
    ('S1151_s1041_hold7',    sig_s1041_s993_or_s1034,   7, 0.20, 50),   # 7d hold
    ('S1152_s1041_hold5',    sig_s1041_s993_or_s1034,   5, 0.20, 50),   # 5d hold
    ('S1153_s1041_stop15',   sig_s1041_s993_or_s1034,  10, 0.15, 50),   # 15% stop
    ('S1154_s1041_stop10',   sig_s1041_s993_or_s1034,  10, 0.10, 50),   # 10% stop
    ('S1155_s1041_stop25',   sig_s1041_s993_or_s1034,  10, 0.25, 50),   # 25% stop
    # -- Seasonal filters --
    ('S1156_s1041_q4q1',     sig_s1156_s1041_q4q1,     10, 0.20, 50),   # Oct-Mar
    ('S1157_s1041_q1',       sig_s1157_s1041_q1,       10, 0.20, 50),   # Jan-Mar
    ('S1158_s1041_q4',       sig_s1158_s1041_q4,       10, 0.20, 50),   # Oct-Dec
    # -- Micro-structure filters --
    ('S1159_s1041_nr7',      sig_s1159_s1041_nr7,      10, 0.20, 50),   # NR7 compression
    ('S1160_s1041_gap_down', sig_s1160_s1041_gap_down, 10, 0.20, 50),   # gap-down entry
]

# ── Phase 11N — Gap-up subset, Q2+Q3, NR7+gap combo, union with independent signals ──
# S1161-S1170. Explores:
#   - Gap-up subset (5 high-value non-gap-down trades)
#   - Q2+Q3 complement (Apr-Sep trades)
#   - Max-compression combo (NR7 + gap-down + S1041)
#   - Can MACD crossup (S1123) or RSI<35 (S1088) add unique clean trades beyond S1041?
#   - Large intraday gap (>2%) as independent panic entry
#   - Week-of-month seasonal effects

def sig_s1161_s1041_gap_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1161: S1041 + gap-UP or flat entry (open >= prev close = 5 non-gap-down trades).
    These 5 trades are S1041's highest-value subset: avg ~$125K vs $78K for gap-downs.
    Tests if entering on strength (no weakness gap) has different quality."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 0
    today_open = float(tsla['open'].iloc[i])
    prev_close = float(tsla['close'].iloc[i - 1])
    return 1 if today_open >= prev_close else 0


def sig_s1162_s1041_q2q3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1162: S1041 + Q2/Q3 (Apr-Sep) — complement of Q4+Q1.
    The 11 warm-weather trades: earnings season + summer dips."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    month = tsla.index[i].month
    return 1 if month in (4, 5, 6, 7, 8, 9) else 0


def sig_s1163_s1041_gap_down_nr7(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1163: S1041 + gap-down + NR7 — triple compression: price contracted + gap lower.
    Maximum setup quality: channel support + ATR compressed + NR7 + opening weakness."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 7:
        return 0
    today_open = float(tsla['open'].iloc[i])
    prev_close = float(tsla['close'].iloc[i - 1])
    if today_open >= prev_close:  # not a gap-down
        return 0
    today_range = float(tsla['high'].iloc[i]) - float(tsla['low'].iloc[i])
    past_ranges = [float(tsla['high'].iloc[i - j]) - float(tsla['low'].iloc[i - j])
                   for j in range(1, 7)]
    return 1 if today_range <= min(past_ranges) else 0


def sig_s1164_s1041_or_macd_crossup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1164: S1041 OR S1123 (MACD crossup at channel). Test: does MACD crossup add unique trades?
    S1123 = S929 + MACD histogram crossed from negative to positive — 3 trades, avg=$155K.
    If those 3 are NOT in S1041, this union > $2,037K."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return sig_s1123_s929_macd_crossup(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s1165_s1041_or_rsi35(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1165: S1041 OR S1088 (RSI<35 at channel). Test: does RSI<35 add unique trades?
    S1088 = OR(20/30/40/50w) + compressed + VIX 18-50 + RSI<35 — 3 trades, avg=$118K."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return sig_s1088_s929_rsi35(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s1166_tsla_gap3pct_spy_flat(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1166: TSLA gap-down >3% + SPY flat (<1% gap) — TSLA-specific panic, market stable.
    Fully independent of channel/ATR/VIX: pure gap-panic entry.
    Tests if large TSLA-only gap-downs near weekly channels have edge."""
    if i < 1:
        return 0
    tsla_open = float(tsla['open'].iloc[i])
    tsla_prev  = float(tsla['close'].iloc[i - 1])
    tsla_gap_pct = (tsla_open - tsla_prev) / tsla_prev
    if tsla_gap_pct > -0.03:  # need >3% gap down
        return 0
    spy_open  = float(spy['open'].iloc[i])
    spy_prev   = float(spy['close'].iloc[i - 1])
    spy_gap_pct = abs((spy_open - spy_prev) / spy_prev)
    if spy_gap_pct >= 0.01:  # SPY must be flat (<1% gap either way)
        return 0
    return 1


def sig_s1167_s1041_or_tsla_gap3(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1167: S1041 OR (TSLA gap-down >3% + SPY flat). Tests if panic gaps add clean trades."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1:
        return 1
    return sig_s1166_tsla_gap3pct_spy_flat(i, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s1168_s1041_first_week(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1168: S1041 + first week of month (days 1-7). Month-open accumulation zone.
    Tests if start-of-month entries have seasonal advantage."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if tsla.index[i].day <= 7 else 0


def sig_s1169_s1041_nov_dec(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1169: S1041 + November or December — year-end tax-loss + window-dressing.
    Tests the 'Santa Claus' bounce hypothesis at channel support."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    month = tsla.index[i].month
    return 1 if month in (11, 12) else 0


def sig_s1170_s1041_large_gap_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1170: S1041 + large gap-down (>1.5%). Entering deeper into weakness.
    Tests if larger opening gaps at channel support produce faster/larger reversals."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 0
    today_open = float(tsla['open'].iloc[i])
    prev_close = float(tsla['close'].iloc[i - 1])
    gap_pct = (today_open - prev_close) / prev_close
    return 1 if gap_pct < -0.015 else 0


SIGNALS_P11N: List[Tuple] = [
    ('S1161_s1041_gap_up',          sig_s1161_s1041_gap_up,          10, 0.20, 50),   # 5 high-val gap-ups
    ('S1162_s1041_q2q3',            sig_s1162_s1041_q2q3,            10, 0.20, 50),   # Apr-Sep complement
    ('S1163_s1041_gap_down_nr7',    sig_s1163_s1041_gap_down_nr7,    10, 0.20, 50),   # max compression combo
    ('S1164_s1041_or_macd_crossup', sig_s1164_s1041_or_macd_crossup, 10, 0.20, 50),   # S1041 + MACD crossup union
    ('S1165_s1041_or_rsi35',        sig_s1165_s1041_or_rsi35,        10, 0.20, 50),   # S1041 + RSI<35 union
    ('S1166_tsla_gap3_spy_flat',    sig_s1166_tsla_gap3pct_spy_flat, 10, 0.20, 50),   # independent panic gap
    ('S1167_s1041_or_gap3',         sig_s1167_s1041_or_tsla_gap3,   10, 0.20, 50),   # S1041 + panic gap union
    ('S1168_s1041_first_week',      sig_s1168_s1041_first_week,      10, 0.20, 50),   # first-week entries
    ('S1169_s1041_nov_dec',         sig_s1169_s1041_nov_dec,         10, 0.20, 50),   # Nov-Dec entries
    ('S1170_s1041_large_gap_down',  sig_s1170_s1041_large_gap_down,  10, 0.20, 50),   # >1.5% gap-down
]

# ── Phase 11O — Capstone: channel depth precision, exhaustion, trend context ──
# S1171-S1180. Final frontier: ultra-channel-depth, consecutive-down-days exhaustion,
# RSI crossover timing, SPY/TSLA trend context, moderate-gap sweet spot.

def sig_s1171_s1041_chan10pct(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1171: S1041 with tighter channel filter — price in bottom 10% (vs 25%).
    Tests if the deepest channel dips have better quality than standard 25%.
    Uses same VIX+ATR+OR conditions as S1041 but checks channel pos < 10%."""
    # Re-implement S1041 logic with 10% channel threshold instead of 25%
    # S1041 = S993 OR S1034. Both use _near_lower with 0.25. Test 0.10.
    if i < 50 or i < 20:
        return 0
    vix_now = float(vix['close'].iloc[i]) if i >= 0 else 20.0
    if not (15.0 <= vix_now <= 50.0):
        return 0
    # ATR compression check (same as S1041 base)
    atr_5 = float((tsla['high'].iloc[i-5:i] - tsla['low'].iloc[i-5:i]).mean())
    atr_20 = float((tsla['high'].iloc[i-20:i] - tsla['low'].iloc[i-20:i]).mean())
    if atr_20 <= 0 or atr_5 >= 0.75 * atr_20:
        return 0
    # Check any of the 4 channel windows at 10% depth
    close_now = float(tsla['close'].iloc[i])
    for win in (20, 30, 40, 50):
        if i < win:
            continue
        ch = _channel_at(tsla.iloc[i - win:i])
        if ch and _near_lower(close_now, ch, 0.10):
            return 1
    return 0


def sig_s1172_s1041_moderate_gap(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1172: S1041 + moderate gap-down (0.3%-1.5%) — the 'sweet spot'.
    Avoids tiny gaps (<0.3%) = flat open, and huge gaps (>1.5%) = panic which added losers.
    Tests if the middle gap-down zone is the cleanest entry."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 0
    today_open = float(tsla['open'].iloc[i])
    prev_close = float(tsla['close'].iloc[i - 1])
    gap_pct = (today_open - prev_close) / prev_close
    return 1 if (-0.015 <= gap_pct < -0.003) else 0


def sig_s1173_s1041_tsla_below_50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1173: S1041 + TSLA below 50d MA — medium-term downtrend context.
    Channel bounce from below 50d MA = catching dip in weaker trend = higher urgency.
    Hypothesis: below-50d entries resolve faster (more oversold)."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 0
    ma50 = float(tsla['close'].iloc[i - 50:i].mean())
    return 1 if float(tsla['close'].iloc[i]) < ma50 else 0


def sig_s1174_s1041_spy_above_50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1174: S1041 + SPY above 50d MA — broad market uptrend (buy-the-dip context).
    TSLA underperforms + market healthy = classic catch-up trade.
    Hypothesis: SPY above 50d means dip is TSLA-specific, not systemic."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 0
    spy_ma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if float(spy['close'].iloc[i]) >= spy_ma50 else 0


def sig_s1175_s1041_rsi_crossup(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1175: S1041 + RSI bullish crossover (yesterday RSI<40, today RSI>=40).
    The exact moment RSI crosses back above 40 = oversold recovery signal.
    More precise than RSI<40 (which catches entries already in oversold zone)."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 0
    rsi_today = rt.iloc[i]
    rsi_prev  = rt.iloc[i - 1]
    if pd.isna(rsi_today) or pd.isna(rsi_prev):
        return 0
    return 1 if (rsi_prev < 40.0 and rsi_today >= 40.0) else 0


def sig_s1176_s1041_consec3_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1176: S1041 + 3 consecutive down closes before entry.
    Exhaustion pattern: 3 red candles at channel support = max near-term selling.
    Hypothesis: extended exhaustion accelerates the mean-reversion snap."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    for j in range(1, 4):  # check last 3 bars all closed down
        if float(tsla['close'].iloc[i - j]) >= float(tsla['open'].iloc[i - j]):
            return 0
    return 1


def sig_s1177_s1041_consec3_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1177: S1041 + at least 1 green candle in prior 3 bars (opposite of S1176).
    Control experiment: does having a prior green bar matter?
    Tests if exhaustion requires all-red or mixed is fine."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 3:
        return 0
    for j in range(1, 4):
        if float(tsla['close'].iloc[i - j]) >= float(tsla['open'].iloc[i - j]):
            return 1  # found at least one green bar
    return 0


def sig_s1178_s1041_spy_below_50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1178: S1041 + SPY below 50d MA — bear/correcting broad market.
    Complement of S1174. Both TSLA and SPY weak = systemic sell-off bounce.
    Tests if broad weakness leads to stronger or weaker S1041 bounces."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 0
    spy_ma50 = float(spy['close'].iloc[i - 50:i].mean())
    return 1 if float(spy['close'].iloc[i]) < spy_ma50 else 0


def sig_s1179_s1041_tsla_above_50ma(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1179: S1041 + TSLA above 50d MA — medium-term uptrend, dipping to channel.
    Complement of S1173. TSLA in medium-term uptrend + short-term dip to channel.
    Classic 'buy the dip in uptrend' setup."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 50:
        return 0
    ma50 = float(tsla['close'].iloc[i - 50:i].mean())
    return 1 if float(tsla['close'].iloc[i]) >= ma50 else 0


def sig_s1180_s1041_atr_extreme_compressed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1180: S1041 with STRICTER ATR compression (atr_5 < 0.60 * atr_20 vs 0.75).
    Tests if the deepest compression level (tighter than standard 0.75 threshold)
    produces higher quality entries — 'more coiled spring' hypothesis."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 20:
        return 0
    atr_5  = float((tsla['high'].iloc[i-5:i] - tsla['low'].iloc[i-5:i]).mean())
    atr_20 = float((tsla['high'].iloc[i-20:i] - tsla['low'].iloc[i-20:i]).mean())
    if atr_20 <= 0:
        return 0
    # Stricter: require atr_5 < 60% of atr_20 (vs 75% in standard S1041)
    return 1 if atr_5 < 0.60 * atr_20 else 0


SIGNALS_P11O: List[Tuple] = [
    ('S1171_s1041_chan10pct',        sig_s1171_s1041_chan10pct,        10, 0.20, 50),   # bottom 10% channel
    ('S1172_s1041_moderate_gap',    sig_s1172_s1041_moderate_gap,    10, 0.20, 50),   # 0.3-1.5% gap-down
    ('S1173_s1041_tsla_below50ma',  sig_s1173_s1041_tsla_below_50ma, 10, 0.20, 50),   # TSLA < 50d MA
    ('S1174_s1041_spy_above50ma',   sig_s1174_s1041_spy_above_50ma,  10, 0.20, 50),   # SPY > 50d MA
    ('S1175_s1041_rsi_crossup',     sig_s1175_s1041_rsi_crossup,     10, 0.20, 50),   # RSI cross above 40
    ('S1176_s1041_consec3_down',    sig_s1176_s1041_consec3_down,    10, 0.20, 50),   # 3 red candles before
    ('S1177_s1041_not_all_red',     sig_s1177_s1041_consec3_up,      10, 0.20, 50),   # has green bar prior 3d
    ('S1178_s1041_spy_below50ma',   sig_s1178_s1041_spy_below_50ma,  10, 0.20, 50),   # SPY < 50d MA
    ('S1179_s1041_tsla_above50ma',  sig_s1179_s1041_tsla_above_50ma, 10, 0.20, 50),   # TSLA > 50d MA
    ('S1180_s1041_atr60pct',        sig_s1180_s1041_atr_extreme_compressed, 10, 0.20, 50),  # ATR < 60%
]

# ── Phase 11P — Multi-day confirmation + candle quality filters ───────────────
# S1181-S1190. Novel territory: does requiring S1041 to fire for ≥2 days improve
# quality? Do candle-quality filters (hammer, close>open, new low, SPY lag day)
# select a cleaner subset?

def sig_s1181_s1041_day2_confirm(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1181: S1041 fires today AND fired yesterday (2-day confirmation).
    Reduces false starts: only enter after the setup has been visible for 2 bars.
    Hypothesis: a 2nd day of channel-support setup = stronger mean-reversion pressure."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 0
    return sig_s1041_s993_or_s1034(i - 1, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s1182_s1041_day3_confirm(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1182: S1041 fires for 3 consecutive days before entry.
    Maximum confirmation — waiting for deepest compression window.
    Risk: might enter too late (after partial recovery has started)."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 2:
        return 0
    if sig_s1041_s993_or_s1034(i - 1, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return sig_s1041_s993_or_s1034(i - 2, tsla, spy, vix, tw, sw, rt, rs, w)


def sig_s1183_s1041_close_up(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1183: S1041 + today closed UP (close > open — bullish candle).
    Entry on a green bar: momentum has already started turning.
    Hypothesis: green close at channel support = reversal is initiated."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if float(tsla['close'].iloc[i]) > float(tsla['open'].iloc[i]) else 0


def sig_s1184_s1041_close_down(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1184: S1041 + today closed DOWN (close < open — red candle).
    Entry on a red bar: still under selling pressure. Complement of S1183.
    Hypothesis: red bar entries at support actually get BETTER fills for mean-reversion."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if float(tsla['close'].iloc[i]) < float(tsla['open'].iloc[i]) else 0


def sig_s1185_s1041_new_5d_low(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1185: S1041 + today's low is a new 5-day low (fresh compression bottom).
    The most oversold bar in the recent 5-day window — maximum urgency to revert.
    Hypothesis: new local lows at channel support have the sharpest snap-back."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 5:
        return 0
    today_low = float(tsla['low'].iloc[i])
    prior_lows = [float(tsla['low'].iloc[i - j]) for j in range(1, 5)]
    return 1 if today_low <= min(prior_lows) else 0


def sig_s1186_s1041_spy_down_day(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1186: S1041 + SPY closed DOWN today (broad market selling = TSLA gets caught).
    Market weakness on entry day: TSLA dragged down by SPY, not just TSLA-specific.
    Hypothesis: cross-market selling creates the best dip-buying opportunities."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if float(spy['close'].iloc[i]) < float(spy['open'].iloc[i]) else 0


def sig_s1187_s1041_spy_up_day(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1187: S1041 + SPY closed UP today — broad market green while TSLA weak.
    Pure TSLA-specific weakness: market green but TSLA still at channel support.
    Hypothesis: TSLA-only dip on a green market day = highest alpha catch-up trade."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    return 1 if float(spy['close'].iloc[i]) >= float(spy['open'].iloc[i]) else 0


def sig_s1188_s1041_high_close(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1188: S1041 + close in upper 40% of today's range (close near high).
    Bar closes near its high despite being at channel support = accumulation signal.
    Hypothesis: high-close at channel floor = smart money absorbing supply."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    bar_low  = float(tsla['low'].iloc[i])
    bar_high = float(tsla['high'].iloc[i])
    bar_range = bar_high - bar_low
    if bar_range <= 0:
        return 0
    close_pos = (float(tsla['close'].iloc[i]) - bar_low) / bar_range
    return 1 if close_pos >= 0.60 else 0  # close in top 40% of range


def sig_s1189_s1041_low_close(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1189: S1041 + close in lower 40% of today's range (close near low).
    Bar closes near its low = still under selling pressure.
    Hypothesis: low-close at channel support creates more gap-fill potential next day."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    bar_low  = float(tsla['low'].iloc[i])
    bar_high = float(tsla['high'].iloc[i])
    bar_range = bar_high - bar_low
    if bar_range <= 0:
        return 0
    close_pos = (float(tsla['close'].iloc[i]) - bar_low) / bar_range
    return 1 if close_pos <= 0.40 else 0  # close in bottom 40% of range


def sig_s1190_s1041_first_day(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1190: S1041 fires today but did NOT fire yesterday (first day of the setup).
    Enters on the FIRST day the signal activates, before confirmation.
    Baseline for comparing S1181 (2-day) vs S1190 (1st-day-only) entries."""
    if sig_s1041_s993_or_s1034(i, tsla, spy, vix, tw, sw, rt, rs, w) == 0:
        return 0
    if i < 1:
        return 1  # first bars always qualify
    prev = sig_s1041_s993_or_s1034(i - 1, tsla, spy, vix, tw, sw, rt, rs, w)
    return 1 if prev == 0 else 0  # only fire on first day of run


SIGNALS_P11P: List[Tuple] = [
    ('S1181_s1041_day2_confirm',  sig_s1181_s1041_day2_confirm,  10, 0.20, 50),   # 2-day confirmation
    ('S1182_s1041_day3_confirm',  sig_s1182_s1041_day3_confirm,  10, 0.20, 50),   # 3-day confirmation
    ('S1183_s1041_close_up',      sig_s1183_s1041_close_up,      10, 0.20, 50),   # bullish (green) bar
    ('S1184_s1041_close_down',    sig_s1184_s1041_close_down,    10, 0.20, 50),   # bearish (red) bar
    ('S1185_s1041_new_5d_low',    sig_s1185_s1041_new_5d_low,    10, 0.20, 50),   # new 5d low
    ('S1186_s1041_spy_down_day',  sig_s1186_s1041_spy_down_day,  10, 0.20, 50),   # SPY red day
    ('S1187_s1041_spy_up_day',    sig_s1187_s1041_spy_up_day,    10, 0.20, 50),   # SPY green day
    ('S1188_s1041_high_close',    sig_s1188_s1041_high_close,    10, 0.20, 50),   # close near high
    ('S1189_s1041_low_close',     sig_s1189_s1041_low_close,     10, 0.20, 50),   # close near low
    ('S1190_s1041_first_day',     sig_s1190_s1041_first_day,     10, 0.20, 50),   # 1st day only
]


# ── Phase 11Q — RSI level + VIX cooldown (no channel, no ATR requirement) ─────
# Question: does "oversold + fear fading" work WITHOUT the weekly channel gate?
# If yes: S1041's channel/ATR gates are over-constrained and S526/S522 can be fixed
#         by swapping MACD/RSI-direction for RSI-level.
# If no:  confirms the channel + ATR gates are load-bearing (not redundant).
# Baseline for comparison: S526 (MACD+VIX_CD), S522 (channel+VIX_CD+RSI_rising).


def sig_s1191_rsi40_vix_cd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1191: RSI < 40 + VIX cooldown — no channel, no ATR requirement.
    The RSI analog of S526: oversold level replaces MACD turning.
    Tests whether 'stock looks washed out + fear fading' is sufficient without
    requiring price to be at a specific channel boundary."""
    if i < 252 + 10:
        return 0
    if float(rt.iloc[i]) >= 40.0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s1192_rsi35_vix_cd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1192: RSI < 35 + VIX cooldown — tighter RSI threshold, no channel/ATR.
    Requires deeper oversold (vs S1191's <40) but same fear-fading gate.
    If S1192 >> S1191: depth of oversold matters more than the level itself."""
    if i < 252 + 10:
        return 0
    if float(rt.iloc[i]) >= 35.0:
        return 0
    return 1 if _vix_was_elevated_now_cooling(vix, i) else 0


def sig_s1193_rsi40_vix_cd_atr(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1193: RSI < 40 + VIX cooldown + ATR compressed (atr_5 < 0.75 * atr_20).
    Adds the ATR gate from S1041 to the RSI+VIX_CD base.
    Tests: does ATR compression add value on top of RSI oversold + VIX cooldown?
    If S1193 >> S1191: the 'coiling spring' gate is independently valuable."""
    if i < 252 + 10:
        return 0
    if float(rt.iloc[i]) >= 40.0:
        return 0
    if not _vix_was_elevated_now_cooling(vix, i):
        return 0
    c = _atr_components(tsla, i)
    if c is None:
        return 1   # can't compute ATR, don't block
    _, atr_5, _, atr_20 = c
    return 1 if atr_5 < 0.75 * atr_20 else 0


def sig_s1194_rsi40_vix_cd_macd(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1194: RSI < 40 + VIX cooldown + MACD histogram turning positive.
    Triple confirmation: oversold level + fear fading + momentum inflecting.
    All three signals at daily TF simultaneously — most demanding of the set."""
    if i < 252 + 10:
        return 0
    if float(rt.iloc[i]) >= 40.0:
        return 0
    if not _vix_was_elevated_now_cooling(vix, i):
        return 0
    return 1 if sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) else 0


SIGNALS_P11Q: List[Tuple] = [
    ('S1191_rsi40_vix_cd',        sig_s1191_rsi40_vix_cd,       10, 0.20, 50),  # RSI<40 + VIX_CD, no channel
    ('S1192_rsi35_vix_cd',        sig_s1192_rsi35_vix_cd,       10, 0.20, 50),  # RSI<35 + VIX_CD, tighter
    ('S1192_rsi35_vix_cd_h20',    sig_s1192_rsi35_vix_cd,       20, 0.20, 50),  # same, hold=20d
    ('S1192_rsi35_vix_cd_h30',    sig_s1192_rsi35_vix_cd,       30, 0.20, 50),  # same, hold=30d
    ('S1193_rsi40_vix_cd_atr',    sig_s1193_rsi40_vix_cd_atr,   10, 0.20, 50),  # + ATR compressed
    ('S1194_rsi40_vix_cd_macd',   sig_s1194_rsi40_vix_cd_macd,  10, 0.20, 50),  # + MACD turning
]

# ── Phase 11R — RSI smooth / divergence / upward hook on S1041 ────────────────
# Question: does the app's smoothed RSI (3-period MA of RSI-14) or classic
# bullish RSI divergence improve S1041 entries beyond raw RSI < 40?
#
# Signals:
#   S1195 — S1041 + RSI upward hook from below 40 (RSI[i] > RSI[i-2], RSI[i-2] < 40)
#   S1196 — S1041 + RSI_smooth < 40 (3-period MA of RSI, same threshold)
#   S1197 — S1041 + RSI bullish divergence (price lower low, RSI higher low, 10d window)
#   S1198 — S1041 + RSI_smooth divergence (same but on smoothed RSI)
#   S1199 — S1041 + RSI hook OR divergence (either condition)
#   S1200 — S1041 + RSI_smooth just bottomed (smooth[i] > smooth[i-1], smooth[i-1] <= smooth[i-2])
#
# Hypotheses:
#   - Hook/divergence = RSI "leading" signal fires before or at same time as price reversal
#   - Smooth reduces noise → may filter out false oversold readings
#   - If S1196 >> S1041: smoothed threshold is cleaner gate than raw RSI
#   - If S1197/S1198 >> S1041: divergence captures better entries than level alone


def _rsi_smooth(rt, i, period=3):
    """3-period simple MA of RSI at index i."""
    if i < period - 1:
        return float('nan')
    return float(rt.iloc[i - period + 1:i + 1].mean())


def _rsi_bullish_div(closes, rt, i, lookback=10, min_rsi_edge=1.5):
    """Bullish RSI divergence: price at/near its N-day low, RSI above its N-day low.
    Requires: current RSI > RSI-at-price-low + min_rsi_edge (avoid noise).
    Returns True if divergence detected."""
    if i < lookback:
        return False
    price_window = closes.iloc[i - lookback:i + 1]
    rsi_window   = rt.iloc[i - lookback:i + 1]
    price_low_idx = price_window.idxmin()
    rsi_at_price_low = rsi_window.loc[price_low_idx]
    curr_price = closes.iloc[i]
    curr_rsi   = float(rt.iloc[i])
    price_low  = float(price_window.min())
    # Price at or within 2% above its N-day low
    if curr_price > price_low * 1.02:
        return False
    # RSI meaningfully above its value at the price low
    return curr_rsi > float(rsi_at_price_low) + min_rsi_edge


def _rsi_smooth_div(closes, rt, i, lookback=10, min_rsi_edge=1.5, smooth_period=3):
    """Same as _rsi_bullish_div but uses smoothed RSI instead of raw."""
    if i < lookback + smooth_period:
        return False
    price_window = closes.iloc[i - lookback:i + 1]
    price_low_idx_pos = price_window.values.argmin()  # position within window
    abs_low_idx = i - lookback + price_low_idx_pos
    smooth_at_low = _rsi_smooth(rt, abs_low_idx, smooth_period)
    curr_smooth   = _rsi_smooth(rt, i, smooth_period)
    curr_price    = float(closes.iloc[i])
    price_low     = float(price_window.min())
    if curr_price > price_low * 1.02:
        return False
    if any(v != v for v in [smooth_at_low, curr_smooth]):  # nan check
        return False
    return curr_smooth > smooth_at_low + min_rsi_edge


def _s1041_core(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1041 core conditions minus the RSI < 40 gate. Used by 11R variants."""
    # Warmup
    if i < 252 + 20:
        return False
    # Weekly channel: at/near lower 25% of 20/30/40/50-week channel
    in_weekly_channel = False
    for win in (20, 30, 40, 50):
        if len(tw) <= win:
            continue
        tw_slice = tw.iloc[len(tw) - win:]
        ch = _channel_at(tw_slice)
        if ch and _near_lower(float(tw['close'].iloc[-1]), ch, 0.25):
            in_weekly_channel = True
            break
    if not in_weekly_channel:
        return False
    # ATR compressed
    c = _atr_components(tsla, i)
    if c is None:
        return False
    _, atr_5, _, atr_20 = c
    if not (atr_5 < 0.75 * atr_20):
        return False
    # VIX in fear regime (15-50)
    vix_now = float(vix['close'].iloc[i])
    if not (15.0 <= vix_now <= 50.0):
        return False
    # At least one of: 3d-selloff, MACD<0, RSI<40, SPY-TSLA div, lag5pct
    closes = tsla['close']
    lag5   = (float(closes.iloc[i]) - float(closes.iloc[i - 5])) / float(closes.iloc[i - 5])
    pct3   = (float(closes.iloc[i]) - float(closes.iloc[i - 3])) / float(closes.iloc[i - 3])
    rsi_now = float(rt.iloc[i])
    macd_ok = sig_s333_s215_macd_turning(i, tsla, spy, vix, tw, sw, rt, rs, w) == 1 if i >= 26 else False
    # SPY-TSLA divergence (simplified: SPY >+3% last 20d, TSLA flat/<0)
    spy_20 = (float(spy['close'].iloc[i]) - float(spy['close'].iloc[i - 20])) / float(spy['close'].iloc[i - 20])
    tsla_20 = (float(closes.iloc[i]) - float(closes.iloc[i - 20])) / float(closes.iloc[i - 20])
    spy_div = spy_20 > 0.03 and tsla_20 < spy_20 - 0.04
    arm_ok = (lag5 <= -0.03 or pct3 <= -0.02 or rsi_now < 40 or macd_ok or spy_div)
    return arm_ok


def sig_s1195_rsi_hook(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1195: S1041 core + RSI upward hook from oversold.
    Hook = RSI[i] > RSI[i-2] AND RSI[i-2] < 40 (RSI turning up from below 40).
    Tests: does a rising-RSI-from-oversold filter sharpen S1041 entries?"""
    if i < 252 + 20:
        return 0
    rsi_now  = float(rt.iloc[i])
    rsi_2ago = float(rt.iloc[i - 2])
    hook = (rsi_now > rsi_2ago) and (rsi_2ago < 40.0)
    if not hook:
        return 0
    return 1 if _s1041_core(i, tsla, spy, vix, tw, sw, rt, rs, w) else 0


def sig_s1196_rsi_smooth(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1196: S1041 core + smoothed RSI < 40 (3-period MA of RSI-14).
    Replaces raw RSI < 40 with the smoothed version. Tests whether
    the smoothed RSI is a cleaner/tighter oversold gate than raw RSI."""
    if i < 252 + 20:
        return 0
    smooth = _rsi_smooth(rt, i, 3)
    if smooth != smooth or smooth >= 40.0:  # nan or not oversold
        return 0
    return 1 if _s1041_core(i, tsla, spy, vix, tw, sw, rt, rs, w) else 0


def sig_s1197_rsi_div(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1197: S1041 core + classic bullish RSI divergence (10-day window).
    Price at/near 10d low, raw RSI above its level at that price low (+1.5 pts).
    Tests whether divergence is additive to or better than RSI < 40 level."""
    if i < 252 + 20:
        return 0
    if not _rsi_bullish_div(tsla['close'], rt, i, lookback=10):
        return 0
    return 1 if _s1041_core(i, tsla, spy, vix, tw, sw, rt, rs, w) else 0


def sig_s1198_rsi_smooth_div(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1198: S1041 core + smoothed RSI bullish divergence.
    Same as S1197 but divergence measured on 3-period-smoothed RSI.
    Tests whether smoothed divergence reduces noise vs raw divergence."""
    if i < 252 + 20:
        return 0
    if not _rsi_smooth_div(tsla['close'], rt, i, lookback=10):
        return 0
    return 1 if _s1041_core(i, tsla, spy, vix, tw, sw, rt, rs, w) else 0


def sig_s1199_rsi_hook_or_div(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1199: S1041 core + (hook OR divergence).
    Fires if either RSI upward hook OR raw divergence is present.
    Broader than S1195/S1197 alone — tests combined vs individual."""
    if i < 252 + 20:
        return 0
    rsi_now  = float(rt.iloc[i])
    rsi_2ago = float(rt.iloc[i - 2])
    hook = (rsi_now > rsi_2ago) and (rsi_2ago < 40.0)
    div  = _rsi_bullish_div(tsla['close'], rt, i, lookback=10)
    if not (hook or div):
        return 0
    return 1 if _s1041_core(i, tsla, spy, vix, tw, sw, rt, rs, w) else 0


def sig_s1200_rsi_smooth_bottomed(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1200: S1041 core + smoothed RSI just bottomed.
    Smooth[i] > Smooth[i-1] AND Smooth[i-1] <= Smooth[i-2] (V-turn in smooth).
    The leading edge signal: smooth RSI inflects before price does."""
    if i < 252 + 20:
        return 0
    s0 = _rsi_smooth(rt, i,     3)
    s1 = _rsi_smooth(rt, i - 1, 3)
    s2 = _rsi_smooth(rt, i - 2, 3)
    if any(v != v for v in [s0, s1, s2]):  # nan check
        return 0
    bottomed = (s0 > s1) and (s1 <= s2)
    if not bottomed:
        return 0
    return 1 if _s1041_core(i, tsla, spy, vix, tw, sw, rt, rs, w) else 0


SIGNALS_P11R: List[Tuple] = [
    ('S1195_rsi_hook',          sig_s1195_rsi_hook,         10, 0.20, 50),  # RSI upward hook from <40
    ('S1196_rsi_smooth',        sig_s1196_rsi_smooth,        10, 0.20, 50),  # smoothed RSI < 40
    ('S1197_rsi_div',           sig_s1197_rsi_div,           10, 0.20, 50),  # classic RSI divergence
    ('S1198_rsi_smooth_div',    sig_s1198_rsi_smooth_div,    10, 0.20, 50),  # smoothed RSI divergence
    ('S1199_rsi_hook_or_div',   sig_s1199_rsi_hook_or_div,   10, 0.20, 50),  # hook OR divergence
    ('S1200_rsi_smooth_bot',    sig_s1200_rsi_smooth_bottomed, 10, 0.20, 50), # smooth RSI V-turn
    # Hold variants for the best performer(s) — add after seeing results
]

# ── Phase 11S — Weekly RSI extremes + S1041 intersection ─────────────────────
#
# Motivated by rsi_bottom_targets.py (Feb 2026):
#   weekly RSI<30 + daily RSI<35 → n=9, E[30d]=$78,773, 100% hit +10%/+20%
#   weekly RSI<30 standalone     → n=13, 100% hit all targets, E[30d]=$75K
#
# Three questions:
#   1. How does standalone weekly RSI<30 perform in a full IS/OOS backtest?
#   2. Does adding weekly RSI gate to S1041 improve it (higher WR, better P&L)?
#   3. Is weekly RSI<30+daily RSI<35 additive to S1041 (adds new quality trades)?
#
# Signals:
#   S1201 — weekly RSI<30 only (standalone, no channel)
#   S1202 — weekly RSI<35 only (broader, n=37 over 10yr)
#   S1203 — weekly RSI<30 + daily RSI<35 (best combo from rsi_bottom_targets)
#   S1204 — weekly RSI<30 + daily RSI<40 (looser daily threshold)
#   S1205 — weekly RSI<35 + daily RSI<40
#   S1206 — S1041_core + weekly RSI<35 (does weekly RSI sharpen S1041 entries?)
#   S1207 — S1041_core + weekly RSI<40 (broader weekly gate on S1041)
#   S1208 — weekly RSI<30 + daily RSI<35 + VIX 15-50 (add S1041's VIX gate)
#   S1209 — weekly RSI<30 + daily RSI<35 + ATR_compressed (add coiling-spring)
#   S1210 — S1041_core OR (weekly RSI<30 + daily RSI<35) [union — additive?]

# Module-level cache for weekly RSI computation (avoids O(n²) recomputation)
_wkly_rsi_cache: dict = {}


def _weekly_rsi_at(tw, date, period: int = 14):
    """
    Return weekly RSI(14) as of `date` using the tw (weekly OHLCV) DataFrame.
    Returns None if insufficient data or NaN.
    Uses a module-level cache keyed on id(tw) to avoid re-computing per call.
    """
    if tw is None or len(tw) < period + 2:
        return None
    ck = id(tw)
    if ck not in _wkly_rsi_cache:
        _wkly_rsi_cache[ck] = _rsi(tw['close'], period)
    rsi_w = _wkly_rsi_cache[ck]
    wi = rsi_w.index.searchsorted(date + pd.Timedelta(days=1), side='left') - 1
    if wi < period:
        return None
    v = float(rsi_w.iloc[wi])
    return None if np.isnan(v) else v


def _atr_compressed_at(tsla, i: int) -> bool:
    """
    True if ATR is compressed: ATR_5 < 0.75 × ATR_20.
    Uses Wilder-smoothed (EWM) ATR on the last 25 daily bars.
    """
    if i < 25:
        return False
    hi = tsla['high'].values
    lo = tsla['low'].values
    cl = tsla['close'].values
    # True range for each bar
    tr = np.array([
        max(hi[j] - lo[j],
            abs(hi[j] - cl[j - 1]),
            abs(lo[j] - cl[j - 1]))
        for j in range(i - 24, i + 1)
    ])
    # Simple average (approximation of Wilder for short lookbacks)
    atr5  = float(np.mean(tr[-5:]))
    atr20 = float(np.mean(tr[-20:]))
    return atr20 > 0 and atr5 < 0.75 * atr20


def sig_s1201_wkly_rsi30(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1201: weekly RSI<30 only — pure RSI extreme, no channel condition.
    Baseline: does the 100%-hit-rate finding from rsi_bottom_targets hold
    in a full position-sized backtest with stops?"""
    if i < 252 or tw is None:
        return 0
    wr = _weekly_rsi_at(tw, tsla.index[i])
    return 1 if (wr is not None and wr < 30.0) else 0


def sig_s1202_wkly_rsi35(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1202: weekly RSI<35 only — broader threshold (n=37 over 10yr).
    Tests frequency/quality trade-off vs S1201."""
    if i < 252 or tw is None:
        return 0
    wr = _weekly_rsi_at(tw, tsla.index[i])
    return 1 if (wr is not None and wr < 35.0) else 0


def sig_s1203_wkly30_daily35(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1203: weekly RSI<30 + daily RSI<35 — best combo from rsi_bottom_targets.
    E[30d]=$78,773, 100% hit all profit targets over 10yr in forward-return test."""
    if i < 252 or tw is None:
        return 0
    if float(rt.iloc[i]) >= 35.0:
        return 0
    wr = _weekly_rsi_at(tw, tsla.index[i])
    return 1 if (wr is not None and wr < 30.0) else 0


def sig_s1204_wkly30_daily40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1204: weekly RSI<30 + daily RSI<40 — looser daily gate.
    How many trades does relaxing daily RSI from <35 to <40 add?"""
    if i < 252 or tw is None:
        return 0
    if float(rt.iloc[i]) >= 40.0:
        return 0
    wr = _weekly_rsi_at(tw, tsla.index[i])
    return 1 if (wr is not None and wr < 30.0) else 0


def sig_s1205_wkly35_daily40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1205: weekly RSI<35 + daily RSI<40 — less restrictive version.
    Covers the weekly RSI<35 range (n=37 over 10yr) with daily confirmation."""
    if i < 252 or tw is None:
        return 0
    if float(rt.iloc[i]) >= 40.0:
        return 0
    wr = _weekly_rsi_at(tw, tsla.index[i])
    return 1 if (wr is not None and wr < 35.0) else 0


def sig_s1206_s1041core_wkly35(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1206: S1041_core conditions AND weekly RSI<35.
    Tests: does requiring weekly RSI<35 at S1041 entries improve WR/P&L?
    Reduces trade count from S1041's 23 — question is whether kept trades score higher."""
    if tw is None:
        return 0
    if not _s1041_core(i, tsla, spy, vix, tw, sw, rt, rs, w):
        return 0
    wr = _weekly_rsi_at(tw, tsla.index[i])
    return 1 if (wr is not None and wr < 35.0) else 0


def sig_s1207_s1041core_wkly40(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1207: S1041_core conditions AND weekly RSI<40 — broader weekly gate.
    How many of S1041's 23+ trades have weekly RSI<40?"""
    if tw is None:
        return 0
    if not _s1041_core(i, tsla, spy, vix, tw, sw, rt, rs, w):
        return 0
    wr = _weekly_rsi_at(tw, tsla.index[i])
    return 1 if (wr is not None and wr < 40.0) else 0


def sig_s1208_wkly30_daily35_vix(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1208: weekly RSI<30 + daily RSI<35 + VIX 15-50.
    Adds S1041's VIX regime gate to the pure RSI signal.
    VIX gate should block complacency (VIX<15) and systemic crisis (VIX>50)."""
    if i < 252 or tw is None or len(vix) <= i:
        return 0
    v = float(vix.iloc[i])
    if not (15.0 <= v <= 50.0):
        return 0
    if float(rt.iloc[i]) >= 35.0:
        return 0
    wr = _weekly_rsi_at(tw, tsla.index[i])
    return 1 if (wr is not None and wr < 30.0) else 0


def sig_s1209_wkly30_daily35_atr(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1209: weekly RSI<30 + daily RSI<35 + ATR_compressed.
    Adds the coiling-spring filter (ATR_5 < 0.75×ATR_20).
    If compression co-occurs with RSI extreme, the bounce may be sharper."""
    if i < 252 or tw is None:
        return 0
    if not _atr_compressed_at(tsla, i):
        return 0
    if float(rt.iloc[i]) >= 35.0:
        return 0
    wr = _weekly_rsi_at(tw, tsla.index[i])
    return 1 if (wr is not None and wr < 30.0) else 0


def sig_s1210_s1041_or_wkly30(i, tsla, spy, vix, tw, sw, rt, rs, w):
    """S1210: S1041_core OR (weekly RSI<30 + daily RSI<35) — union test.
    S1041_core already fires on S1041's trades; weekly RSI<30+daily RSI<35
    fires on additional RSI-extreme trades outside S1041's channel structure.
    Union P&L = S1041 + any incremental RSI trades not already in S1041."""
    if tw is None:
        return 0
    # Check S1041 core (channel + ATR + VIX + momentum)
    if _s1041_core(i, tsla, spy, vix, tw, sw, rt, rs, w):
        return 1
    # Also fire on pure weekly RSI extreme (may catch trades S1041 misses)
    if i < 252:
        return 0
    if float(rt.iloc[i]) >= 35.0:
        return 0
    wr = _weekly_rsi_at(tw, tsla.index[i])
    return 1 if (wr is not None and wr < 30.0) else 0


SIGNALS_P11S: List[Tuple] = [
    ('S1201_wkly_rsi30',        sig_s1201_wkly_rsi30,        10, 0.20, 50),  # weekly RSI<30 standalone
    ('S1202_wkly_rsi35',        sig_s1202_wkly_rsi35,        10, 0.20, 50),  # weekly RSI<35 standalone
    ('S1203_wkly30_daily35',    sig_s1203_wkly30_daily35,    10, 0.20, 50),  # best combo from targets
    ('S1204_wkly30_daily40',    sig_s1204_wkly30_daily40,    10, 0.20, 50),  # looser daily threshold
    ('S1205_wkly35_daily40',    sig_s1205_wkly35_daily40,    10, 0.20, 50),  # broader weekly+daily
    ('S1206_s1041core_wkly35',  sig_s1206_s1041core_wkly35,  10, 0.20, 50),  # S1041 core + wkly RSI<35
    ('S1207_s1041core_wkly40',  sig_s1207_s1041core_wkly40,  10, 0.20, 50),  # S1041 core + wkly RSI<40
    ('S1208_wkly30_d35_vix',    sig_s1208_wkly30_daily35_vix, 10, 0.20, 50), # +VIX gate
    ('S1209_wkly30_d35_atr',    sig_s1209_wkly30_daily35_atr, 10, 0.20, 50), # +ATR compressed
    ('S1210_s1041_or_wkly30',   sig_s1210_s1041_or_wkly30,   10, 0.20, 50),  # union
]

SIGNALS = (SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D
           + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L
           + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S
           + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y
           + SIGNALS_P7Z + SIGNALS_P8A + SIGNALS_P8B + SIGNALS_P8C + SIGNALS_P8D + SIGNALS_P8E
           + SIGNALS_P8F + SIGNALS_P8G + SIGNALS_P8H + SIGNALS_P8I + SIGNALS_P8J + SIGNALS_P8K
           + SIGNALS_P8L + SIGNALS_P8M + SIGNALS_P8N + SIGNALS_P8O + SIGNALS_P8P + SIGNALS_P8Q
           + SIGNALS_P8R + SIGNALS_P8S + SIGNALS_P8T + SIGNALS_P8U + SIGNALS_P8V + SIGNALS_P8W
           + SIGNALS_P8X + SIGNALS_P8Y + SIGNALS_P8Z + SIGNALS_P9A + SIGNALS_P9B + SIGNALS_P9C
           + SIGNALS_P9D + SIGNALS_P9E + SIGNALS_P9F + SIGNALS_P9G + SIGNALS_P9H + SIGNALS_P9I
           + SIGNALS_P9J + SIGNALS_P9K + SIGNALS_P9L + SIGNALS_P9M + SIGNALS_P9N + SIGNALS_P9O
           + SIGNALS_P9P + SIGNALS_P9Q + SIGNALS_P9R + SIGNALS_P9S + SIGNALS_P9T
           + SIGNALS_P9U + SIGNALS_P9V + SIGNALS_P9W + SIGNALS_P9X + SIGNALS_P9Y + SIGNALS_P9Z
           + SIGNALS_P10A + SIGNALS_P10B + SIGNALS_P10C + SIGNALS_P10D + SIGNALS_P10E
           + SIGNALS_P10F + SIGNALS_P10G + SIGNALS_P10H + SIGNALS_P10I + SIGNALS_P10J
           + SIGNALS_P10K + SIGNALS_P10L + SIGNALS_P10M + SIGNALS_P10N + SIGNALS_P10O
           + SIGNALS_P10P + SIGNALS_P10Q + SIGNALS_P10R + SIGNALS_P10S + SIGNALS_P10T
           + SIGNALS_P10U + SIGNALS_P10V + SIGNALS_P10W + SIGNALS_P10X + SIGNALS_P10Y
           + SIGNALS_P10Z + SIGNALS_P11A + SIGNALS_P11B + SIGNALS_P11C + SIGNALS_P11D + SIGNALS_P11E
           + SIGNALS_P11F + SIGNALS_P11G + SIGNALS_P11H + SIGNALS_P11I + SIGNALS_P11J + SIGNALS_P11K
           + SIGNALS_P11L + SIGNALS_P11M + SIGNALS_P11N + SIGNALS_P11O + SIGNALS_P11P
           + SIGNALS_P11Q
           + SIGNALS_P11R
           + SIGNALS_P11S)

# ── sentinel — do not remove ──────────────────────────────────────────────────


# ── Phase 5 (weekly) — Weekly bar signals ─────────────────────────────────────
# Primary bars are weekly OHLCV (resampled from daily).
# "max_hold_days" = max hold in weeks (same engine, weekly bars passed).
# Wider stops (8-10%) appropriate for multi-week candles.
# Full 10yr history (2015-2025) — same as daily phases.

def sig_w01_weekly_channel_bounce(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """W01: TSLA weekly channel bounce — near lower 25% of 20-week channel."""
    if i < win:
        return 0
    ch = _channel_at(tsla.iloc[i - win:i])
    if ch is None:
        return 0
    return 1 if _near_lower(tsla['close'].iloc[i], ch, 0.25) else 0


def sig_w02_spy_weekly_lag(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """W02: SPY near 52-week high, TSLA lagging ≥8% (weekly lag effect)."""
    lookback = 52
    if i < lookback:
        return 0
    spy_now      = spy['close'].iloc[i]
    spy_high     = spy['high'].iloc[i - lookback:i].max()
    tsla_now     = tsla['close'].iloc[i]
    tsla_high    = tsla['high'].iloc[i - lookback:i].max()
    spy_strong   = spy_now >= spy_high * 0.95   # within 5% of 52w high
    tsla_lagging = tsla_now < tsla_high * 0.92  # lagging ≥8%
    return 1 if (spy_strong and tsla_lagging) else 0


def sig_w03_weekly_rsi_divergence(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """W03: TSLA weekly RSI < 40 + SPY weekly RSI > 50 (RSI divergence)."""
    if i < 20:
        return 0
    t_rsi = rt.iloc[i]
    s_rsi = rs.iloc[i]
    if pd.isna(t_rsi) or pd.isna(s_rsi):
        return 0
    return 1 if (t_rsi < 40 and s_rsi > 50) else 0


def sig_w04_weekly_channel_vix_moderate(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """W04: Weekly channel bounce + VIX weekly close < 30 (no extreme panic)."""
    if sig_w01_weekly_channel_bounce(i, tsla, spy, vix, tw, sw, rt, rs, win) == 0:
        return 0
    if vix['close'].iloc[i] >= 30:
        return 0
    return 1


def sig_w05_weekly_channel_spy_uptrend(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """W05: Weekly channel bounce + SPY above 20-week MA (broad uptrend)."""
    if sig_w01_weekly_channel_bounce(i, tsla, spy, vix, tw, sw, rt, rs, win) == 0:
        return 0
    if i < 20:
        return 0
    spy_ma20w = spy['close'].iloc[i - 20:i].mean()
    if spy['close'].iloc[i] < spy_ma20w:
        return 0
    return 1


def sig_w06_spy_weekly_lag_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """W06: Weekly SPY lag + VIX weekly 15–35 (same sweet spot as daily S29)."""
    if sig_w02_spy_weekly_lag(i, tsla, spy, vix, tw, sw, rt, rs, win) == 0:
        return 0
    vix_now = vix['close'].iloc[i]
    if not (15 <= vix_now <= 35):
        return 0
    return 1


def sig_w07_weekly_channel_rsi_oversold(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """W07: Weekly channel bounce + TSLA weekly RSI < 45."""
    if sig_w01_weekly_channel_bounce(i, tsla, spy, vix, tw, sw, rt, rs, win) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if pd.isna(t_rsi) or t_rsi > 45:
        return 0
    return 1


def sig_w08_spy_weekly_lag_no_bear(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """W08: Weekly SPY lag + SPY not in bear (not >15% below 52w high)."""
    if sig_w02_spy_weekly_lag(i, tsla, spy, vix, tw, sw, rt, rs, win) == 0:
        return 0
    spy_52w_high = spy['high'].iloc[max(0, i - 52):i].max()
    if spy['close'].iloc[i] < spy_52w_high * 0.85:
        return 0
    return 1


def sig_w09_union_weekly(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """W09: Union of W06 (VIX-gated lag) + W05 (channel+SPY trend)."""
    if sig_w06_spy_weekly_lag_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, win) == 1:
        return 1
    if sig_w05_weekly_channel_spy_uptrend(i, tsla, spy, vix, tw, sw, rt, rs, win) == 1:
        return 1
    return 0


def sig_w10_best_weekly(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """W10: W09 union + TSLA weekly RSI not overbought (< 60)."""
    if sig_w09_union_weekly(i, tsla, spy, vix, tw, sw, rt, rs, win) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi > 60:
        return 0
    return 1


# Phase 5: (name, fn, max_hold_weeks, stop_pct, channel_window_weeks)
SIGNALS_P5: List[Tuple] = [
    ('W01_weekly_channel_bounce',       sig_w01_weekly_channel_bounce,       8, 0.08, 20),
    ('W02_spy_weekly_lag',              sig_w02_spy_weekly_lag,              8, 0.08, 20),
    ('W03_weekly_rsi_divergence',       sig_w03_weekly_rsi_divergence,      10, 0.10, 20),
    ('W04_weekly_channel_vix_moderate', sig_w04_weekly_channel_vix_moderate, 8, 0.08, 20),
    ('W05_weekly_channel_spy_uptrend',  sig_w05_weekly_channel_spy_uptrend,  8, 0.08, 20),
    ('W06_spy_weekly_lag_vix_regime',   sig_w06_spy_weekly_lag_vix_regime,   8, 0.08, 20),
    ('W07_weekly_channel_rsi_oversold', sig_w07_weekly_channel_rsi_oversold, 8, 0.08, 20),
    ('W08_spy_weekly_lag_no_bear',      sig_w08_spy_weekly_lag_no_bear,      8, 0.08, 20),
    ('W09_union_lag_channel',           sig_w09_union_weekly,                8, 0.08, 20),
    ('W10_best_weekly_composite',       sig_w10_best_weekly,                 8, 0.08, 20),
]


# ── Phase 6 — Hourly bar signals ──────────────────────────────────────────────
# Primary bars are 1h OHLCV (yfinance, ~2yr history: 2023-2025).
# IS: 2023-2024 | OOS: 2025.
# "max_hold_days" = max hold in 1h bars (~33 bars = 5 trading days).
# VIX: daily close forward-filled to hourly timestamps.
# Lookbacks scaled: 1 trading day ≈ 6.5 hourly bars.

def sig_h01_spy_hourly_lag(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """H01: SPY near 5d high (33h bars), TSLA lagging ≥3%."""
    lookback = 33   # ~5 trading days
    if i < lookback:
        return 0
    spy_now      = spy['close'].iloc[i]
    spy_high     = spy['high'].iloc[i - lookback:i].max()
    tsla_now     = tsla['close'].iloc[i]
    tsla_high    = tsla['high'].iloc[i - lookback:i].max()
    spy_strong   = spy_now >= spy_high * 0.98
    tsla_lagging = tsla_now < tsla_high * 0.97
    return 1 if (spy_strong and tsla_lagging) else 0


def sig_h02_hourly_channel_bounce(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """H02: TSLA 1h channel bounce — near lower 25% of 40-bar (~6d) channel."""
    if i < win:
        return 0
    ch = _channel_at(tsla.iloc[i - win:i])
    if ch is None:
        return 0
    return 1 if _near_lower(tsla['close'].iloc[i], ch, 0.25) else 0


def sig_h03_hourly_rsi_divergence(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """H03: TSLA 1h RSI < 35 + SPY 1h RSI > 50 (intraday RSI divergence)."""
    if i < 20:
        return 0
    t_rsi = rt.iloc[i]
    s_rsi = rs.iloc[i]
    if pd.isna(t_rsi) or pd.isna(s_rsi):
        return 0
    return 1 if (t_rsi < 35 and s_rsi > 50) else 0


def sig_h04_hourly_lag_vix(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """H04: H01 + daily VIX 15–35 (forward-filled to hourly)."""
    if sig_h01_spy_hourly_lag(i, tsla, spy, vix, tw, sw, rt, rs, win) == 0:
        return 0
    vix_now = vix['close'].iloc[i]
    if pd.isna(vix_now) or not (15 <= vix_now <= 35):
        return 0
    return 1


def sig_h05_hourly_channel_spy_ma(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """H05: H02 + SPY above 20d MA (~130 hourly bars) — bounce in uptrend."""
    if sig_h02_hourly_channel_bounce(i, tsla, spy, vix, tw, sw, rt, rs, win) == 0:
        return 0
    lookback_ma = 130   # ~20 trading days
    if i < lookback_ma:
        return 0
    spy_ma = spy['close'].iloc[i - lookback_ma:i].mean()
    if spy['close'].iloc[i] < spy_ma:
        return 0
    return 1


def sig_h06_hourly_vix_spike_bounce(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """H06: Daily VIX > 25 + TSLA 1h channel bounce (fear + channel support)."""
    vix_now = vix['close'].iloc[i]
    if pd.isna(vix_now) or vix_now < 25:
        return 0
    return sig_h02_hourly_channel_bounce(i, tsla, spy, vix, tw, sw, rt, rs, win)


def sig_h07_union_h01_h02(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """H07: Union of H01 (5d lag) + H02 (channel bounce)."""
    if sig_h01_spy_hourly_lag(i, tsla, spy, vix, tw, sw, rt, rs, win) == 1:
        return 1
    if sig_h02_hourly_channel_bounce(i, tsla, spy, vix, tw, sw, rt, rs, win) == 1:
        return 1
    return 0


def sig_h08_union_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """H08: H07 union + daily VIX 15–35 regime filter."""
    if sig_h07_union_h01_h02(i, tsla, spy, vix, tw, sw, rt, rs, win) == 0:
        return 0
    vix_now = vix['close'].iloc[i]
    if pd.isna(vix_now) or not (15 <= vix_now <= 35):
        return 0
    return 1


def sig_h09_spy_momentum_tsla_lag(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """H09: SPY makes new 3d high (20h bars) AND closes up — TSLA still lagging."""
    lookback = 20   # ~3 trading days
    if i < lookback + 1:
        return 0
    spy_now  = spy['close'].iloc[i]
    spy_prev = spy['close'].iloc[i - 1]
    spy_high = spy['high'].iloc[i - lookback:i].max()
    if spy_now < spy_high * 0.99:   # SPY at/near 3d high
        return 0
    if spy_now <= spy_prev:         # SPY must be up bar
        return 0
    tsla_now  = tsla['close'].iloc[i]
    tsla_high = tsla['high'].iloc[i - lookback:i].max()
    if tsla_now >= tsla_high * 0.97:  # TSLA lagging ≥3%
        return 0
    return 1


def sig_h10_best_hourly(i, tsla, spy, vix, tw, sw, rt, rs, win):
    """H10: H08 + TSLA 1h RSI not overbought (< 65)."""
    if sig_h08_union_vix_regime(i, tsla, spy, vix, tw, sw, rt, rs, win) == 0:
        return 0
    t_rsi = rt.iloc[i]
    if not pd.isna(t_rsi) and t_rsi > 65:
        return 0
    return 1


# Phase 6: (name, fn, max_hold_bars, stop_pct, channel_window_bars)
# max_hold_bars here = hourly bars; 33 bars ≈ 5 trading days
SIGNALS_P6: List[Tuple] = [
    ('H01_hourly_spy_lag',            sig_h01_spy_hourly_lag,          33, 0.04, 40),
    ('H02_hourly_channel_bounce',     sig_h02_hourly_channel_bounce,   33, 0.04, 40),
    ('H03_hourly_rsi_divergence',     sig_h03_hourly_rsi_divergence,   40, 0.04, 40),
    ('H04_hourly_lag_vix',            sig_h04_hourly_lag_vix,          33, 0.04, 40),
    ('H05_hourly_channel_spy_ma',     sig_h05_hourly_channel_spy_ma,   33, 0.04, 40),
    ('H06_hourly_vix_spike_bounce',   sig_h06_hourly_vix_spike_bounce, 33, 0.04, 40),
    ('H07_union_lag_channel_1h',      sig_h07_union_h01_h02,           33, 0.04, 40),
    ('H08_union_vix_regime_1h',       sig_h08_union_vix_regime,        33, 0.04, 40),
    ('H09_spy_momentum_tsla_lag_1h',  sig_h09_spy_momentum_tsla_lag,   33, 0.04, 40),
    ('H10_best_hourly_composite',     sig_h10_best_hourly,             33, 0.04, 40),
]


# ── Reporting ─────────────────────────────────────────────────────────────────
def _fmt_result_line(r: BacktestResult) -> str:
    if not r.trades:
        return f"  {r.signal:<42s}| NO TRADES"
    exits = r.exit_breakdown()
    return (
        f"  {r.signal:<42s}| "
        f"n={r.n_trades:3d} | WR={r.win_rate:3.0%} | "
        f"PF={r.profit_factor:4.2f} | "
        f"P&L=${r.total_pnl:>11,.0f} | "
        f"avg=${r.avg_pnl:>7,.0f} | "
        f"hold={r.avg_hold_days:4.1f}d | "
        f"yrs={r.profitable_years()}/{len(r.by_year())} | "
        f"exits: sig={exits.get('signal',0)} stop={exits.get('stop',0)} "
        f"t/o={exits.get('timeout',0)}"
    )


def print_results_table(results: List[BacktestResult]) -> None:
    header = (
        f"  {'Signal':<42s}| "
        f"{'n':>4s} | {'WR':>4s} | "
        f"{'PF':>5s} | "
        f"{'Total P&L':>12s} | "
        f"{'Avg/trade':>9s} | "
        f"{'Hold':>6s} | "
        f"{'Yrs'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in sorted(results, key=lambda x: x.total_pnl, reverse=True):
        print(_fmt_result_line(r))


def print_year_breakdown(r: BacktestResult) -> None:
    print(f"\n  Year breakdown — {r.signal}:")
    by_year = r.by_year()
    for y, pnl in by_year.items():
        bar = "+" * int(max(pnl, 0) / 10000) + "-" * int(max(-pnl, 0) / 10000)
        bar = bar[:40]
        print(f"    {y}: ${pnl:>10,.0f}  {bar}")
    print(f"    Total:  ${r.total_pnl:>10,.0f}  ({r.profitable_years()}/{len(by_year)} years profitable)")


def print_trade_samples(r: BacktestResult, n: int = 10) -> None:
    if not r.trades:
        return
    print(f"\n  Top {n} trades by P&L — {r.signal}:")
    for t in sorted(r.trades, key=lambda x: x.pnl_usd, reverse=True)[:n]:
        print(f"    {t.entry_date.date()} → {t.exit_date.date()} "
              f"({t.hold_days}d, {t.exit_reason}) "
              f"${t.entry_price:.2f} → ${t.exit_price:.2f}  "
              f"P&L ${t.pnl_usd:>8,.0f} ({t.pnl_pct:+.2%})")
    print(f"\n  Worst {n} trades by P&L:")
    for t in sorted(r.trades, key=lambda x: x.pnl_usd)[:n]:
        print(f"    {t.entry_date.date()} → {t.exit_date.date()} "
              f"({t.hold_days}d, {t.exit_reason}) "
              f"${t.entry_price:.2f} → ${t.exit_price:.2f}  "
              f"P&L ${t.pnl_usd:>8,.0f} ({t.pnl_pct:+.2%})")


def run_param_sweep(tsla_d, spy_d, vix_d, tw, sw, signal_name: str) -> None:
    """Sweep hold_days and stop_pct for a given signal."""
    fn_map = {s[0]: s[1] for s in SIGNALS}
    fn     = fn_map.get(signal_name)
    if fn is None:
        print(f"Unknown signal: {signal_name}")
        return

    print(f"\n{'='*80}")
    print(f"Parameter sweep: {signal_name}")
    print(f"{'='*80}")
    print(f"  {'hold_days':>10s} {'stop_pct':>9s} | n   | WR   |  PF   |    P&L      | yrs")
    print("  " + "-" * 65)

    for hold in [5, 8, 10, 15, 20, 30]:
        for stop in [0.03, 0.05, 0.07, 0.10]:
            r = run_swing_backtest(
                tsla_d, spy_d, vix_d, tw, sw,
                signal_fn=fn, signal_name=signal_name,
                max_hold_days=hold, stop_pct=stop,
            )
            if r.n_trades == 0:
                continue
            print(f"  hold={hold:2d}d  stop={stop:.0%}  | "
                  f"{r.n_trades:3d} | {r.win_rate:3.0%} | "
                  f"{r.profit_factor:4.2f} | "
                  f"${r.total_pnl:>10,.0f} | "
                  f"{r.profitable_years()}/{len(r.by_year())}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='c11 Swing Backtest Explorer')
    parser.add_argument('--start-year', type=int, default=None,
                        help='First year to trade (default: 2015 daily/weekly, 2023 hourly)')
    parser.add_argument('--end-year',   type=int, default=2024,
                        help='Last year to trade (default: 2024)')
    parser.add_argument('--signal',     default='all',
                        help='Signal ID/name to run, or "all" (default: all)')
    parser.add_argument('--sweep',      default=None,
                        help='Run param sweep on this signal name (daily mode only)')
    parser.add_argument('--detail',     default=None,
                        help='Show year breakdown + trade samples for this signal')
    parser.add_argument('--mode',       default='daily',
                        choices=['daily', 'weekly', 'hourly'],
                        help='Bar frequency: daily (S01-S40, 10yr), '
                             'weekly (W01-W10, 10yr), hourly (H01-H10, ~2yr). '
                             'Default: daily')
    parser.add_argument('--no-stop',    action='store_true',
                        help='Disable stop losses (stop_pct=0.99) — test timeout-only exits')
    parser.add_argument('--trade-usd',  type=int, default=None,
                        help='Position size in USD (default: 1000000). '
                             'Use 10000 to match c10/c9 sizing ($10K/trade, $100K equity).')
    parser.add_argument('--compound',   action='store_true',
                        help='Compound equity: position size scales with equity growth, '
                             'same model as c10/c9 surfer. Pair with --trade-usd 10000.')
    parser.add_argument('--ticker',     default='TSLA',
                        help='Ticker to trade (default: TSLA). E.g. NVDA, AAPL.')
    parser.add_argument('--trail-pct',  type=float, default=0.0,
                        help='Trailing stop %% from highest close (0=disabled, use fixed stop). '
                             'E.g. 0.03 = 3%% trailing.')
    parser.add_argument('--persist',    type=int, default=1,
                        help='Bars signal must fire consecutively before entry (default=1=immediate). '
                             'E.g. 2 = require 2 consecutive signal days.')
    args = parser.parse_args()

    # Apply trade size override globally before any backtest runs
    if args.trade_usd is not None:
        import v15.validation.swing_backtest as _self
        _self.MAX_TRADE_USD = args.trade_usd
        globals()['MAX_TRADE_USD'] = args.trade_usd

    # Resolve default start year by mode
    if args.start_year is None:
        args.start_year = 2023 if args.mode == 'hourly' else 2015

    print(f"\n{'='*80}")
    print(f"c11 Swing Backtest — {args.mode} bars | {args.start_year}–{args.end_year}")
    print(f"MAX_TRADE_USD=${MAX_TRADE_USD:,.0f}  COST={COST_PER_SIDE*2*100:.2f}% round-trip")
    print(f"{'='*80}")

    t0 = time.time()

    # ── Hourly mode ────────────────────────────────────────────────────────────
    if args.mode == 'hourly':
        from datetime import date, timedelta
        print("Loading data (yfinance 1h, ~2yr history) ...")
        # yfinance 1h limit: 729 days from today
        hourly_end   = date.today().strftime('%Y-%m-%d')
        hourly_start = (date.today() - timedelta(days=728)).strftime('%Y-%m-%d')
        tsla_h = _strip_tz_intraday(
            fetch_native_tf('TSLA', '1h', hourly_start, hourly_end))
        spy_h  = _strip_tz_intraday(
            fetch_native_tf('SPY',  '1h', hourly_start, hourly_end))
        # Align on common 1h timestamps first
        common_h = tsla_h.index.intersection(spy_h.index)
        tsla_h = tsla_h.loc[common_h]
        spy_h  = spy_h.loc[common_h]
        # Load daily VIX and forward-fill to hourly grid
        vix_d_raw = _normalize_tz(
            fetch_native_tf('^VIX', 'daily', hourly_start, hourly_end))
        vix_h = _align_daily_to_hourly(vix_d_raw, common_h)
        print(f"TSLA 1h: {len(tsla_h)} bars | SPY 1h: {len(spy_h)} bars | "
              f"VIX (daily→1h): {len(vix_h)} bars")
        print(f"Data loaded in {time.time()-t0:.1f}s\n")

        sig_list = SIGNALS_P6
        results  = []
        for (name, fn, max_hold, stop, window) in sig_list:
            if args.signal != 'all' and args.signal not in name:
                continue
            t1 = time.time()
            r  = run_swing_backtest(
                tsla=tsla_h, spy=spy_h, vix=vix_h,
                tsla_weekly=None, spy_weekly=None,
                signal_fn=fn, signal_name=name,
                max_hold_days=max_hold, stop_pct=0.99 if args.no_stop else stop,
                channel_window=window,
                warmup_bars=200,          # ~1 month of hourly bars
                start_year=args.start_year,
                end_year=args.end_year,
                compound=args.compound,
                trail_pct=args.trail_pct,
                persist_bars=args.persist,
            )
            elapsed = time.time() - t1
            results.append(r)
            print(f"  {name:<46s} done in {elapsed:.1f}s → {r.n_trades} trades")

    # ── Weekly mode ────────────────────────────────────────────────────────────
    elif args.mode == 'weekly':
        print("Loading data (yfinance daily → resample weekly) ...")
        fetch_start = f'{args.start_year - 1}-01-01'
        fetch_end   = f'{args.end_year}-12-31'
        tsla_d = _normalize_tz(fetch_native_tf('TSLA', 'daily', fetch_start, fetch_end))
        spy_d  = _normalize_tz(fetch_native_tf('SPY',  'daily', fetch_start, fetch_end))
        vix_d  = _normalize_tz(fetch_native_tf('^VIX', 'daily', fetch_start, fetch_end))
        tsla_w = _resample_weekly(tsla_d)
        spy_w  = _resample_weekly(spy_d)
        vix_w  = _resample_weekly(vix_d)
        print(f"TSLA weekly: {len(tsla_w)} bars | SPY weekly: {len(spy_w)} bars | "
              f"VIX weekly: {len(vix_w)} bars")
        print(f"Data loaded in {time.time()-t0:.1f}s\n")

        sig_list = SIGNALS_P5
        results  = []
        for (name, fn, max_hold, stop, window) in sig_list:
            if args.signal != 'all' and args.signal not in name:
                continue
            t1 = time.time()
            r  = run_swing_backtest(
                tsla=tsla_w, spy=spy_w, vix=vix_w,
                tsla_weekly=None, spy_weekly=None,
                signal_fn=fn, signal_name=name,
                max_hold_days=max_hold, stop_pct=0.99 if args.no_stop else stop,
                channel_window=window,
                warmup_bars=60,           # ~1yr of weekly bars
                start_year=args.start_year,
                end_year=args.end_year,
                compound=args.compound,
                trail_pct=args.trail_pct,
                persist_bars=args.persist,
            )
            elapsed = time.time() - t1
            results.append(r)
            print(f"  {name:<46s} done in {elapsed:.1f}s → {r.n_trades} trades")

    # ── Daily mode (default) ────────────────────────────────────────────────────
    else:
        print("Loading data (yfinance daily) ...")
        fetch_start = f'{args.start_year - 1}-01-01'
        fetch_end   = f'{args.end_year}-12-31'
        ticker = args.ticker.upper()
        tsla_d = _normalize_tz(fetch_native_tf(ticker,  'daily', fetch_start, fetch_end))
        spy_d  = _normalize_tz(fetch_native_tf('SPY',   'daily', fetch_start, fetch_end))
        vix_d  = _normalize_tz(fetch_native_tf('^VIX',  'daily', fetch_start, fetch_end))
        tsla_w = _resample_weekly(tsla_d)
        spy_w  = _resample_weekly(spy_d)
        print(f"{ticker} daily: {len(tsla_d)} bars | SPY daily: {len(spy_d)} bars | "
              f"VIX daily: {len(vix_d)} bars")
        print(f"{ticker} weekly: {len(tsla_w)} bars | SPY weekly: {len(spy_w)} bars")
        print(f"Data loaded in {time.time()-t0:.1f}s\n")

        if args.sweep:
            run_param_sweep(tsla_d, spy_d, vix_d, tsla_w, spy_w, args.sweep)
            return

        sig_list = SIGNALS
        results  = []
        for (name, fn, max_hold, stop, window) in sig_list:
            if args.signal != 'all' and args.signal not in name:
                continue
            t1 = time.time()
            r  = run_swing_backtest(
                tsla=tsla_d, spy=spy_d, vix=vix_d,
                tsla_weekly=tsla_w, spy_weekly=spy_w,
                signal_fn=fn, signal_name=name,
                max_hold_days=max_hold, stop_pct=0.99 if args.no_stop else stop,
                channel_window=window,
                start_year=args.start_year,
                end_year=args.end_year,
                compound=args.compound,
                trail_pct=args.trail_pct,
                persist_bars=args.persist,
            )
            elapsed = time.time() - t1
            results.append(r)
            print(f"  {name:<46s} done in {elapsed:.1f}s → {r.n_trades} trades")

    # ── Common reporting ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"RESULTS  (sorted by total P&L)  [{args.mode} mode]:")
    print(f"{'='*80}")
    print_results_table(results)

    if args.detail:
        for r in results:
            if args.detail in r.signal:
                print_year_breakdown(r)
                print_trade_samples(r, n=10)

    if results:
        best = max(results, key=lambda r: r.total_pnl)
        if best.n_trades > 0:
            print_year_breakdown(best)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
