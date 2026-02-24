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


SIGNALS = SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4 + SIGNALS_P5D + SIGNALS_P6D + SIGNALS_P7D + SIGNALS_P7F + SIGNALS_P7H + SIGNALS_P7J + SIGNALS_P7K + SIGNALS_P7L + SIGNALS_P7M + SIGNALS_P7N + SIGNALS_P7P + SIGNALS_P7Q + SIGNALS_P7R + SIGNALS_P7S + SIGNALS_P7T + SIGNALS_P7U + SIGNALS_P7V + SIGNALS_P7W + SIGNALS_P7X + SIGNALS_P7Y


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
