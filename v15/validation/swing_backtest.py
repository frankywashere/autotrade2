#!/usr/bin/env python3
"""
c11 Swing Backtest — multi-day/week TSLA trading via channel + SPY/RSI patterns.

All signals use daily bars (11 years of history from yfinance).
Entry: next bar open after signal fires. No look-ahead bias.
Sizing: fixed $1M per trade (consistent with intraday system).
Costs: 0.05% slippage + 0.01% commission per side.

Signals:
  S01  TSLA daily channel bounce (baseline)
  S02  TSLA daily bounce + SPY uptrend filter
  S03  RSI divergence (TSLA oversold, SPY healthy)
  S04  SPY-TSLA lag (SPY breaks high, TSLA hasn't caught up)
  S05  High-quality channel bounce (r² > 0.85)
  S06  Multi-window bounce (near lower band on 30d + 50d + 70d)
  S07  Combined: channel bottom + RSI div + SPY above MA  [user idea]
  S08  VIX spike + channel bounce (buy fear)
  S09  TSLA weekly channel bounce
  S10  SPY channel break up → TSLA entry

Usage:
    python3 -m v15.validation.swing_backtest
    python3 -m v15.validation.swing_backtest --start-year 2025 --end-year 2025
    python3 -m v15.validation.swing_backtest --sweep S01   # param sweep on one signal
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
    """Strip timezone info from index (convert to UTC first, then tz-naive date)."""
    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC').normalize().tz_localize(None)
    return df


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
                  reason: str, hold_days: int) -> None:
    eff_entry = entry_price * (1 + COST_PER_SIDE * direction)
    eff_exit  = exit_price  * (1 - COST_PER_SIDE * direction)
    pnl_pct   = direction * (eff_exit - eff_entry) / eff_entry
    pnl_usd   = pnl_pct * MAX_TRADE_USD
    result.trades.append(Trade(
        signal=signal, direction=direction,
        entry_date=entry_date, entry_price=entry_price,
        exit_date=exit_date, exit_price=exit_price,
        exit_reason=reason, hold_days=hold_days,
        pnl_usd=pnl_usd, pnl_pct=pnl_pct,
    ))


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
) -> BacktestResult:
    result = BacktestResult(signal=signal_name)

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

    in_trade    = False
    entry_price = 0.0
    entry_bar   = 0
    entry_date  = None
    direction   = 0

    for i in range(trade_start, n - 1):
        bar_year = tsla_full.index[i].year
        if bar_year < start_year or bar_year > end_year:
            continue

        if in_trade:
            price     = tsla_full['close'].iloc[i]
            hold_days = i - entry_bar

            # Stop loss (check vs close, exit at next open)
            if direction == 1 and price < entry_price * (1 - stop_pct):
                exit_price = tsla_full['open'].iloc[i + 1]
                _record_trade(result, signal_name, direction,
                              entry_date, entry_price,
                              tsla_full.index[i + 1], exit_price,
                              'stop', hold_days)
                in_trade = False
                continue

            # Timeout
            if hold_days >= max_hold_days:
                exit_price = tsla_full['open'].iloc[i + 1]
                _record_trade(result, signal_name, direction,
                              entry_date, entry_price,
                              tsla_full.index[i + 1], exit_price,
                              'timeout', hold_days)
                in_trade = False
                continue

            # Exit signal (must hold at least 2 days to avoid whipsaw)
            if hold_days >= 2:
                sig = signal_fn(i, tsla_full, spy_full, vix_full,
                                tsla_weekly, spy_weekly,
                                rsi_tsla, rsi_spy, channel_window)
                if sig != direction:
                    exit_price = tsla_full['open'].iloc[i + 1]
                    _record_trade(result, signal_name, direction,
                                  entry_date, entry_price,
                                  tsla_full.index[i + 1], exit_price,
                                  'signal', hold_days)
                    in_trade = False
                    continue

        else:
            sig = signal_fn(i, tsla_full, spy_full, vix_full,
                            tsla_weekly, spy_weekly,
                            rsi_tsla, rsi_spy, channel_window)
            if sig != 0:
                entry_price = tsla_full['open'].iloc[i + 1]
                entry_date  = tsla_full.index[i + 1]
                entry_bar   = i + 1
                direction   = sig
                in_trade    = True

    # Close any still-open trade at end of data
    if in_trade:
        exit_price = tsla_full['close'].iloc[-1]
        _record_trade(result, signal_name, direction,
                      entry_date, entry_price,
                      tsla_full.index[-1], exit_price,
                      'timeout', n - 1 - entry_bar)

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

SIGNALS = SIGNALS_P1 + SIGNALS_P2 + SIGNALS_P3 + SIGNALS_P4


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
    parser.add_argument('--start-year', type=int, default=2015,
                        help='First year to trade (default: 2015)')
    parser.add_argument('--end-year',   type=int, default=2024,
                        help='Last year to trade (default: 2024)')
    parser.add_argument('--signal',     default='all',
                        help='Signal ID/name to run, or "all" (default: all)')
    parser.add_argument('--sweep',      default=None,
                        help='Run param sweep on this signal name')
    parser.add_argument('--detail',     default=None,
                        help='Show year breakdown + trade samples for this signal')
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"c11 Swing Backtest — daily bars | {args.start_year}–{args.end_year}")
    print(f"MAX_TRADE_USD=${MAX_TRADE_USD:,.0f}  COST={COST_PER_SIDE*2*100:.2f}% round-trip")
    print(f"{'='*80}")

    t0 = time.time()
    print("Loading data (yfinance daily) ...")
    fetch_start = f'{args.start_year - 1}-01-01'   # 1 extra year for warmup
    fetch_end   = f'{args.end_year}-12-31'

    tsla_d = _normalize_tz(fetch_native_tf('TSLA',  'daily', fetch_start, fetch_end))
    spy_d  = _normalize_tz(fetch_native_tf('SPY',   'daily', fetch_start, fetch_end))
    vix_d  = _normalize_tz(fetch_native_tf('^VIX',  'daily', fetch_start, fetch_end))

    # Build weekly bars by resampling daily
    tsla_w = _resample_weekly(tsla_d)
    spy_w  = _resample_weekly(spy_d)

    print(f"TSLA daily: {len(tsla_d)} bars | SPY daily: {len(spy_d)} bars | "
          f"VIX daily: {len(vix_d)} bars")
    print(f"TSLA weekly: {len(tsla_w)} bars | SPY weekly: {len(spy_w)} bars")
    print(f"Data loaded in {time.time()-t0:.1f}s\n")

    # Param sweep mode
    if args.sweep:
        run_param_sweep(tsla_d, spy_d, vix_d, tsla_w, spy_w, args.sweep)
        return

    # Run all (or selected) signals
    results = []
    for (name, fn, max_hold, stop, window) in SIGNALS:
        if args.signal != 'all' and args.signal not in name:
            continue
        t1 = time.time()
        r  = run_swing_backtest(
            tsla=tsla_d, spy=spy_d, vix=vix_d,
            tsla_weekly=tsla_w, spy_weekly=spy_w,
            signal_fn=fn, signal_name=name,
            max_hold_days=max_hold, stop_pct=stop,
            channel_window=window,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        elapsed = time.time() - t1
        results.append(r)
        print(f"  {name:<42s} done in {elapsed:.1f}s → {r.n_trades} trades")

    print(f"\n{'='*80}")
    print(f"RESULTS  (sorted by total P&L):")
    print(f"{'='*80}")
    print_results_table(results)

    # Detail mode
    if args.detail:
        for r in results:
            if args.detail in r.signal:
                print_year_breakdown(r)
                print_trade_samples(r, n=10)

    # Auto-detail top signal
    if results:
        best = max(results, key=lambda r: r.total_pnl)
        if best.n_trades > 0:
            print_year_breakdown(best)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
