#!/usr/bin/env python3
"""
Channel Surfer Backtester

Tests the Channel Surfer signal engine against historical 5-min TSLA data.
Walks forward bar-by-bar, evaluates signals, simulates trades, and reports
win rate, profit factor, and other key metrics.

Usage:
    python3 -m v15.core.surfer_backtest [--days 30]
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Completed trade record."""
    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    direction: str       # 'BUY' or 'SELL'
    confidence: float
    stop_pct: float
    tp_pct: float
    exit_reason: str     # 'stop', 'tp', 'timeout', 'signal_flip'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    hold_bars: int = 0
    primary_tf: str = ''
    signal_type: str = 'bounce'  # 'bounce' or 'break'
    trade_size: float = 10000.0
    mae_pct: float = 0.0  # Maximum Adverse Excursion (worst unrealized loss %)
    mfe_pct: float = 0.0  # Maximum Favorable Excursion (best unrealized gain %)


@dataclass
class OpenPosition:
    """Currently open position."""
    entry_bar: int
    entry_price: float
    direction: str
    confidence: float
    stop_price: float
    tp_price: float
    primary_tf: str
    signal_type: str = 'bounce'     # 'bounce' or 'break'
    trade_size: float = 10000.0     # Confidence-scaled position size
    ou_half_life: float = 5.0
    max_hold_bars: int = 60  # 5 hours max
    trailing_stop: float = 0.0  # Best price seen for trailing
    worst_price: float = 0.0    # Worst price seen (for MAE)
    best_price: float = 0.0     # Best price seen (for MFE)


@dataclass
class BacktestMetrics:
    """Summary metrics."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_hold_bars: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / max(self.total_trades, 1)

    @property
    def profit_factor(self) -> float:
        return self.gross_profit / max(abs(self.gross_loss), 1e-6)

    @property
    def expectancy(self) -> float:
        """Expected $ per trade."""
        return self.total_pnl / max(self.total_trades, 1)

    def summary(self) -> str:
        return (
            f"Trades: {self.total_trades} | Win Rate: {self.win_rate:.0%} | "
            f"PF: {self.profit_factor:.2f} | Total P&L: ${self.total_pnl:,.2f} | "
            f"Avg Hold: {self.avg_hold_bars:.0f} bars | "
            f"Avg Win: {self.avg_win_pct:.2%} | Avg Loss: {self.avg_loss_pct:.2%} | "
            f"Max DD: {self.max_drawdown_pct:.1%} | "
            f"Expectancy: ${self.expectancy:,.2f}/trade"
        )


def run_backtest(
    days: int = 30,
    eval_interval: int = 3,     # Check every 3 bars = 15 min
    max_hold_bars: int = 60,    # Max 5 hours (60 * 5min)
    position_size: float = 10000.0,  # $10k per trade
    min_confidence: float = 0.45,
    use_multi_tf: bool = True,  # Use higher TF data for context
) -> tuple:
    """
    Run Channel Surfer backtest on historical 5-min TSLA data.

    Returns:
        (metrics, trades) tuple
    """
    import yfinance as yf
    from v15.core.channel import detect_channels_multi_window, select_best_channel
    from v15.core.channel_surfer import analyze_channels, SIGNAL_TFS, TF_WINDOWS

    # Fetch data
    print(f"Fetching {days}d of 5-min TSLA data...")
    tsla = yf.download('TSLA', period=f'{days}d', interval='5m', progress=False)
    if isinstance(tsla.columns, pd.MultiIndex):
        tsla.columns = tsla.columns.get_level_values(0)
    tsla.columns = [c.lower() for c in tsla.columns]
    print(f"Got {len(tsla)} bars")

    # Fetch higher TF data for context
    higher_tf_data = {}
    if use_multi_tf:
        for tf_label, yf_interval, yf_period in [
            ('1h', '1h', '2y'),
            ('daily', '1d', '5y'),
        ]:
            print(f"  Fetching {tf_label} data...")
            df = yf.download('TSLA', period=yf_period, interval=yf_interval, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            higher_tf_data[tf_label] = df
            print(f"  {tf_label}: {len(df)} bars")

    if len(tsla) < 200:
        print("Not enough data for backtest")
        return BacktestMetrics(), []

    closes = tsla['close'].values
    highs = tsla['high'].values
    lows = tsla['low'].values

    # Compute ATR(14) for volatility-adjusted stops
    atr_period = 14
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    tr = np.concatenate([[highs[0] - lows[0]], tr])  # First bar uses H-L
    atr = np.full_like(closes, np.nan)
    atr[atr_period - 1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i - 1] * (atr_period - 1) + tr[i]) / atr_period
    # Fill initial NaN with first valid ATR
    first_valid = atr[atr_period - 1]
    atr[:atr_period - 1] = first_valid

    trades: List[Trade] = []
    equity_curve: List[Tuple[int, float]] = []  # (bar_idx, equity)
    position: Optional[OpenPosition] = None
    equity = position_size * 10  # Start with 100k
    peak_equity = equity
    max_dd = 0.0
    consecutive_losses = 0  # Track losing streak for position reduction




    # Walk forward from bar 100 (need lookback)
    start_bar = 100
    total_bars = len(tsla)
    last_print = 0

    print(f"\nBacktesting from bar {start_bar} to {total_bars} (interval={eval_interval})...")
    t_start = time.time()

    for bar in range(start_bar, total_bars, eval_interval):
        # Progress
        if bar - last_print >= 500:
            pct = (bar - start_bar) / (total_bars - start_bar) * 100
            print(f"  [{pct:.0f}%] bar={bar}/{total_bars}, trades={len(trades)}, equity=${equity:,.0f}")
            last_print = bar

        current_price = float(closes[bar])

        # --- Check exits for open position ---
        if position is not None:
            bars_held = bar - position.entry_bar
            should_exit = False
            exit_reason = ''
            exit_price = current_price

            # Check high/low for stop/TP hit within the evaluation window
            window_highs = highs[max(0, bar - eval_interval):bar + 1]
            window_lows = lows[max(0, bar - eval_interval):bar + 1]
            window_high = float(np.max(window_highs))
            window_low = float(np.min(window_lows))

            # Track MAE/MFE
            if position.direction == 'BUY':
                if position.worst_price == 0 or window_low < position.worst_price:
                    position.worst_price = window_low
                if window_high > position.best_price:
                    position.best_price = window_high
            else:
                if position.worst_price == 0 or window_high > position.worst_price:
                    position.worst_price = window_high
                if position.best_price == 0 or window_low < position.best_price:
                    position.best_price = window_low

            # Trailing stop logic (different for bounces vs breakouts)
            entry = position.entry_price
            initial_stop_dist = abs(position.stop_price - entry) / entry
            tp_dist = abs(position.tp_price - entry) / entry
            is_breakout = position.signal_type == 'break'

            if position.direction == 'BUY':
                if window_high > position.trailing_stop:
                    position.trailing_stop = window_high

                if is_breakout:
                    # Breakouts: simple progressive trail (let them run)
                    profit_from_best = (position.trailing_stop - entry) / entry
                    if profit_from_best > 0.008:
                        trail_from_best = position.trailing_stop * (1 - initial_stop_dist * 0.3)
                        effective_stop = max(position.stop_price, trail_from_best)
                    else:
                        effective_stop = position.stop_price
                else:
                    # Bounces: multi-stage profit locking
                    profit_from_entry = (position.trailing_stop - entry) / entry
                    profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
                    if profit_ratio >= 0.80:
                        trail_from_best = position.trailing_stop * (1 - initial_stop_dist * 0.15)
                        effective_stop = max(position.stop_price, trail_from_best)
                    elif profit_ratio >= 0.50:
                        trail_from_best = position.trailing_stop * (1 - initial_stop_dist * 0.40)
                        effective_stop = max(position.stop_price, trail_from_best)
                    elif profit_ratio >= 0.25:
                        effective_stop = max(position.stop_price, entry * 0.999)
                    else:
                        effective_stop = position.stop_price

                if window_low <= effective_stop:
                    should_exit = True
                    exit_reason = 'stop' if effective_stop == position.stop_price else 'trail'
                    exit_price = effective_stop
                elif window_high >= position.tp_price:
                    should_exit = True
                    exit_reason = 'tp'
                    exit_price = position.tp_price
                elif not is_breakout and bars_held >= max(6, int(position.ou_half_life * 3)):
                    should_exit = True
                    exit_reason = 'ou_timeout'

            else:  # SELL
                if position.trailing_stop == 0 or window_low < position.trailing_stop:
                    position.trailing_stop = window_low

                if is_breakout:
                    profit_from_best = (entry - position.trailing_stop) / entry
                    if profit_from_best > 0.008:
                        trail_from_best = position.trailing_stop * (1 + initial_stop_dist * 0.3)
                        effective_stop = min(position.stop_price, trail_from_best)
                    else:
                        effective_stop = position.stop_price
                else:
                    profit_from_entry = (entry - position.trailing_stop) / entry
                    profit_ratio = profit_from_entry / max(tp_dist, 1e-6)
                    if profit_ratio >= 0.80:
                        trail_from_best = position.trailing_stop * (1 + initial_stop_dist * 0.15)
                        effective_stop = min(position.stop_price, trail_from_best)
                    elif profit_ratio >= 0.50:
                        trail_from_best = position.trailing_stop * (1 + initial_stop_dist * 0.40)
                        effective_stop = min(position.stop_price, trail_from_best)
                    elif profit_ratio >= 0.25:
                        effective_stop = min(position.stop_price, entry * 1.001)
                    else:
                        effective_stop = position.stop_price

                if window_high >= effective_stop:
                    should_exit = True
                    exit_reason = 'stop' if effective_stop == position.stop_price else 'trail'
                    exit_price = effective_stop
                elif window_low <= position.tp_price:
                    should_exit = True
                    exit_reason = 'tp'
                    exit_price = position.tp_price
                elif not is_breakout and bars_held >= max(6, int(position.ou_half_life * 3)):
                    should_exit = True
                    exit_reason = 'ou_timeout'

            if not should_exit and bars_held >= position.max_hold_bars:
                should_exit = True
                exit_reason = 'timeout'

            if should_exit:
                # Compute P&L
                if position.direction == 'BUY':
                    pnl_pct = (exit_price - position.entry_price) / position.entry_price
                else:
                    pnl_pct = (position.entry_price - exit_price) / position.entry_price

                pnl = pnl_pct * position.trade_size

                # Compute MAE/MFE
                if position.direction == 'BUY':
                    mae = (position.entry_price - position.worst_price) / position.entry_price if position.worst_price > 0 else 0
                    mfe = (position.best_price - position.entry_price) / position.entry_price if position.best_price > 0 else 0
                else:
                    mae = (position.worst_price - position.entry_price) / position.entry_price if position.worst_price > 0 else 0
                    mfe = (position.entry_price - position.best_price) / position.entry_price if position.best_price > 0 else 0

                trade = Trade(
                    entry_bar=position.entry_bar,
                    exit_bar=bar,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    direction=position.direction,
                    confidence=position.confidence,
                    stop_pct=(abs(position.stop_price - position.entry_price) / position.entry_price),
                    tp_pct=(abs(position.tp_price - position.entry_price) / position.entry_price),
                    exit_reason=exit_reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    hold_bars=bars_held,
                    primary_tf=position.primary_tf,
                    signal_type=position.signal_type,
                    trade_size=position.trade_size,
                    mae_pct=round(mae, 6),
                    mfe_pct=round(mfe, 6),
                )
                trades.append(trade)
                equity += pnl
                equity_curve.append((bar, equity))
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity
                max_dd = max(max_dd, dd)

                # Track consecutive losses for position sizing
                if pnl <= 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0

                position = None

        # --- Generate new signal (only if flat) ---
        if position is None:
            # Get lookback data for channel detection
            lookback = min(bar + 1, 100)
            df_slice = tsla.iloc[bar - lookback + 1:bar + 1]

            if len(df_slice) < 20:
                continue

            # Detect channels at multiple windows
            try:
                multi = detect_channels_multi_window(df_slice, windows=[10, 15, 20, 30, 40])
                best_ch, _ = select_best_channel(multi)
            except Exception:
                continue

            if best_ch is None or not best_ch.valid:
                continue

            # Build multi-TF channels
            slice_closes = df_slice['close'].values
            channels_by_tf = {'5min': best_ch}
            prices_by_tf = {'5min': slice_closes}
            current_prices_dict = {'5min': current_price}
            volumes_dict = {}

            if 'volume' in df_slice.columns:
                volumes_dict['5min'] = df_slice['volume'].values

            # Add higher TF channels (rolling window relative to current bar time)
            if use_multi_tf:
                current_time = tsla.index[bar]
                # Normalize to tz-naive for comparison
                if current_time.tzinfo is not None:
                    current_time_naive = current_time.tz_localize(None)
                else:
                    current_time_naive = current_time
                for tf_label, tf_df in higher_tf_data.items():
                    # Only use higher-TF data available at current time (no lookahead)
                    tf_idx = tf_df.index
                    if tf_idx.tz is not None:
                        tf_available = tf_df[tf_idx <= current_time]
                    else:
                        tf_available = tf_df[tf_idx <= current_time_naive]
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
                            current_prices_dict[tf_label] = float(tf_recent['close'].iloc[-1])
                            if 'volume' in tf_recent.columns:
                                volumes_dict[tf_label] = tf_recent['volume'].values
                    except Exception:
                        pass

            try:
                analysis = analyze_channels(
                    channels_by_tf, prices_by_tf, current_prices_dict,
                    volumes_by_tf=volumes_dict if volumes_dict else None,
                )
            except Exception:
                continue

            sig = analysis.signal

            if sig.action in ('BUY', 'SELL') and sig.confidence >= min_confidence:
                # Enter position
                entry_price = current_price

                # Confidence-scaled position sizing
                if sig.confidence >= 0.70:
                    trade_size = position_size * 1.5
                elif sig.confidence >= 0.60:
                    trade_size = position_size * 1.2
                else:
                    trade_size = position_size

                # Consecutive loss protection (defensive — keep for live trading)
                # With 0.6% max DD in backtest, this rarely activates
                if consecutive_losses >= 4:
                    trade_size *= 0.50  # Half size after 4+ consecutive losses

                # Volatility-adjusted stops: blend channel width with ATR
                # Floor at 1.5*ATR (survive noise), cap at 2.5*ATR (don't overexpose)
                current_atr = atr[bar]
                atr_floor = (1.5 * current_atr) / entry_price
                atr_cap = (2.5 * current_atr) / entry_price
                adjusted_stop_pct = np.clip(sig.suggested_stop_pct, atr_floor, atr_cap)

                if sig.action == 'BUY':
                    stop = entry_price * (1 - adjusted_stop_pct)
                    tp = entry_price * (1 + sig.suggested_tp_pct)
                else:
                    stop = entry_price * (1 + adjusted_stop_pct)
                    tp = entry_price * (1 - sig.suggested_tp_pct)

                # Breakout trades get longer max hold (trends persist)
                effective_max_hold = max_hold_bars * 2 if sig.signal_type == 'break' else max_hold_bars

                # Get OU half-life from primary TF state
                primary_state = analysis.tf_states.get(sig.primary_tf)
                ou_hl = primary_state.ou_half_life if primary_state else 5.0

                position = OpenPosition(
                    entry_bar=bar,
                    entry_price=entry_price,
                    direction=sig.action,
                    confidence=sig.confidence,
                    stop_price=stop,
                    tp_price=tp,
                    primary_tf=sig.primary_tf,
                    signal_type=sig.signal_type,
                    trade_size=trade_size,
                    ou_half_life=ou_hl,
                    max_hold_bars=effective_max_hold,
                    trailing_stop=entry_price,
                )

    # Close any remaining position
    if position is not None:
        exit_price = float(closes[-1])
        if position.direction == 'BUY':
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
        pnl = pnl_pct * position.trade_size
        trades.append(Trade(
            entry_bar=position.entry_bar,
            exit_bar=total_bars - 1,
            entry_price=position.entry_price,
            exit_price=exit_price,
            direction=position.direction,
            confidence=position.confidence,
            stop_pct=0, tp_pct=0,
            exit_reason='end_of_data',
            pnl=pnl, pnl_pct=pnl_pct,
            hold_bars=total_bars - 1 - position.entry_bar,
            primary_tf=position.primary_tf,
            signal_type=position.signal_type,
            trade_size=position.trade_size,
        ))
        equity += pnl

    elapsed = time.time() - t_start

    # Compute metrics
    metrics = BacktestMetrics()
    if trades:
        metrics.total_trades = len(trades)
        metrics.wins = sum(1 for t in trades if t.pnl > 0)
        metrics.losses = sum(1 for t in trades if t.pnl <= 0)
        metrics.total_pnl = sum(t.pnl for t in trades)
        metrics.gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        metrics.gross_loss = sum(t.pnl for t in trades if t.pnl < 0)
        metrics.max_drawdown_pct = max_dd
        metrics.avg_hold_bars = np.mean([t.hold_bars for t in trades])

        win_pcts = [t.pnl_pct for t in trades if t.pnl > 0]
        loss_pcts = [t.pnl_pct for t in trades if t.pnl < 0]
        metrics.avg_win_pct = np.mean(win_pcts) if win_pcts else 0
        metrics.avg_loss_pct = np.mean(loss_pcts) if loss_pcts else 0

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"\n{'='*70}")
    print(f"CHANNEL SURFER BACKTEST RESULTS")
    print(f"{'='*70}")
    print(metrics.summary())

    # Breakdown by exit reason
    if trades:
        print(f"\nExit reason breakdown:")
        for reason in set(t.exit_reason for t in trades):
            reason_trades = [t for t in trades if t.exit_reason == reason]
            reason_wins = sum(1 for t in reason_trades if t.pnl > 0)
            reason_pnl = sum(t.pnl for t in reason_trades)
            print(f"  {reason:12s}: {len(reason_trades):3d} trades, "
                  f"WR={reason_wins/len(reason_trades):.0%}, P&L=${reason_pnl:,.2f}")

        # Direction breakdown
        for direction in ('BUY', 'SELL'):
            dir_trades = [t for t in trades if t.direction == direction]
            if dir_trades:
                dir_wins = sum(1 for t in dir_trades if t.pnl > 0)
                dir_pnl = sum(t.pnl for t in dir_trades)
                print(f"  {direction:12s}: {len(dir_trades):3d} trades, "
                      f"WR={dir_wins/len(dir_trades):.0%}, P&L=${dir_pnl:,.2f}")

        # Signal type breakdown (bounce vs break)
        for stype in ('bounce', 'break'):
            type_trades = [t for t in trades if t.signal_type == stype]
            if type_trades:
                type_wins = sum(1 for t in type_trades if t.pnl > 0)
                type_pnl = sum(t.pnl for t in type_trades)
                avg_size = np.mean([t.trade_size for t in type_trades])
                print(f"  {stype:12s}: {len(type_trades):3d} trades, "
                      f"WR={type_wins/len(type_trades):.0%}, P&L=${type_pnl:,.2f}, "
                      f"avg size=${avg_size:,.0f}")

        # MAE/MFE analysis (trade quality indicators)
        maes = [t.mae_pct for t in trades if t.mae_pct > 0]
        mfes = [t.mfe_pct for t in trades if t.mfe_pct > 0]
        if maes and mfes:
            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl <= 0]
            win_eff = [t.pnl_pct / max(t.mfe_pct, 1e-6) for t in winners if t.mfe_pct > 0]
            loss_eff = [t.mae_pct / max(t.mfe_pct, 1e-6) for t in losers if t.mfe_pct > 0]
            print(f"\nTrade quality (MAE/MFE):")
            print(f"  Avg MAE: {np.mean(maes):.3%} (worst drawdown before exit)")
            print(f"  Avg MFE: {np.mean(mfes):.3%} (best unrealized gain)")
            if win_eff:
                print(f"  Winner efficiency: {np.mean(win_eff):.0%} (% of MFE captured at exit)")
            if loss_eff:
                print(f"  Loser MAE/MFE: {np.mean(loss_eff):.1f}x (how far wrong vs best)")
            win_maes = [t.mae_pct for t in winners if t.mae_pct > 0]
            loss_maes = [t.mae_pct for t in losers if t.mae_pct > 0]
            if win_maes:
                print(f"  Winner MAE: {np.mean(win_maes):.3%}")
            if loss_maes:
                print(f"  Loser  MAE: {np.mean(loss_maes):.3%}")

        # Time-of-day and day-of-week analysis
        timestamps = tsla.index
        from collections import defaultdict
        hour_stats = defaultdict(lambda: {'pnl': 0, 'wins': 0, 'total': 0})
        day_stats = defaultdict(lambda: {'pnl': 0, 'wins': 0, 'total': 0})
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        for t in trades:
            if t.entry_bar < len(timestamps):
                ts = timestamps[t.entry_bar]
                # Convert to ET (UTC-5) for display
                hour_et = (ts.hour - 5) % 24 if hasattr(ts, 'hour') else 0
                dow = ts.dayofweek if hasattr(ts, 'dayofweek') else 0
                hour_stats[hour_et]['pnl'] += t.pnl
                hour_stats[hour_et]['total'] += 1
                if t.pnl > 0:
                    hour_stats[hour_et]['wins'] += 1
                day_stats[dow]['pnl'] += t.pnl
                day_stats[dow]['total'] += 1
                if t.pnl > 0:
                    day_stats[dow]['wins'] += 1

        if hour_stats:
            print(f"\nPerformance by hour (ET):")
            for h in sorted(hour_stats.keys()):
                s = hour_stats[h]
                wr = s['wins'] / s['total'] if s['total'] > 0 else 0
                avg_pnl = s['pnl'] / s['total'] if s['total'] > 0 else 0
                bar = '█' * max(1, int(abs(avg_pnl) / 5))
                sign = '+' if avg_pnl >= 0 else ''
                print(f"  {h:2d}:00  {s['total']:3d} trades  WR={wr:.0%}  "
                      f"avg={sign}${avg_pnl:.1f}  {'🟢' if avg_pnl > 0 else '🔴'}{bar}")

        if day_stats:
            print(f"\nPerformance by day:")
            for d in sorted(day_stats.keys()):
                s = day_stats[d]
                wr = s['wins'] / s['total'] if s['total'] > 0 else 0
                avg_pnl = s['pnl'] / s['total'] if s['total'] > 0 else 0
                sign = '+' if avg_pnl >= 0 else ''
                print(f"  {day_names[d]:3s}  {s['total']:3d} trades  WR={wr:.0%}  "
                      f"P&L=${s['pnl']:,.0f}  avg={sign}${avg_pnl:.1f}")

    return metrics, trades, equity_curve


def run_walk_forward(eval_interval: int = 3, max_hold_bars: int = 60,
                      min_confidence: float = 0.45):
    """
    Walk-forward validation: run 60-day backtest, split trades into
    in-sample (first 40 days) and out-of-sample (last 20 days).
    Compares metrics side by side.
    """
    metrics, trades, equity_curve = run_backtest(
        days=60, eval_interval=eval_interval,
        max_hold_bars=max_hold_bars, min_confidence=min_confidence,
    )

    if not trades:
        print("No trades to analyze")
        return

    # Split by entry bar — first 2/3 of bars = IS, last 1/3 = OOS
    total_bars = max(t.exit_bar for t in trades)
    split_bar = int(total_bars * 2 / 3)

    is_trades = [t for t in trades if t.entry_bar < split_bar]
    oos_trades = [t for t in trades if t.entry_bar >= split_bar]

    def summarize(label, tlist):
        if not tlist:
            print(f"\n  {label}: No trades")
            return
        wins = sum(1 for t in tlist if t.pnl > 0)
        total_pnl = sum(t.pnl for t in tlist)
        gross_win = sum(t.pnl for t in tlist if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in tlist if t.pnl <= 0))
        pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
        avg_win = np.mean([t.pnl_pct for t in tlist if t.pnl > 0]) if wins > 0 else 0
        avg_loss = np.mean([t.pnl_pct for t in tlist if t.pnl <= 0]) if len(tlist) - wins > 0 else 0
        bounce = [t for t in tlist if t.signal_type == 'bounce']
        brk = [t for t in tlist if t.signal_type == 'break']
        bounce_wr = sum(1 for t in bounce if t.pnl > 0) / len(bounce) if bounce else 0
        brk_wr = sum(1 for t in brk if t.pnl > 0) / len(brk) if brk else 0
        print(f"\n  {label}:")
        print(f"    Trades: {len(tlist)} | WR: {wins/len(tlist):.0%} | PF: {pf:.2f} | "
              f"P&L: ${total_pnl:,.0f} | Exp: ${total_pnl/len(tlist):.1f}/trade")
        print(f"    Avg Win: {avg_win:.2%} | Avg Loss: {avg_loss:.2%}")
        print(f"    Bounce: {len(bounce)} trades, {bounce_wr:.0%} WR | "
              f"Break: {len(brk)} trades, {brk_wr:.0%} WR")

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION (split at bar {split_bar}/{total_bars})")
    print(f"{'='*60}")
    summarize("IN-SAMPLE (first ~40 days)", is_trades)
    summarize("OUT-OF-SAMPLE (last ~20 days)", oos_trades)

    # Stability metrics
    if is_trades and oos_trades:
        is_wr = sum(1 for t in is_trades if t.pnl > 0) / len(is_trades)
        oos_wr = sum(1 for t in oos_trades if t.pnl > 0) / len(oos_trades)
        is_exp = sum(t.pnl for t in is_trades) / len(is_trades)
        oos_exp = sum(t.pnl for t in oos_trades) / len(oos_trades)
        wr_decay = (oos_wr - is_wr) / is_wr if is_wr > 0 else 0
        exp_decay = (oos_exp - is_exp) / is_exp if is_exp > 0 else 0
        print(f"\n  Stability:")
        print(f"    WR decay: {wr_decay:+.0%} (IS→OOS)")
        print(f"    Exp decay: {exp_decay:+.0%} (IS→OOS)")
        if abs(wr_decay) < 0.15 and abs(exp_decay) < 0.40:
            print(f"    ✅ Strategy appears STABLE out-of-sample")
        elif abs(wr_decay) < 0.25:
            print(f"    ⚠️  Moderate OOS degradation — monitor closely")
        else:
            print(f"    ❌ Significant OOS degradation — possible overfit")


def main():
    parser = argparse.ArgumentParser(description='Channel Surfer Backtest')
    parser.add_argument('--days', type=int, default=30, help='Days of 5min data')
    parser.add_argument('--eval-interval', type=int, default=6, help='Bars between evaluations')
    parser.add_argument('--max-hold', type=int, default=60, help='Max bars to hold')
    parser.add_argument('--min-conf', type=float, default=0.45, help='Minimum signal confidence')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward validation')
    args = parser.parse_args()

    if args.walk_forward:
        run_walk_forward(
            eval_interval=args.eval_interval,
            max_hold_bars=args.max_hold,
            min_confidence=args.min_conf,
        )
    else:
        run_backtest(
            days=args.days,
            eval_interval=args.eval_interval,
            max_hold_bars=args.max_hold,
            min_confidence=args.min_conf,
        )


if __name__ == '__main__':
    main()
