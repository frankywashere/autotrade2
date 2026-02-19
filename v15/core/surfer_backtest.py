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
    ou_half_life: float = 5.0
    max_hold_bars: int = 60  # 5 hours max
    trailing_stop: float = 0.0  # Best price seen for trailing


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
    eval_interval: int = 6,     # Check every 6 bars = 30 min
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

    trades: List[Trade] = []
    equity_curve: List[Tuple[int, float]] = []  # (bar_idx, equity)
    position: Optional[OpenPosition] = None
    equity = position_size * 10  # Start with 100k
    peak_equity = equity
    max_dd = 0.0

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

            # Trailing stop distance (tightens as we profit)
            entry = position.entry_price
            trail_pct = abs(position.stop_price - entry) / entry  # Initial stop distance

            if position.direction == 'BUY':
                # Update best price
                if window_high > position.trailing_stop:
                    position.trailing_stop = window_high

                # Progressive trail: tighten as profit grows
                profit_from_best = (position.trailing_stop - entry) / entry
                if profit_from_best > 0.003:  # Once 0.3% profit locked in
                    # Trail from the best price at 50% of initial stop
                    trail_from_best = position.trailing_stop * (1 - trail_pct * 0.5)
                    effective_stop = max(position.stop_price, trail_from_best)
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
                elif bars_held >= max(6, int(position.ou_half_life * 3)):
                    should_exit = True
                    exit_reason = 'ou_timeout'

            else:  # SELL
                if position.trailing_stop == 0 or window_low < position.trailing_stop:
                    position.trailing_stop = window_low

                profit_from_best = (entry - position.trailing_stop) / entry
                if profit_from_best > 0.003:
                    trail_from_best = position.trailing_stop * (1 + trail_pct * 0.5)
                    effective_stop = min(position.stop_price, trail_from_best)
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
                elif bars_held >= max(6, int(position.ou_half_life * 3)):
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

                pnl = pnl_pct * position_size

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
                )
                trades.append(trade)
                equity += pnl
                equity_curve.append((bar, equity))
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity
                max_dd = max(max_dd, dd)
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

            # Add higher TF channels (use recent window, not full history)
            if use_multi_tf:
                for tf_label, tf_df in higher_tf_data.items():
                    # Use last 100 bars for channel detection (not full history)
                    tf_recent = tf_df.tail(100)
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
                if sig.action == 'BUY':
                    stop = entry_price * (1 - sig.suggested_stop_pct)
                    tp = entry_price * (1 + sig.suggested_tp_pct)
                else:
                    stop = entry_price * (1 + sig.suggested_stop_pct)
                    tp = entry_price * (1 - sig.suggested_tp_pct)

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
                    ou_half_life=ou_hl,
                    max_hold_bars=max_hold_bars,
                    trailing_stop=entry_price,
                )

    # Close any remaining position
    if position is not None:
        exit_price = float(closes[-1])
        if position.direction == 'BUY':
            pnl_pct = (exit_price - position.entry_price) / position.entry_price
        else:
            pnl_pct = (position.entry_price - exit_price) / position.entry_price
        pnl = pnl_pct * position_size
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

    return metrics, trades, equity_curve


def main():
    parser = argparse.ArgumentParser(description='Channel Surfer Backtest')
    parser.add_argument('--days', type=int, default=30, help='Days of 5min data')
    parser.add_argument('--eval-interval', type=int, default=6, help='Bars between evaluations')
    parser.add_argument('--max-hold', type=int, default=60, help='Max bars to hold')
    parser.add_argument('--min-conf', type=float, default=0.45, help='Minimum signal confidence')
    args = parser.parse_args()

    run_backtest(
        days=args.days,
        eval_interval=args.eval_interval,
        max_hold_bars=args.max_hold,
        min_confidence=args.min_conf,
    )


if __name__ == '__main__':
    main()
