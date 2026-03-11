"""
BacktestEngine — Main simulation loop.

Walks forward through 1-min bars, dispatches to algorithms at their
primary TF, and manages position fills and exits.

Loop order per bar (causal — no intrabar lookahead):
1. Fill pending entries (at this bar's open)
2. Process exits using stop/trail known at bar open (before ratcheting)
3. Update best/worst prices (ratchet trail — effective next bar)
4. Generate new signals → queue as pending entries for next bar
"""

import time as _time
from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np
import pandas as pd

from .algo_base import AlgoBase, AlgoConfig, Signal, ExitSignal, TradeContext
from .data_provider import DataProvider
from .portfolio import PortfolioManager, Position


@dataclass
class PendingEntry:
    """An entry signal waiting to be filled at next bar's open."""
    signal: Signal
    algo: AlgoBase
    queued_time: pd.Timestamp
    fill_at: str = 'next_primary_open'  # 'next_primary_open' or 'next_rth_open'


class BacktestEngine:
    """Unified backtesting engine that walks 1-min bars and dispatches to algos."""

    def __init__(self, data: DataProvider, algos: List[AlgoBase],
                 portfolio: PortfolioManager, verbose: bool = True):
        self.data = data
        self.algos = algos
        self.portfolio = portfolio
        self.verbose = verbose

        # Build a lookup of algo_id -> AlgoBase
        self._algo_map: Dict[str, AlgoBase] = {a.algo_id: a for a in algos}

        # Pre-compute bar timestamps for each algo's primary TF
        # For daily+ TFs, map to the last 1-min bar of each period so they
        # match the 1-min walk timestamps
        self._algo_bar_times: Dict[str, Set[pd.Timestamp]] = {}
        for algo in algos:
            tf = algo.config.primary_tf
            tf_df = data._tf_data.get(tf)
            if tf_df is not None:
                self._algo_bar_times[algo.algo_id] = self._remap_tf_times(
                    tf, tf_df, data._df1m)
            else:
                raise ValueError(f"Primary TF '{tf}' not available for algo '{algo.algo_id}'")

        # Pre-compute bar timestamps for exit checking TFs
        self._algo_exit_bar_times: Dict[str, Set[pd.Timestamp]] = {}
        for algo in algos:
            etf = algo.config.exit_check_tf
            etf_df = data._tf_data.get(etf)
            if etf_df is not None:
                self._algo_exit_bar_times[algo.algo_id] = self._remap_tf_times(
                    etf, etf_df, data._df1m)
            else:
                # Fallback to 1-min
                self._algo_exit_bar_times[algo.algo_id] = set(data._df1m.index)

        # Map daily bar dates to first 1-min bar (for signal generation at day open)
        self._daily_first_bar: Dict = {}  # {date: pd.Timestamp}
        self._daily_last_bar: Dict = {}   # {date: pd.Timestamp}
        for day in sorted(set(data._df1m.index.date)):
            day_bars = data._df1m[data._df1m.index.date == day]
            if len(day_bars) > 0:
                self._daily_first_bar[day] = day_bars.index[0]
                self._daily_last_bar[day] = day_bars.index[-1]

        # Track eval interval counters per algo
        self._eval_counters: Dict[str, int] = {a.algo_id: 0 for a in algos}

        # Pending entries per algo (filled at next bar's open)
        self._pending: Dict[str, List[PendingEntry]] = {a.algo_id: [] for a in algos}

        # Broker stop checking state (for fixed/pessimistic modes)
        self._broker_stops: Dict[str, float] = {}          # pos_id → locked stop
        self._broker_check_counters: Dict[str, int] = {}   # pos_id → bar counter
        self._broker_stop_algos: Set[str] = {
            a.algo_id for a in algos if a.config.stop_check_mode in ('fixed', 'pessimistic')
        }

        # Sequential stop checking state (per-1m-bar, no ordering ambiguity)
        self._sequential_algos: Set[str] = {
            a.algo_id for a in algos if a.config.stop_check_mode == 'sequential'
        }
        self._entry_bar_counts: Dict[str, int] = {}  # pos_id → 1-min bars since entry
        self._5s_bar_counts: Dict[str, int] = {}     # pos_id → running 5-sec bar counter

        # 5-sec data: use honest fills when available
        self._has_5s_data = hasattr(data, '_5s_by_minute') and data._5s_by_minute is not None

    def _build_trade_context(self, algo_id: str) -> TradeContext:
        """Build TradeContext from PortfolioManager state for ML features."""
        trades = self.portfolio.get_trades(algo_id)
        recent = trades[-10:] if trades else []
        # Convert to dicts for TradeContext
        recent_dicts = []
        for t in recent:
            recent_dicts.append({
                'pnl': t.net_pnl,
                'pnl_pct': t.pnl_pct,
            })
        # Win/loss streaks
        win_streak = 0
        loss_streak = 0
        for t in reversed(trades):
            if t.net_pnl > 0:
                if loss_streak > 0:
                    break
                win_streak += 1
            elif t.net_pnl < 0:
                if win_streak > 0:
                    break
                loss_streak += 1
        # Daily P&L (approximate: sum of recent closed trades)
        daily_pnl = sum(t.net_pnl for t in trades[-5:]) if trades else 0.0
        equity = self.portfolio.get_equity(algo_id)
        return TradeContext(
            recent_trades=recent_dicts,
            daily_pnl=daily_pnl,
            win_streak=win_streak,
            loss_streak=loss_streak,
            equity=equity,
        )

    @staticmethod
    def _remap_tf_times(tf: str, tf_df: 'pd.DataFrame',
                        df1m: 'pd.DataFrame') -> Set[pd.Timestamp]:
        """Remap TF bar timestamps to match 1-min walk timestamps.

        For ALL TFs, map to the last 1-min bar within each period so
        dispatch happens after bar completion (no lookahead).
        - Daily/weekly/monthly: last 1-min bar of each calendar day
        - Intraday (5min, 1h, etc.): last 1-min bar before next TF bar start
        """
        if tf in ('daily', 'weekly', 'monthly'):
            # Map each TF bar to the last 1-min bar within that period
            remapped = set()
            dates_1m = df1m.index.date
            for tf_ts in tf_df.index:
                tf_date = tf_ts.date() if hasattr(tf_ts, 'date') else tf_ts
                day_mask = dates_1m == tf_date
                if day_mask.any():
                    last_1m = df1m.index[day_mask][-1]
                    remapped.add(last_1m)
            return remapped

        # Intraday TFs: map each bar to the last 1-min bar before the next
        # TF bar starts so dispatch happens only after the TF bar completes.
        remapped = set()
        idx_1m = df1m.index
        tf_index = tf_df.index
        dates_1m = idx_1m.date

        for i, tf_ts in enumerate(tf_index):
            if i + 1 < len(tf_index):
                next_tf_ts = tf_index[i + 1]
                end_pos = idx_1m.searchsorted(next_tf_ts, side='left') - 1
                if end_pos >= 0:
                    remapped.add(idx_1m[end_pos])
                continue

            day_mask = dates_1m == tf_ts.date()
            if day_mask.any():
                remapped.add(idx_1m[day_mask][-1])

        return remapped

    def _check_broker_stops(self, algo: AlgoBase, ts: pd.Timestamp, bar_1m: dict):
        """Check locked broker stops against 1-min bar on non-boundary bars.

        For 'fixed' mode: exit at locked stop price.
        For 'pessimistic' mode: exit at bar's worst price (gap-through stress test).
        """
        algo_id = algo.algo_id
        mode = algo.config.stop_check_mode
        interval = algo.config.stop_check_interval
        delay = algo.config.stop_check_delay
        closed_ids = []

        for pos in self.portfolio.get_open_positions(algo_id):
            pid = pos.pos_id
            locked_stop = self._broker_stops.get(pid)
            if locked_stop is None:
                continue

            # Increment counter
            self._broker_check_counters[pid] = self._broker_check_counters.get(pid, 0) + 1

            # Skip if still in delay window (simulates GoodAfterTime)
            if self._broker_check_counters[pid] <= delay:
                continue

            # After delay, check every `interval` bars
            bars_since_delay = self._broker_check_counters[pid] - delay
            if bars_since_delay % interval != 0:
                continue

            # Check if stop is breached
            breached = False
            if pos.direction == 'long' and bar_1m['low'] <= locked_stop:
                breached = True
                if mode == 'pessimistic':
                    exit_price = bar_1m['low']
                else:
                    exit_price = locked_stop
            elif pos.direction == 'short' and bar_1m['high'] >= locked_stop:
                breached = True
                if mode == 'pessimistic':
                    exit_price = bar_1m['high']
                else:
                    exit_price = locked_stop

            if breached:
                trade = self.portfolio.close_position(
                    pid, exit_price, ts, 'broker_stop')
                if trade:
                    algo.on_fill(trade)
                    closed_ids.append(pid)

        # Clean up closed positions
        for pid in closed_ids:
            self._broker_stops.pop(pid, None)
            self._broker_check_counters.pop(pid, None)

    def _check_sequential_stops(self, algo: AlgoBase, ts: pd.Timestamp, bar_1m: dict):
        """Check stops per-1m-bar with grace period. No ordering ambiguity.

        For each open position:
        1. Skip if still in grace period (first N bars after entry)
        2. Skip if not on check interval (first check right after grace, then every N)
        3. Compute effective_stop from get_effective_stop() (uses prior best_price)
        4. Check configured price field(s) against effective_stop
        5. Exit at effective_stop or bar close depending on mode

        Ratcheting (Step 3 in main loop) happens AFTER this, so effective_stop
        is always computed from prior bars — preserving chronological order.

        Configurable via AlgoConfig:
        - seq_check_price: 'low', 'open', 'close', 'open_close', 'open_fill_close'
        - seq_check_interval: 1 (every bar), 5 (every 5th bar), etc.

        open_fill_close mode: check open against stop. If breached, exit at
        bar's close (realistic market order fill). If open OK, trade survives.
        """
        algo_id = algo.algo_id
        grace = algo.config.exit_grace_bars
        check_price = algo.config.seq_check_price
        check_interval = algo.config.seq_check_interval
        closed_ids = []

        for pos in self.portfolio.get_open_positions(algo_id):
            pid = pos.pos_id
            count = self._entry_bar_counts.get(pid, 0) + 1
            self._entry_bar_counts[pid] = count

            if count <= grace:
                continue

            # Check interval: first check right after grace, then every N bars
            bars_since_grace = count - grace
            if check_interval > 1 and (bars_since_grace - 1) % check_interval != 0:
                continue

            effective_stop = algo.get_effective_stop(pos)
            if effective_stop is None:
                continue

            # Profit-activated stop: skip breach check until in profit
            if algo.config.profit_activated_stop:
                underwater = ((pos.direction == 'long' and pos.best_price <= pos.entry_price) or
                              (pos.direction == 'short' and pos.best_price >= pos.entry_price))
                if underwater:
                    max_uw = algo.config.max_underwater_mins
                    if max_uw > 0 and count > max_uw:
                        trade = self.portfolio.close_position(
                            pid, bar_1m['close'], ts, 'uw_timeout')
                        if trade:
                            algo.on_fill(trade)
                            closed_ids.append(pid)
                    continue

            breached = False
            # For 'low' with interval=1: exit at effective_stop (resting IB stop,
            # price passed through the level). For all other modes: exit at the
            # price that triggered the breach (can't fill at stop if price already past).
            fill_at_stop = (check_price == 'low' and check_interval == 1)
            exit_price = effective_stop if fill_at_stop else bar_1m['close']

            if pos.direction == 'long':
                if check_price == 'open_fill_close':
                    # Check open; if breached, fill at close (market order)
                    if bar_1m['open'] <= effective_stop:
                        breached = True
                        exit_price = bar_1m['close']
                elif check_price == 'open_close':
                    # Check open first, then close; fill at close
                    if bar_1m['open'] <= effective_stop or bar_1m['close'] <= effective_stop:
                        breached = True
                elif check_price == 'low':
                    if bar_1m['low'] <= effective_stop:
                        breached = True
                        # low/1m: fill at stop (resting order). low/5m: fill at close
                        exit_price = effective_stop if fill_at_stop else bar_1m['close']
                else:
                    # 'open' or 'close'
                    if bar_1m[check_price] <= effective_stop:
                        breached = True

            elif pos.direction == 'short':
                if check_price == 'open_fill_close':
                    if bar_1m['open'] >= effective_stop:
                        breached = True
                        exit_price = bar_1m['close']
                elif check_price == 'open_close':
                    if bar_1m['open'] >= effective_stop or bar_1m['close'] >= effective_stop:
                        breached = True
                elif check_price == 'low':
                    # For shorts, check high
                    if bar_1m['high'] >= effective_stop:
                        breached = True
                        exit_price = effective_stop if fill_at_stop else bar_1m['close']
                else:
                    if bar_1m[check_price] >= effective_stop:
                        breached = True

            if breached:
                reason = 'trail' if effective_stop != pos.stop_price else 'stop'
                trade = self.portfolio.close_position(pid, exit_price, ts, reason)
                if trade:
                    algo.on_fill(trade)
                    closed_ids.append(pid)

        for pid in closed_ids:
            self._entry_bar_counts.pop(pid, None)

    def _check_sequential_stops_5s(self, algo: AlgoBase, ts: pd.Timestamp, bar_1m: dict):
        """5-sec honest fill stop check with configurable knobs.

        Two independent knobs control the sub-loop:
        - stop_update_secs: how often to ratchet best_price + recompute effective_stop
        - stop_check_secs: how often to check if price breached the stop

        Both are in seconds (5=every 5s bar, 60=every 1min, 300=every 5min).
        The sub-loop walks all 5-sec bars but only acts on the configured intervals.

        Honest fills: gap-through → fill at open; crossed → fill at stop.
        """
        algo_id = algo.algo_id
        grace = algo.config.exit_grace_bars
        check_interval = algo.config.seq_check_interval
        update_every_n = max(1, algo.config.stop_update_secs // 5)  # 5s bars between updates
        check_every_n = max(1, algo.config.stop_check_secs // 5)    # 5s bars between checks
        grace_ratchet_n = algo.config.grace_ratchet_secs // 5 if algo.config.grace_ratchet_secs > 0 else 0
        profit_gate = algo.config.profit_activated_stop
        closed_ids = []

        # Get 5-sec bars for this minute
        bars_5s = self.data.get_5s_bars_for_minute(ts)
        if bars_5s is None or len(bars_5s) == 0:
            # Fall back to 1-min check if no 5s data for this minute
            self._check_sequential_stops(algo, ts, bar_1m)
            return

        for pos in self.portfolio.get_open_positions(algo_id):
            pid = pos.pos_id
            if pid in closed_ids:
                continue

            count = self._entry_bar_counts.get(pid, 0) + 1
            self._entry_bar_counts[pid] = count

            if count <= grace:
                # During grace: ratchet only (no exit checks) at configured interval
                if grace_ratchet_n > 0:
                    bar5_count = self._5s_bar_counts.get(pid, 0)
                    for _, bar5 in bars_5s.iterrows():
                        bar5_count += 1
                        if bar5_count % grace_ratchet_n == 0:
                            self.portfolio.update_position(
                                pid, float(bar5['high']), float(bar5['low']))
                    self._5s_bar_counts[pid] = bar5_count
                continue

            # Check interval at 1-min level: first check right after grace, then every N
            bars_since_grace = count - grace
            if check_interval > 1 and (bars_since_grace - 1) % check_interval != 0:
                continue

            # Compute initial effective_stop for this minute
            effective_stop = algo.get_effective_stop(pos)
            if effective_stop is None:
                continue

            # Get the running 5s bar counter for this position (persists across minutes)
            bar5_count = self._5s_bar_counts.get(pid, 0)

            exited = False
            exit_price = 0.0
            exit_reason_override = None
            for _, bar5 in bars_5s.iterrows():
                bar5_count += 1
                b_open = float(bar5['open'])
                b_high = float(bar5['high'])
                b_low  = float(bar5['low'])

                # Knob A: ratchet + recompute stop on schedule
                if bar5_count % update_every_n == 0:
                    self.portfolio.update_position(pid, b_high, b_low)
                    effective_stop = algo.get_effective_stop(pos)
                    if effective_stop is None:
                        break

                # Knob B: check if price breached the stop on schedule
                if bar5_count % check_every_n != 0:
                    continue

                # Profit-activated stop: skip breach check until in profit
                if profit_gate:
                    underwater = ((pos.direction == 'long' and pos.best_price <= pos.entry_price) or
                                  (pos.direction == 'short' and pos.best_price >= pos.entry_price))
                    if underwater:
                        # Check underwater timeout (count = 1-min bars since entry)
                        max_uw = algo.config.max_underwater_mins
                        if max_uw > 0 and count > max_uw:
                            exit_price = float(bar5['close'])
                            exited = True
                            exit_reason_override = 'uw_timeout'
                            break
                        continue

                b_close = float(bar5['close'])

                if pos.direction == 'long':
                    # 1. Check open — gap-through: fill at open
                    if b_open <= effective_stop:
                        exit_price = b_open
                        exited = True
                        break
                    # 2. Check if stop is between open and close (crossed through)
                    #    In 5 seconds, reasonable that it hit the stop level
                    if b_close <= effective_stop:
                        exit_price = effective_stop
                        exited = True
                        break
                else:  # short
                    # 1. Check open — gap-through: fill at open
                    if b_open >= effective_stop:
                        exit_price = b_open
                        exited = True
                        break
                    # 2. Check if stop is between open and close (crossed through)
                    if b_close >= effective_stop:
                        exit_price = effective_stop
                        exited = True
                        break

            # Persist counter
            self._5s_bar_counts[pid] = bar5_count

            if exited:
                reason = exit_reason_override or ('trail' if effective_stop != pos.stop_price else 'stop')
                trade = self.portfolio.close_position(pid, exit_price, ts, reason)
                if trade:
                    algo.on_fill(trade)
                    closed_ids.append(pid)

        for pid in closed_ids:
            self._entry_bar_counts.pop(pid, None)
            self._5s_bar_counts.pop(pid, None)

    def run(self) -> Dict[str, dict]:
        """Run the backtest. Returns metrics dict per algo_id."""
        from .results import compute_metrics, print_report, print_summary_table

        t0 = _time.time()
        total_1m_bars = len(self.data._df1m)

        if self.verbose:
            algo_names = [a.algo_id for a in self.algos]
            print(f"\nUnified Backtester — {len(self.algos)} algo(s): {algo_names}")
            print(f"Data: {self.data.start_time} to {self.data.end_time} ({total_1m_bars:,} 1-min bars)")
            if self._has_5s_data:
                n_5s = len(self.data._df5s) if hasattr(self.data, '_df5s') else 0
                print(f"  5-sec bars: {n_5s:,} (honest fill mode)")
            print(f"Running...\n")

        prev_day = None
        bars_processed = 0

        for ts, bar_1m in self.data.iter_bars('1min'):
            current_day = ts.date()
            new_day = current_day != prev_day
            prev_day = current_day

            for algo in self.algos:
                algo_id = algo.algo_id

                # ---- STEP 1: Fill pending entries at this bar's open ----
                positions_before = set(p.pos_id for p in self.portfolio.get_open_positions(algo_id))
                self._fill_pending(algo, ts, bar_1m)

                # Lock initial broker stop for newly filled positions
                if algo_id in self._broker_stop_algos:
                    for pos in self.portfolio.get_open_positions(algo_id):
                        if pos.pos_id not in positions_before:
                            initial_stop = algo.get_effective_stop(pos)
                            if initial_stop is not None:
                                self._broker_stops[pos.pos_id] = initial_stop
                                self._broker_check_counters[pos.pos_id] = 0

                # Initialize entry bar counter for newly filled positions (sequential mode)
                if algo_id in self._sequential_algos:
                    for pos in self.portfolio.get_open_positions(algo_id):
                        if pos.pos_id not in positions_before:
                            self._entry_bar_counts[pos.pos_id] = 0

                is_exit_boundary = ts in self._algo_exit_bar_times.get(algo_id, set())

                # ---- STEP 1.5a: Sequential stop check (EVERY bar) ----
                if algo_id in self._sequential_algos:
                    if self._has_5s_data:
                        self._check_sequential_stops_5s(algo, ts, bar_1m)
                    else:
                        self._check_sequential_stops(algo, ts, bar_1m)

                # ---- STEP 1.5b: Broker stop check on non-boundary bars ----
                elif algo_id in self._broker_stop_algos and not is_exit_boundary:
                    self._check_broker_stops(algo, ts, bar_1m)

                # ---- STEP 2: Process exits BEFORE updating best/worst (causal) ----
                # Stop/trail level is whatever was known at bar open.
                # High/low ratcheting happens AFTER exit checks (Step 3).
                positions = self.portfolio.get_open_positions(algo_id)
                if positions and is_exit_boundary:
                    # Get the exit-resolution bar (may be 1min, 5min, or daily)
                    exit_bar = self.data.get_current_bar(algo.config.exit_check_tf, ts)
                    if exit_bar is None:
                        exit_bar = bar_1m
                    positions = self.portfolio.get_open_positions(algo_id)
                    exits = algo.check_exits(ts, exit_bar, positions)
                    for ex in exits:
                        # In sequential mode, stop/trail already handled per-1m-bar.
                        # Only process non-stop exits (TP, timeout, EOD) from check_exits.
                        if algo_id in self._sequential_algos and ex.reason in ('stop', 'trail'):
                            continue
                        trade = self.portfolio.close_position(
                            ex.pos_id, ex.price, ts, ex.reason)
                        if trade:
                            algo.on_fill(trade)
                            # Clean up broker dicts for closed positions
                            self._broker_stops.pop(ex.pos_id, None)
                            self._broker_check_counters.pop(ex.pos_id, None)
                            self._entry_bar_counts.pop(ex.pos_id, None)
                            self._5s_bar_counts.pop(ex.pos_id, None)

                # ---- STEP 2.5: Re-lock broker stops for survivors at boundaries ----
                if algo_id in self._broker_stop_algos and is_exit_boundary:
                    for pos in self.portfolio.get_open_positions(algo_id):
                        stop = algo.get_effective_stop(pos)
                        if stop is not None:
                            self._broker_stops[pos.pos_id] = stop
                            self._broker_check_counters[pos.pos_id] = 0

                # ---- STEP 3: Update best/worst prices AFTER exits (causal) ----
                # Ratchet best/worst on every 1-min bar to capture true price
                # extremes (matching broker-side stops that track continuously
                # in live). hold_bars only increments at exit-check TF boundaries
                # so algos count in their natural units (5-min bars).
                no_grace_ratchet = (algo.config.grace_ratchet_secs == 0
                                    and algo_id in self._sequential_algos)
                for pos in self.portfolio.get_open_positions(algo_id):
                    # Skip ratcheting during grace if grace_ratchet_secs=0
                    if no_grace_ratchet:
                        count = self._entry_bar_counts.get(pos.pos_id, 0)
                        if count <= algo.config.exit_grace_bars:
                            continue
                    self.portfolio.update_position(
                        pos.pos_id, bar_1m['high'], bar_1m['low'],
                        increment_hold=is_exit_boundary)

                # ---- STEP 4: Generate new signals → queue as pending ----
                if ts in self._algo_bar_times.get(algo_id, set()):
                    self._eval_counters[algo_id] += 1
                    if self._eval_counters[algo_id] >= algo.config.eval_interval:
                        self._eval_counters[algo_id] = 0

                        primary_bar = self.data.get_current_bar(algo.config.primary_tf, ts)
                        if primary_bar is None:
                            primary_bar = bar_1m

                        # Active hours hint: skip on_bar outside active window
                        # (exits still run above regardless)
                        bar_time = primary_bar.get('time', ts).time()
                        if algo.config.active_start is not None and bar_time < algo.config.active_start:
                            continue
                        if algo.config.active_end is not None and bar_time > algo.config.active_end:
                            continue

                        positions = self.portfolio.get_open_positions(algo_id)
                        context = self._build_trade_context(algo_id)
                        signals = algo.on_bar(ts, primary_bar, positions,
                                              context=context)

                        for sig in signals:
                            fill_at = 'next_rth_open' if sig.delayed_entry else 'next_primary_open'
                            self._pending[algo_id].append(PendingEntry(
                                signal=sig, algo=algo, queued_time=ts, fill_at=fill_at,
                            ))

            # Record equity periodically
            bars_processed += 1
            if bars_processed % 390 == 0:
                self.portfolio.record_equity(ts)

            if self.verbose and bars_processed % 50000 == 0:
                pct = bars_processed / total_1m_bars * 100
                elapsed = _time.time() - t0
                print(f"  {bars_processed:,}/{total_1m_bars:,} ({pct:.0f}%) — {elapsed:.0f}s")

        # End of data: close remaining positions
        final_time = self.data.end_time
        final_price = self.data.get_price_at(final_time)
        for algo in self.algos:
            for pos in self.portfolio.get_open_positions(algo.algo_id):
                trade = self.portfolio.close_position(
                    pos.pos_id, final_price, final_time, 'end_of_data')
                if trade:
                    algo.on_fill(trade)
                    self._broker_stops.pop(pos.pos_id, None)
                    self._broker_check_counters.pop(pos.pos_id, None)
                    self._entry_bar_counts.pop(pos.pos_id, None)
                    self._5s_bar_counts.pop(pos.pos_id, None)

        self.portfolio.record_equity(final_time)

        elapsed = _time.time() - t0
        if self.verbose:
            print(f"\nDone in {elapsed:.1f}s ({bars_processed:,} bars processed)")

        results = {}
        algo_ids = [a.algo_id for a in self.algos]
        for algo in self.algos:
            trades = self.portfolio.get_trades(algo.algo_id)
            initial = self.portfolio._algos[algo.algo_id].initial_equity
            m = compute_metrics(trades, initial)
            results[algo.algo_id] = m
            if self.verbose:
                print_report(trades, algo.algo_id, initial)

        if self.verbose and len(self.algos) > 1:
            print_summary_table(self.portfolio, algo_ids)

        return results

    def _fill_pending(self, algo: AlgoBase, time: pd.Timestamp, bar: dict):
        """Fill pending entries at this bar's open price."""
        algo_id = algo.algo_id
        if not self._pending[algo_id]:
            return

        to_fill = []
        remaining = []

        for pe in self._pending[algo_id]:
            # Don't fill on the same bar the signal was generated
            if pe.queued_time >= time:
                remaining.append(pe)
                continue

            if pe.fill_at == 'next_rth_open':
                # Only fill at RTH open (9:30 bar of a new day)
                if time.time() == pd.Timestamp('09:30').time() and pe.queued_time.date() < time.date():
                    to_fill.append(pe)
                else:
                    remaining.append(pe)
            else:
                # next_primary_open: fill at the next 1-min bar's open
                # (signal fires at bar-end, so next 1-min bar = next TF bar's open)
                to_fill.append(pe)

        self._pending[algo_id] = remaining

        for pe in to_fill:
            sig = pe.signal
            sig.price = bar['open']  # Fill at bar's open
            self._execute_entry(sig, pe.algo, time)

    def _execute_entry(self, sig: Signal, algo: AlgoBase, time: pd.Timestamp):
        """Execute a single entry signal."""
        if not self.portfolio.can_open(sig.algo_id):
            return

        price = sig.price
        if price <= 0:
            return

        # Apply slippage to entry price
        cost = self.portfolio._algos[sig.algo_id].cost_model
        if sig.direction == 'long':
            price = price * (1.0 + cost.slippage_pct)
        else:
            price = price * (1.0 - cost.slippage_pct)

        shares = sig.shares
        if shares <= 0:
            flat = algo.config.params.get('flat_sizing', True)
            shares = self.portfolio.compute_shares(
                sig.algo_id, price, confidence=sig.confidence, flat_sizing=flat)
        if shares <= 0:
            return

        if sig.direction == 'long':
            stop_price = price * (1.0 - sig.stop_pct)
            tp_price = price * (1.0 + sig.tp_pct)
        else:
            stop_price = price * (1.0 + sig.stop_pct)
            tp_price = price * (1.0 - sig.tp_pct)

        pos = self.portfolio.open_position(
            algo_id=sig.algo_id,
            direction=sig.direction,
            price=price,
            shares=shares,
            stop_price=stop_price,
            tp_price=tp_price,
            confidence=sig.confidence,
            signal_type=sig.signal_type,
            time=time,
            metadata=sig.metadata,
        )
        if pos:
            algo.on_position_opened(pos)
