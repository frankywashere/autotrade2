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

from .algo_base import AlgoBase, AlgoConfig, Signal, ExitSignal
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

    def run(self) -> Dict[str, dict]:
        """Run the backtest. Returns metrics dict per algo_id."""
        from .results import compute_metrics, print_report, print_summary_table

        t0 = _time.time()
        total_1m_bars = len(self.data._df1m)

        if self.verbose:
            algo_names = [a.algo_id for a in self.algos]
            print(f"\nUnified Backtester — {len(self.algos)} algo(s): {algo_names}")
            print(f"Data: {self.data.start_time} to {self.data.end_time} ({total_1m_bars:,} 1-min bars)")
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
                self._fill_pending(algo, ts, bar_1m)

                # ---- STEP 2: Process exits BEFORE updating best/worst (causal) ----
                # Stop/trail level is whatever was known at bar open.
                # High/low ratcheting happens AFTER exit checks (Step 3).
                positions = self.portfolio.get_open_positions(algo_id)
                if positions and ts in self._algo_exit_bar_times.get(algo_id, set()):
                    # Get the exit-resolution bar (may be 1min, 5min, or daily)
                    exit_bar = self.data.get_current_bar(algo.config.exit_check_tf, ts)
                    if exit_bar is None:
                        exit_bar = bar_1m
                    positions = self.portfolio.get_open_positions(algo_id)
                    exits = algo.check_exits(ts, exit_bar, positions)
                    for ex in exits:
                        trade = self.portfolio.close_position(
                            ex.pos_id, ex.price, ts, ex.reason)
                        if trade:
                            algo.on_fill(trade)

                # ---- STEP 3: Update best/worst prices AFTER exits (causal) ----
                # This ensures trail ratcheting only affects NEXT bar's stop level.
                # hold_bars only increments at exit-check TF boundaries so algos
                # count in their natural units (5-min bars, daily bars, etc.).
                is_exit_boundary = ts in self._algo_exit_bar_times.get(algo_id, set())
                for pos in self.portfolio.get_open_positions(algo_id):
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
                        signals = algo.on_bar(ts, primary_bar, positions)

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
