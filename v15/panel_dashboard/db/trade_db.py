"""
Trade database — SQLite single source of truth for all trade records.

Replaces per-scanner JSON state files. Thread-safe via RLock + BEGIN IMMEDIATE.
"""

import json
import logging
import os
import sqlite3
import threading
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source          TEXT NOT NULL,
    algo_id         TEXT NOT NULL,
    symbol          TEXT NOT NULL DEFAULT 'TSLA',
    direction       TEXT NOT NULL,
    entry_time      TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    exit_time       TEXT,
    exit_price      REAL,
    shares          INTEGER NOT NULL,
    stop_price      REAL NOT NULL,
    tp_price        REAL NOT NULL,
    confidence      REAL,
    signal_type     TEXT,
    exit_reason     TEXT,
    pnl             REAL,
    pnl_pct         REAL,
    best_price      REAL,
    worst_price     REAL,
    trail_width     REAL,
    hold_bars       INTEGER DEFAULT 0,
    breakeven_applied INTEGER DEFAULT 0,
    ou_half_life      REAL,
    el_flagged        INTEGER DEFAULT 0,
    trail_width_mult  REAL DEFAULT 1.0,
    ib_entry_order_id INTEGER,
    ib_exit_order_id  INTEGER,
    ib_perm_id        INTEGER,
    ib_fill_status    TEXT DEFAULT 'pending',
    filled_shares     INTEGER DEFAULT 0,
    open_shares       INTEGER DEFAULT 0,
    avg_fill_price    REAL,
    exit_filled_shares INTEGER DEFAULT 0,
    avg_exit_price    REAL,
    ib_exit_perm_id   INTEGER,
    ib_stop_order_id  INTEGER,
    ib_stop_perm_id   INTEGER,
    management_mode TEXT DEFAULT 'algo',
    legacy_pos_id   TEXT,
    metadata        TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_legacy
    ON trades(legacy_pos_id) WHERE legacy_pos_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_trades_source_algo ON trades(source, algo_id);
CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(exit_time) WHERE exit_time IS NULL;
CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(entry_time);

CREATE TABLE IF NOT EXISTS daily_pnl (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,
    source      TEXT NOT NULL,
    algo_id     TEXT NOT NULL,
    pnl         REAL NOT NULL,
    trades      INTEGER NOT NULL,
    wins        INTEGER NOT NULL,
    UNIQUE(date, source, algo_id)
);

CREATE TABLE IF NOT EXISTS signals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    time        TEXT NOT NULL,
    source      TEXT NOT NULL,
    algo_id     TEXT NOT NULL,
    action      TEXT NOT NULL,
    confidence  REAL,
    rejected    INTEGER DEFAULT 0,
    reject_reason TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(time DESC);

CREATE TABLE IF NOT EXISTS metadata (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
"""

# Columns in the trades table (order matches schema)
_TRADE_COLUMNS = [
    'id', 'source', 'algo_id', 'symbol', 'direction', 'entry_time',
    'entry_price', 'exit_time', 'exit_price', 'shares', 'stop_price',
    'tp_price', 'confidence', 'signal_type', 'exit_reason', 'pnl', 'pnl_pct',
    'best_price', 'worst_price', 'trail_width', 'hold_bars',
    'breakeven_applied', 'ou_half_life', 'el_flagged', 'trail_width_mult',
    'ib_entry_order_id', 'ib_exit_order_id', 'ib_perm_id', 'ib_fill_status',
    'filled_shares', 'open_shares', 'avg_fill_price', 'exit_filled_shares',
    'avg_exit_price', 'ib_exit_perm_id', 'ib_stop_order_id',
    'ib_stop_perm_id', 'management_mode', 'legacy_pos_id', 'metadata', 'created_at',
]


def _row_to_dict(row):
    """Convert a sqlite3.Row or tuple to a dict."""
    if row is None:
        return None
    return dict(zip(_TRADE_COLUMNS, row))


class TradeDB:
    """Thread-safe SQLite trade database."""

    def __init__(self, db_path='~/.x14/trades.db'):
        self._db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.RLock()

        self._create_tables()

    def _create_tables(self):
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # Migration: add management_mode column if missing (existing DBs)
            try:
                self._conn.execute(
                    "ALTER TABLE trades ADD COLUMN management_mode TEXT DEFAULT 'algo'")
                self._conn.commit()
                logger.info("Migration: added management_mode column")
            except sqlite3.OperationalError:
                pass  # Column already exists

    # ------------------------------------------------------------------
    # Core trade operations
    # ------------------------------------------------------------------

    def open_trade(self, source, algo_id, symbol, direction, entry_time,
                   entry_price, shares, stop_price, tp_price, confidence,
                   signal_type, best_price=None, worst_price=None,
                   trail_width=None, hold_bars=0, breakeven_applied=False,
                   ou_half_life=None, el_flagged=False, trail_width_mult=1.0,
                   ib_entry_order_id=None, ib_perm_id=None,
                   ib_fill_status='pending',
                   filled_shares=0, open_shares=0, avg_fill_price=None,
                   exit_filled_shares=0, avg_exit_price=None,
                   ib_exit_order_id=None, ib_exit_perm_id=None,
                   ib_stop_order_id=None, ib_stop_perm_id=None,
                   legacy_pos_id=None,
                   metadata=None) -> int:
        """Insert a new trade row. Returns trade_id."""
        if best_price is None:
            best_price = entry_price
        if worst_price is None:
            worst_price = entry_price
        meta_json = json.dumps(metadata) if metadata and not isinstance(metadata, str) else metadata

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._conn.execute("""
                    INSERT INTO trades (
                        source, algo_id, symbol, direction, entry_time,
                        entry_price, shares, stop_price, tp_price, confidence,
                        signal_type, best_price, worst_price, trail_width,
                        hold_bars, breakeven_applied, ou_half_life, el_flagged,
                        trail_width_mult, ib_entry_order_id, ib_exit_order_id,
                        ib_perm_id, ib_fill_status, filled_shares, open_shares,
                        avg_fill_price, exit_filled_shares, avg_exit_price,
                        ib_exit_perm_id, ib_stop_order_id, ib_stop_perm_id,
                        legacy_pos_id, metadata
                    ) VALUES (
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?,
                        ?, ?, ?,
                        ?, ?
                    )
                """, (
                    source, algo_id, symbol, direction, entry_time,
                    entry_price, shares, stop_price, tp_price, confidence,
                    signal_type, best_price, worst_price, trail_width,
                    hold_bars, int(breakeven_applied), ou_half_life,
                    int(el_flagged), trail_width_mult,
                    ib_entry_order_id, ib_exit_order_id,
                    ib_perm_id, ib_fill_status, filled_shares, open_shares,
                    avg_fill_price, exit_filled_shares, avg_exit_price,
                    ib_exit_perm_id, ib_stop_order_id, ib_stop_perm_id,
                    legacy_pos_id, meta_json,
                ))
                trade_id = cursor.lastrowid
                self._conn.commit()
                return trade_id
            except Exception:
                self._conn.rollback()
                raise

    def close_trade(self, trade_id, exit_time, exit_price, exit_reason,
                    effective_filled_shares=None,
                    effective_avg_fill_price=None,
                    effective_avg_exit_price=None) -> dict:
        """Close a trade. Computes P&L, updates row, updates daily_pnl.

        Returns the closed trade dict.
        """
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT * FROM trades WHERE id = ?", (trade_id,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"Trade {trade_id} not found")
                trade = _row_to_dict(row)

                # Use effective values if provided (from _unapplied_fills)
                fill_shares = effective_filled_shares or trade['filled_shares'] or trade['shares']
                fill_price = effective_avg_fill_price or trade['avg_fill_price'] or trade['entry_price']
                ex_price = effective_avg_exit_price or exit_price

                direction_sign = 1 if trade['direction'] == 'long' else -1
                pnl = (ex_price - fill_price) * fill_shares * direction_sign
                pnl_pct = (ex_price - fill_price) / fill_price * direction_sign if fill_price else 0.0

                self._conn.execute("""
                    UPDATE trades SET
                        exit_time = ?, exit_price = ?, exit_reason = ?,
                        pnl = ?, pnl_pct = ?, open_shares = 0
                    WHERE id = ?
                """, (exit_time, exit_price, exit_reason, pnl, pnl_pct, trade_id))

                # Update daily_pnl
                exit_date = exit_time[:10] if exit_time else None
                if exit_date:
                    is_win = 1 if pnl > 0 else 0
                    self._conn.execute("""
                        INSERT INTO daily_pnl (date, source, algo_id, pnl, trades, wins)
                        VALUES (?, ?, ?, ?, 1, ?)
                        ON CONFLICT(date, source, algo_id)
                        DO UPDATE SET
                            pnl = pnl + excluded.pnl,
                            trades = trades + 1,
                            wins = wins + excluded.wins
                    """, (exit_date, trade['source'], trade['algo_id'], pnl, is_win))

                self._conn.commit()

                trade.update({
                    'exit_time': exit_time,
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'open_shares': 0,
                })
                return trade
            except Exception:
                self._conn.rollback()
                raise

    def update_trade_state(self, trade_id, **kwargs) -> None:
        """Persist runtime exit-critical state changes to DB."""
        if not kwargs:
            return
        allowed = {
            'best_price', 'worst_price', 'trail_width', 'hold_bars',
            'breakeven_applied', 'stop_price', 'open_shares',
            'ib_stop_order_id', 'ib_stop_perm_id', 'ib_exit_order_id',
            'ib_exit_perm_id', 'ib_fill_status', 'filled_shares',
            'avg_fill_price', 'exit_filled_shares', 'avg_exit_price',
            'entry_price', 'entry_time', 'shares', 'tp_price',
            'el_flagged', 'trail_width_mult', 'ou_half_life',
            'ib_entry_order_id', 'ib_perm_id', 'exit_reason',
            'management_mode',
        }
        invalid = set(kwargs) - allowed
        if invalid:
            raise ValueError(f"Cannot update: {invalid}")

        set_clause = ', '.join(f'{k} = ?' for k in kwargs)
        values = list(kwargs.values()) + [trade_id]

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    f"UPDATE trades SET {set_clause} WHERE id = ?", values
                )
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def delete_trade(self, trade_id) -> None:
        """Delete a trade row (used for zero-fill rejected entries)."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("DELETE FROM trades WHERE id = ?", (trade_id,))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_trade(self, trade_id) -> dict | None:
        """Get a single trade by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM trades WHERE id = ?", (trade_id,)
            ).fetchone()
            return _row_to_dict(row)

    def get_trade_by_order_id(self, order_id, side='entry') -> dict | None:
        """Look up trade by IB order_id (entry or exit side)."""
        col = 'ib_entry_order_id' if side == 'entry' else 'ib_exit_order_id'
        with self._lock:
            row = self._conn.execute(
                f"SELECT * FROM trades WHERE {col} = ?", (order_id,)
            ).fetchone()
            return _row_to_dict(row)

    def get_trade_by_stop_order_id(self, stop_order_id) -> dict | None:
        """Look up trade by protective stop order_id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM trades WHERE ib_stop_order_id = ?",
                (stop_order_id,)
            ).fetchone()
            return _row_to_dict(row)

    def get_trade_by_perm_id(self, perm_id) -> dict | None:
        """Look up trade by IB permanent order ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM trades WHERE ib_perm_id = ? OR ib_exit_perm_id = ? OR ib_stop_perm_id = ?",
                (perm_id, perm_id, perm_id)
            ).fetchone()
            return _row_to_dict(row)

    def get_open_trades(self, source=None, algo_id=None,
                        include_pending=False) -> list[dict]:
        """Get open trades (exit_time IS NULL).

        For source='ib':
          - Default: returns filled + partial rows (real broker exposure)
          - include_pending=True: also returns pending rows (for entry gating)
          - NEVER returns rejected rows
        """
        conditions = ["exit_time IS NULL"]
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source)

        if algo_id:
            conditions.append("algo_id = ?")
            params.append(algo_id)

        if source == 'ib':
            if include_pending:
                conditions.append("ib_fill_status IN ('pending', 'partial', 'filled')")
            else:
                conditions.append("ib_fill_status IN ('partial', 'filled')")

        where = ' AND '.join(conditions)

        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM trades WHERE {where}", params
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

    def get_closed_trades(self, source=None, algo_id=None,
                          since=None, limit=100) -> list[dict]:
        """Get closed trades, newest first."""
        conditions = ["exit_time IS NOT NULL"]
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source)
        if algo_id:
            conditions.append("algo_id = ?")
            params.append(algo_id)
        if since:
            conditions.append("exit_time >= ?")
            params.append(since)

        where = ' AND '.join(conditions)

        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM trades WHERE {where} ORDER BY exit_time DESC LIMIT ?",
                params + [limit]
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

    def get_daily_pnl(self, source=None, algo_id=None,
                      since=None) -> pd.DataFrame:
        """Get daily P&L summary."""
        conditions = []
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source)
        if algo_id:
            conditions.append("algo_id = ?")
            params.append(algo_id)
        if since:
            conditions.append("date >= ?")
            params.append(since)

        where = (' WHERE ' + ' AND '.join(conditions)) if conditions else ''

        with self._lock:
            df = pd.read_sql_query(
                f"SELECT * FROM daily_pnl{where} ORDER BY date",
                self._conn, params=params
            )
            return df

    def get_algo_summary(self, source=None) -> pd.DataFrame:
        """Per-algo summary: total P&L, day P&L, trade count, win rate, open count."""
        conditions = []
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source)

        # Exclude rejected/orphaned
        conditions.append("ib_fill_status NOT IN ('rejected', 'orphaned') OR ib_fill_status IS NULL")

        where = ' WHERE ' + ' AND '.join(conditions)

        with self._lock:
            df = pd.read_sql_query(f"""
                SELECT
                    algo_id,
                    source,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN exit_time IS NOT NULL THEN 1 ELSE 0 END) as closed_trades,
                    SUM(CASE WHEN exit_time IS NULL THEN 1 ELSE 0 END) as open_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN pnl <= 0 AND exit_time IS NOT NULL THEN 1 ELSE 0 END) as losses
                FROM trades
                {where}
                GROUP BY algo_id, source
                ORDER BY algo_id
            """, self._conn, params=params)
            return df

    # ------------------------------------------------------------------
    # Signal logging
    # ------------------------------------------------------------------

    def log_signal(self, time, source, algo_id, action, confidence=None,
                   rejected=0, reject_reason=None) -> None:
        """Log a signal event."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("""
                    INSERT INTO signals (time, source, algo_id, action,
                                         confidence, rejected, reject_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (time, source, algo_id, action, confidence,
                      rejected, reject_reason))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def get_recent_signals(self, source=None, algo_id=None,
                           limit=50) -> list[dict]:
        """Get recent signals, newest first."""
        conditions = []
        params = []

        if source:
            conditions.append("source = ?")
            params.append(source)
        if algo_id:
            conditions.append("algo_id = ?")
            params.append(algo_id)

        where = (' WHERE ' + ' AND '.join(conditions)) if conditions else ''
        sig_cols = ['id', 'time', 'source', 'algo_id', 'action',
                    'confidence', 'rejected', 'reject_reason', 'created_at']

        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM signals{where} ORDER BY time DESC LIMIT ?",
                params + [limit]
            ).fetchall()
            return [dict(zip(sig_cols, r)) for r in rows]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self, key) -> str | None:
        """Get a metadata value."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
            return row[0] if row else None

    def set_metadata(self, key, value) -> None:
        """Set a metadata value (upsert)."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute("""
                    INSERT INTO metadata (key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """, (key, str(value)))
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close the database connection."""
        with self._lock:
            self._conn.close()
