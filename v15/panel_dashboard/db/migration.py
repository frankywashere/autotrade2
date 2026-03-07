"""
Migration: JSON state files → SQLite trade database.

Imports existing ~/.x14/surfer_state_*.json into TradeDB on first startup.
Atomic — entire import is one transaction. Crash-safe via legacy_pos_id dedupe.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .trade_db import TradeDB

logger = logging.getLogger(__name__)

ET = ZoneInfo('US/Eastern')

# Map model_tag (from filename) to algo_id
MODEL_TAG_TO_ALGO = {
    'c16': 'c16',
    'c16-dw': 'c16-dw',
    'c16-ml': 'c16-ml',
    'c16-intra': 'c16-intra',
    'c16-oe': 'c16-oe',
    'c14a': 'c14a',
    'c14a-dw': 'c14a-dw',
    'c14a-ml': 'c14a-ml',
    'c14a-intra': 'c14a-intra',
    'yf-c16': 'yf-c16',
    'yf-c16-dw': 'yf-c16-dw',
    'yf-c16-ml': 'yf-c16-ml',
    'yf-c16-intra': 'yf-c16-intra',
    'yf-c16-oe': 'yf-c16-oe',
}

# Signal source mapping from scanner model_tag
SIGNAL_SOURCE_MAP = {
    'c16': 'CS-5TF',
    'c16-dw': 'CS-DW',
    'c16-ml': 'surfer_ml',
    'c16-intra': 'intraday',
    'c16-oe': 'oe_sig5',
    'c14a': 'CS-5TF',
    'c14a-dw': 'CS-DW',
    'c14a-ml': 'surfer_ml',
    'c14a-intra': 'intraday',
}


def _normalize_timestamp(ts: str) -> str:
    """Convert a timestamp to US/Eastern-aware ISO 8601."""
    if not ts:
        return ts
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=ET)
        else:
            dt = dt.astimezone(ET)
        return dt.isoformat()
    except (ValueError, TypeError):
        return ts


def _extract_model_tag(filename: str) -> str:
    """Extract model_tag from filename like 'surfer_state_c16-ml.json'."""
    name = Path(filename).stem  # 'surfer_state_c16-ml'
    prefix = 'surfer_state_'
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def _signal_type_from_source(signal_source: str) -> str:
    """Map signal_source to signal_type for DB."""
    return {
        'CS-5TF': 'cs',
        'CS-DW': 'cs',
        'surfer_ml': 'ml_breakout',
        'intraday': 'intraday',
        'oe_sig5': 'oe_bounce',
    }.get(signal_source, 'unknown')


def run_migration(db: TradeDB, state_dir: str = '~/.x14') -> bool:
    """Run JSON → SQLite migration if not already done.

    Returns True if migration ran, False if already migrated.
    Raises on migration failure (corrupt file, insert error).
    """
    # Check if already migrated
    if db.get_metadata('migration_done') == '1':
        logger.info("Migration already done — skipping")
        return False

    state_dir = os.path.expanduser(state_dir)
    state_files = sorted(Path(state_dir).glob('surfer_state_*.json'))

    if not state_files:
        logger.info("No state files found — marking migration done (empty)")
        db.set_metadata('migration_done', '1')
        return True

    logger.info(f"Migrating {len(state_files)} state files from {state_dir}")

    # Atomic migration: single transaction for all files
    with db._lock:
        db._conn.execute("BEGIN IMMEDIATE")
        try:
            for state_file in state_files:
                _migrate_file(db, state_file)

            # Mark done inside the same transaction
            db._conn.execute("""
                INSERT INTO metadata (key, value) VALUES ('migration_done', '1')
                ON CONFLICT(key) DO UPDATE SET value = '1'
            """)

            db._conn.commit()
            logger.info(f"Migration complete — {len(state_files)} files imported")
            return True

        except Exception:
            db._conn.rollback()
            logger.exception("Migration FAILED — rolled back")
            raise


def _migrate_file(db: TradeDB, state_file: Path):
    """Migrate a single state file (within existing transaction)."""
    model_tag = _extract_model_tag(state_file.name)
    algo_id = MODEL_TAG_TO_ALGO.get(model_tag, model_tag)

    # ALL legacy JSON files are hypothetical — source='yf'
    source = 'yf'

    logger.info(f"Migrating {state_file.name} → algo_id={algo_id}, source={source}")

    try:
        data = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError) as e:
        raise RuntimeError(
            f"Cannot parse state file {state_file}: {e}. "
            f"HALTING migration — fix or remove the file."
        ) from e

    signal_source = SIGNAL_SOURCE_MAP.get(model_tag, model_tag)
    signal_type = _signal_type_from_source(signal_source)

    # 1. Import closed trades
    closed_trades = data.get('closed_trades', [])
    for i, ct in enumerate(closed_trades):
        entry_time = _normalize_timestamp(ct.get('entry_time', ''))
        exit_time = _normalize_timestamp(ct.get('exit_time', ''))
        legacy_id = f"{algo_id}:{ct.get('pos_id', f'closed_{i}')}"

        db._conn.execute("""
            INSERT OR IGNORE INTO trades (
                source, algo_id, symbol, direction, entry_time,
                entry_price, exit_time, exit_price, shares,
                stop_price, tp_price, confidence, signal_type,
                exit_reason, pnl, pnl_pct, legacy_pos_id,
                ib_fill_status, filled_shares, open_shares
            ) VALUES (?, ?, 'TSLA', ?, ?,
                      ?, ?, ?, ?,
                      0, 0, 0, ?,
                      ?, ?, ?, ?,
                      'filled', ?, 0)
        """, (
            source, algo_id, ct.get('direction', 'long'), entry_time,
            ct.get('entry_price', 0), exit_time, ct.get('exit_price', 0),
            ct.get('shares', 0),
            signal_type,
            ct.get('exit_reason', 'unknown'),
            ct.get('pnl', 0), ct.get('pnl_pct', 0),
            legacy_id,
            ct.get('shares', 0),
        ))

    # 2. Import open positions
    positions = data.get('positions', {})
    for pos_id, pos in positions.items():
        entry_time = _normalize_timestamp(pos.get('entry_time', ''))
        entry_price = pos.get('entry_price', 0)
        legacy_id = f"{algo_id}:{pos_id}"

        db._conn.execute("""
            INSERT OR IGNORE INTO trades (
                source, algo_id, symbol, direction, entry_time,
                entry_price, shares, stop_price, tp_price,
                confidence, signal_type, best_price, worst_price,
                trail_width, hold_bars, breakeven_applied,
                ou_half_life, el_flagged, trail_width_mult,
                legacy_pos_id, ib_fill_status, filled_shares, open_shares
            ) VALUES (?, ?, 'TSLA', ?, ?,
                      ?, ?, ?, ?,
                      ?, ?, ?, ?,
                      ?, ?, ?,
                      ?, ?, ?,
                      ?, 'filled', ?, ?)
        """, (
            source, algo_id, pos.get('direction', 'long'), entry_time,
            entry_price, pos.get('shares', 0),
            pos.get('stop_price', entry_price * 0.98),
            pos.get('tp_price', entry_price * 1.05),
            pos.get('confidence', 0.5),
            pos.get('signal_source', signal_type),
            pos.get('best_price', entry_price),
            pos.get('worst_price', entry_price),
            pos.get('initial_stop_pct', 0.02),  # trail_width
            0,  # hold_bars
            1 if pos.get('breakeven_applied', False) else 0,
            pos.get('ou_half_life', 5.0),
            1 if pos.get('el_flagged', False) else 0,
            pos.get('trail_width_mult', 1.0),
            legacy_id,
            pos.get('shares', 0),
            pos.get('shares', 0),
        ))

    # 3. Import daily counters
    daily_date = data.get('daily_date')
    daily_pnl = data.get('daily_pnl', 0)
    daily_trade_count = data.get('daily_trade_count', 0)
    if daily_date and daily_trade_count > 0:
        db._conn.execute("""
            INSERT OR IGNORE INTO daily_pnl (date, source, algo_id, pnl, trades, wins)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (daily_date, source, algo_id, daily_pnl, daily_trade_count))

    # 4. Import AH counters
    ext_opens = data.get('ext_opens_today', 0)
    ext_closes = data.get('ext_closes_today', 0)
    ext_wins = data.get('ext_wins_today', 0)
    if daily_date:
        for key, val in [
            (f'{algo_id}:ext_opens_today:{daily_date}', ext_opens),
            (f'{algo_id}:ext_closes_today:{daily_date}', ext_closes),
            (f'{algo_id}:ext_wins_today:{daily_date}', ext_wins),
        ]:
            if val > 0:
                db._conn.execute("""
                    INSERT INTO metadata (key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """, (key, str(val)))

    # 5. Import signal history (last 200)
    signal_history = data.get('signal_history', [])
    for sig in signal_history[-200:]:
        sig_time = _normalize_timestamp(sig.get('time', ''))
        if not sig_time:
            continue
        db._conn.execute("""
            INSERT INTO signals (time, source, algo_id, action, confidence)
            VALUES (?, ?, ?, ?, ?)
        """, (
            sig_time, source, algo_id,
            sig.get('action', 'HOLD'),
            sig.get('confidence'),
        ))
