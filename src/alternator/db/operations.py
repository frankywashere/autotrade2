import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import sqlite3

from .schema import PREDICTIONS_DB, TRADES_DB


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def log_prediction(
    db_dir: Path,
    *,
    timestamp: datetime,
    symbol: str,
    timeframe: str,
    predicted_high: float,
    predicted_low: float,
    confidence: float,
    model_name: str,
    extra: Dict[str, Any] | None = None,
) -> int:
    db_path = db_dir / PREDICTIONS_DB
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions (
                timestamp, symbol, timeframe,
                predicted_high, predicted_low, confidence,
                model_name, extra
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp.isoformat(),
                symbol,
                timeframe,
                predicted_high,
                predicted_low,
                confidence,
                model_name,
                json.dumps(extra or {}),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def log_high_confidence_trade(
    db_dir: Path,
    *,
    timestamp: datetime,
    symbol: str,
    direction: str,
    timeframe: str,
    entry_price: float,
    target_price: float,
    stop_price: float,
    max_hold_time_minutes: int,
    confidence: float,
    expected_return_pct: float,
    rationale: Dict[str, Any] | None = None,
) -> int:
    db_path = db_dir / TRADES_DB
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO high_confidence_trades (
                timestamp, symbol, direction, timeframe,
                entry_price, target_price, stop_price,
                max_hold_time_minutes,
                confidence, expected_return_pct, rationale
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp.isoformat(),
                symbol,
                direction,
                timeframe,
                entry_price,
                target_price,
                stop_price,
                max_hold_time_minutes,
                confidence,
                expected_return_pct,
                json.dumps(rationale or {}),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def fetch_recent_predictions(db_dir: Path, limit: int = 50) -> List[Dict[str, Any]]:
    db_path = db_dir / PREDICTIONS_DB
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, symbol, timeframe,
                   predicted_high, predicted_low, confidence,
                   model_name, extra
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    results = []
    for row in rows:
        (
            pid,
            ts,
            symbol,
            timeframe,
            ph,
            pl,
            conf,
            model_name,
            extra,
        ) = row
        results.append(
            {
                "id": pid,
                "timestamp": ts,
                "symbol": symbol,
                "timeframe": timeframe,
                "predicted_high": ph,
                "predicted_low": pl,
                "confidence": conf,
                "model_name": model_name,
                "extra": json.loads(extra or "{}"),
            }
        )
    return results


def fetch_recent_high_confidence_trades(
    db_dir: Path, limit: int = 50
) -> List[Dict[str, Any]]:
    db_path = db_dir / TRADES_DB
    with _connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, timestamp, symbol, direction, timeframe,
                   entry_price, target_price, stop_price,
                   max_hold_time_minutes, confidence,
                   expected_return_pct, rationale, status
            FROM high_confidence_trades
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    results = []
    for row in rows:
        (
            tid,
            ts,
            symbol,
            direction,
            timeframe,
            entry,
            target,
            stop,
            max_hold,
            conf,
            expected_ret,
            rationale,
            status,
        ) = row
        results.append(
            {
                "id": tid,
                "timestamp": ts,
                "symbol": symbol,
                "direction": direction,
                "timeframe": timeframe,
                "entry_price": entry,
                "target_price": target,
                "stop_price": stop,
                "max_hold_time_minutes": max_hold,
                "confidence": conf,
                "expected_return_pct": expected_ret,
                "rationale": json.loads(rationale or "{}"),
                "status": status,
            }
        )
    return results


