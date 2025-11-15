from pathlib import Path
import sqlite3


PREDICTIONS_DB = "predictions.db"
TRADES_DB = "high_confidence_trades.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


def init_databases(db_dir: Path) -> None:
    predictions_path = db_dir / PREDICTIONS_DB
    trades_path = db_dir / TRADES_DB

    with _connect(predictions_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                predicted_high REAL,
                predicted_low REAL,
                confidence REAL,
                model_name TEXT,
                extra JSON
            )
            """
        )
        conn.commit()

    with _connect(trades_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS high_confidence_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                entry_price REAL,
                target_price REAL,
                stop_price REAL,
                max_hold_time_minutes INTEGER,
                confidence REAL,
                expected_return_pct REAL,
                rationale TEXT,
                status TEXT DEFAULT 'pending'
            )
            """
        )
        conn.commit()


