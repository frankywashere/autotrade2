import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str


@dataclass
class PathsConfig:
    data_dir: Path
    models_dir: Path
    db_dir: Path


@dataclass
class TrainingConfig:
    symbol: str
    benchmark_symbol: str
    sequence_length: int
    batch_size: int
    learning_rate: float
    epochs: int


@dataclass
class BacktestConfig:
    test_start: str
    test_end: str


@dataclass
class AlertsConfig:
    confidence_threshold: float


@dataclass
class ServerConfig:
    host: str
    port: int


@dataclass
class AppConfig:
    paths: PathsConfig
    training: TrainingConfig
    backtest: BacktestConfig
    alerts: AlertsConfig
    server: ServerConfig
    telegram: TelegramConfig


def _load_yaml_config() -> Dict[str, Any]:
    cfg_path = PROJECT_ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file at {cfg_path}")
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def _load_telegram_config() -> TelegramConfig:
    api_path = PROJECT_ROOT / "config" / "api_keys.json"
    if not api_path.exists():
        raise FileNotFoundError(f"Missing Telegram api_keys.json at {api_path}")
    with api_path.open("r") as f:
        data = json.load(f)
    tg = data.get("telegram", {})
    return TelegramConfig(
        bot_token=tg.get("bot_token", ""),
        chat_id=str(tg.get("chat_id", "")),
    )


def load_app_config() -> AppConfig:
    raw = _load_yaml_config()
    paths_cfg = raw.get("paths", {})
    training_cfg = raw.get("training", {})
    backtest_cfg = raw.get("backtest", {})
    alerts_cfg = raw.get("alerts", {})
    server_cfg = raw.get("server", {})

    paths = PathsConfig(
        data_dir=PROJECT_ROOT / paths_cfg.get("data_dir", "data"),
        models_dir=PROJECT_ROOT / paths_cfg.get("models_dir", "models"),
        db_dir=PROJECT_ROOT / paths_cfg.get("db_dir", "db"),
    )

    training = TrainingConfig(
        symbol=training_cfg.get("symbol", "TSLA"),
        benchmark_symbol=training_cfg.get("benchmark_symbol", "SPY"),
        sequence_length=int(training_cfg.get("sequence_length", 256)),
        batch_size=int(training_cfg.get("batch_size", 64)),
        learning_rate=float(training_cfg.get("learning_rate", 0.001)),
        epochs=int(training_cfg.get("epochs", 10)),
    )

    backtest = BacktestConfig(
        test_start=backtest_cfg.get("test_start", "2023-01-01"),
        test_end=backtest_cfg.get("test_end", "2023-12-31"),
    )

    alerts = AlertsConfig(
        confidence_threshold=float(alerts_cfg.get("confidence_threshold", 0.8))
    )

    server = ServerConfig(
        host=server_cfg.get("host", "0.0.0.0"),
        port=int(server_cfg.get("port", 8000)),
    )

    telegram = _load_telegram_config()

    for p in [paths.data_dir, paths.models_dir, paths.db_dir]:
        os.makedirs(p, exist_ok=True)

    return AppConfig(
        paths=paths,
        training=training,
        backtest=backtest,
        alerts=alerts,
        server=server,
        telegram=telegram,
    )


