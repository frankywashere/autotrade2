from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from alternator.alerts import TelegramClient
from alternator.config import AppConfig
from alternator.data import TimeSeriesDataset
from alternator.db import log_prediction, log_high_confidence_trade
from alternator.model import HierarchicalTimeSeriesModel


def backtest_model(
    app_config: AppConfig,
    csv_path: Path,
    model_path: Path,
    device: str = "cpu",
    model_name: str = "hierarchical",
) -> None:
    sequence_length = app_config.training.sequence_length

    dataset = TimeSeriesDataset(csv_path, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = HierarchicalTimeSeriesModel()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    confidence_threshold = app_config.alerts.confidence_threshold
    telegram = TelegramClient(
        bot_token=app_config.telegram.bot_token,
        chat_id=app_config.telegram.chat_id,
    )

    for idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        preds = model.predict(x_batch)
        high = preds["high"].item()
        low = preds["low"].item()
        confidence = preds["confidence"].item()

        timestamp = datetime.utcnow()

        log_prediction(
            app_config.paths.db_dir,
            timestamp=timestamp,
            symbol=app_config.training.symbol,
            timeframe="1min",
            predicted_high=high,
            predicted_low=low,
            confidence=confidence,
            model_name=model_name,
            extra={},
        )

        if confidence >= confidence_threshold:
            current_price = float(y_batch[0][0].item())
            direction = "long" if high > current_price else "short"
            expected_return_pct = (
                (high - current_price) / current_price * 100.0
                if direction == "long"
                else (current_price - low) / current_price * 100.0
            )
            log_high_confidence_trade(
                app_config.paths.db_dir,
                timestamp=timestamp,
                symbol=app_config.training.symbol,
                direction=direction,
                timeframe="1min",
                entry_price=current_price,
                target_price=high if direction == "long" else low,
                stop_price=low if direction == "long" else high,
                max_hold_time_minutes=120,
                confidence=confidence,
                expected_return_pct=expected_return_pct,
                rationale={"source": "backtest"},
            )
            telegram.send_trade_alert(
                {
                    "symbol": app_config.training.symbol,
                    "direction": direction,
                    "timeframe": "1min",
                    "entry_price": current_price,
                    "target_price": high if direction == "long" else low,
                    "stop_price": low if direction == "long" else high,
                    "confidence": confidence,
                    "expected_return_pct": expected_return_pct,
                }
            )


