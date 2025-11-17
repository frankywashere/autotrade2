from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from alternator.config import AppConfig
from alternator.data import TimeSeriesDataset
from alternator.model import HierarchicalTimeSeriesModel


def train_model(
    app_config: AppConfig,
    csv_path: Path,
    model_out_path: Optional[Path] = None,
    device: str = "cpu",
) -> Path:
    sequence_length = app_config.training.sequence_length
    batch_size = app_config.training.batch_size
    learning_rate = app_config.training.learning_rate
    epochs = app_config.training.epochs

    dataset = TimeSeriesDataset(csv_path, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = HierarchicalTimeSeriesModel()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            preds = outputs[:, :2]
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / max(1, len(dataset))
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

    model_out_path = model_out_path or (
        app_config.paths.models_dir / "hierarchical_model.pth"
    )
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out_path)
    print(f"Saved model to {model_out_path}")
    return model_out_path


