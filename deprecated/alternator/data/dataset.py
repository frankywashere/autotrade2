from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Simple time-series dataset for OHLCV-based features.
    """

    def __init__(self, csv_path: Path, sequence_length: int):
        self.sequence_length = sequence_length
        self.df = pd.read_csv(csv_path)
        self.features = self.df[["Open", "High", "Low", "Close", "Volume"]].values
        self.targets = self.df[["High", "Low"]].values

    def __len__(self) -> int:
        return max(0, len(self.df) - self.sequence_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[idx : idx + self.sequence_length]
        y = self.targets[idx + self.sequence_length - 1]
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor


