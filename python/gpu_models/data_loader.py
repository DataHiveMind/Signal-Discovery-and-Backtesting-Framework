import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    Dataset for sliding-window time-series feature arrays.
    """
    def __init__(self, features: torch.Tensor, window: int):
        self.features = features
        self.window = window

    def __len__(self):
        return self.features.size(0) - self.window

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.window]
        y = self.features[idx + self.window, 0]  # example: next-step target
        return x, y
