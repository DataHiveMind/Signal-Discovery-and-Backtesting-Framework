import torch
from gpu_models.data_loader import TimeSeriesDataset
from gpu_models.model_def import LSTMCNNModel


def test_dataset_and_model():
    # Create synthetic features: 50 timesteps, 4 features
    features = torch.randn(50, 4)
    window = 10
    ds = TimeSeriesDataset(features, window)
    x, y = ds[0]
    assert x.shape == (window, 4)
    assert isinstance(y.item(), float)

    # Instantiate and forward
    model = LSTMCNNModel(input_dim=4, hidden_dim=8)
    x_batch = torch.randn(2, window, 4)
    out = model(x_batch)
    assert out.shape == (2, 1)
