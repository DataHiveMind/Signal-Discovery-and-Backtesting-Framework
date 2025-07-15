import torch
import torch.nn as nn


class LSTMCNNModel(nn.Module):
    """
    Hybrid LSTM + CNN for sequence regression/classification.
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, kernel_size=3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=False)
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # transpose for conv: (batch, hidden_dim, seq_len)
        conv_in = lstm_out.transpose(1, 2)
        conv_out = self.conv(conv_in)
        pooled = torch.mean(conv_out, dim=2)
        return self.fc(pooled)
