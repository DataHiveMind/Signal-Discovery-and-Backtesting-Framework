import torch
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from .data_loader import TimeSeriesDataset
from .model_def import LSTMCNNModel
from torch.utils.tensorboard import SummaryWriter


def train(config: dict) -> str:
    """
    Train the GPU-accelerated model and save best checkpoint.

    Args:
        config: Dictionary with keys:
          - features: torch.Tensor of shape (T, D)
          - window: int
          - batch_size: int
          - epochs: int
          - lr: float
          - device: 'cuda' or 'cpu'

    Returns:
        Path to best model checkpoint.
    """
    device = torch.device(config['device'])
    dataset = TimeSeriesDataset(config['features'], config['window'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = LSTMCNNModel(
        input_dim=config['features'].shape[1],
        hidden_dim=config.get('hidden_dim', 64)
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['lr'],
        steps_per_epoch=len(loader),
        epochs=config['epochs']
    )
    scaler = GradScaler()
    writer = SummaryWriter(log_dir=config.get('log_dir', 'runs/experiment'))

    best_loss = float('inf')
    best_path = 'best_model.pt'

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            with autocast():
                preds = model(x)
                loss = torch.nn.functional.mse_loss(preds, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        writer.add_scalar('train/loss', avg_loss, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_path)

    writer.close()
    return best_path
