import torch.nn as nn


class MLPSingleRegressor(nn.Module):
    """Arquiteturas exportadas no Java; os pesos são treinados do zero."""
    def __init__(self, target_idx):
        super().__init__()
        if target_idx not in (0, 1):
            raise ValueError("target_idx deve ser 0 (altura) ou 1 (biomassa).")
        hidden = (3, 50) if target_idx == 0 else (2, 250)
        self.network = nn.Sequential(
            nn.Linear(2, hidden[0]), nn.Tanh(),
            nn.Linear(hidden[0], hidden[1]), nn.Tanh(),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)
