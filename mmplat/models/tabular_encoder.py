import torch
import torch.nn as nn

def mlp(in_dim: int, hidden_dims, out_dim: int, dropout: float = 0.0):
    dims = [in_dim] + list(hidden_dims) + [out_dim]
    layers = []
    for i in range(len(dims)-2):
        layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
    layers += [nn.Linear(dims[-2], dims[-1])]
    return nn.Sequential(*layers)

class TabularEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=(64,64), out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = mlp(in_dim, hidden_dims, out_dim, dropout)

    def forward(self, x):
        return self.net(x)
