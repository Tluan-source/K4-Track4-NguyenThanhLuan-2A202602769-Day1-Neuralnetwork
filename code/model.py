from __future__ import annotations

import torch
import torch.nn as nn

EXPECTED_PARAMS = {
    (256, 128): 47_879,
    (512, 256): 161_287,
    (256, 128, 64): 55_687,
}


class MLP(nn.Module):
    """Simple MLP matching the assignment specification."""

    def __init__(self, hidden=(256, 128), dropout: float = 0.0, init: str = "he",
                 in_features: int = 54, num_classes: int = 7):
        super().__init__()
        hidden = tuple(hidden)
        layers = []
        current_in = in_features
        for h in hidden:
            layers.append(nn.Linear(current_in, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_in = h
        layers.append(nn.Linear(current_in, num_classes))
        self.net = nn.Sequential(*layers)
        self.hidden = hidden
        self.dropout = dropout
        self.init = init
        init_weights(self, init)
        needed = EXPECTED_PARAMS.get(hidden)
        if needed is not None:
            assert count_params(self) == needed, (count_params(self), needed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_weights(model: nn.Module, init: str) -> None:
    """Initialize linear layer weights according to the requested scheme."""
    for m in model.modules():
        if not isinstance(m, nn.Linear):
            continue
        if init == "zeros":
            nn.init.zeros_(m.weight)
        elif init == "normal":
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
        elif init == "xavier":
            nn.init.xavier_normal_(m.weight)
        elif init == "he":
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        elif init == "default":
            pass
        else:
            raise ValueError(f"Unknown init={init!r}")
        nn.init.zeros_(m.bias)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> list[float]:
    """Return the per-layer activation standard deviation for diagnostics."""
    model.eval()
    h = x
    stats = []
    for module in model.net:
        if isinstance(module, nn.Linear):
            h = module(h)
            stats.append(float(h.std(unbiased=False).item()))
        elif isinstance(module, (nn.ReLU, nn.Dropout)):
            h = module(h)
    return stats
