from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Build a torch optimizer from the requested name."""
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name!r}; expected one of {OPTIMIZERS}")


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """Optional LR scheduler placeholder."""
    if name is None:
        return None
    if name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), **kwargs)
    return None


def clip_gradients(params, max_norm: float | None) -> float:
    """Clip gradients by L2 norm and return the pre-clip total norm."""
    if max_norm is None:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
        return float(total_norm)
    total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
    return float(total_norm)
