from __future__ import annotations
import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(params, name: str, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    name = name.lower()
    if name not in OPTIMIZERS:
        raise ValueError(f"Optimizer {name} không hợp lệ. Chỉ chấp nhận {OPTIMIZERS}.")
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    """Trả về một torch.optim.Optimizer.

    Các bước:
      1. kiểm tra name nằm trong OPTIMIZERS, nếu không raise ValueError
      2. "sgd"          -> torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
         "sgd_momentum" -> torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
         "adam"         -> torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
         "adamw"        -> torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    Chú ý: weight_decay của Adam (L2 trộn vào gradient) khác weight_decay của AdamW (suy giảm tách riêng).
    """


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    if name is None:
        return None
    if name.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), **kwargs)
    raise ValueError(f"Unknown scheduler: {name!r}")
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, ví dụ cosine (slide có ví dụ CosineAnnealingLR).

    Trả về None nếu name là None. Nếu bạn dùng scheduler ở một thí nghiệm, hãy ghi vào bảng (cột notes).
    """


def clip_gradients(params, max_norm: float | None) -> float:
    if max_norm is None:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float("inf"))
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)

    return float(total_norm)
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt.

    Các bước:
      1. nếu max_norm là None: tính chuẩn toàn cục mà không cắt (ví dụ clip_grad_norm_ với max_norm=inf)
      2. ngược lại: total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
      3. return float(total_norm)
    Giá trị trả về chính là `grad_norm` bạn phải ghi lại ở mỗi bước (để thấy "gai" gradient).
    Khi dùng mixed precision FP16 + GradScaler: phải scaler.unscale_(optimizer) TRƯỚC khi gọi hàm này.
    """
