"""plots.py — per-experiment and comparison figures.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cfg = result["cfg"]
    history = result["history"]
    epochs = history.get("epoch", [])
    if not epochs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f"Experiment {cfg['exp_id']} | optimizer={cfg['optimizer']} | "
        f"lr={cfg['lr']} | batch={cfg['batch']} | dropout={cfg['dropout']} | seed={cfg['seed']}",
        fontsize=12,
    )
    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_loss"], label="val loss")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, history["val_acc"], label="val accuracy")
    axes[1].plot(epochs, history["val_macro_f1"], label="val macro-F1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score")
    axes[1].legend()
    axes[2].plot(epochs, history["grad_norm"], label="grad norm", color="tab:purple")
    axes[2].set(title="Gradient norm before clipping", xlabel="Epoch", ylabel="L2 norm")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc (và nên có val_macro_f1) theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    Yêu cầu: tiêu đề ghi exp_id và cấu hình chính (optimizer, lr, batch, ...), có nhãn trục và chú thích.
    Các bước: fig, axes = plt.subplots(1, 3, figsize=...); plot; set_title/xlabel/legend;
              fig.savefig(path, dpi=..., bbox_inches="tight"); plt.close(fig)
    Gợi ý: đánh dấu best_epoch bằng đường thẳng đứng.
    """

def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    for result in results:
        exp_id = result["cfg"]["exp_id"]
        values = result.get("history", {}).get(metric)
        if values is None:
            continue
        ax.plot(result["history"]["epoch"], values, label=exp_id)
    ax.legend(frameon=False)
    ax.set_title(title or f"Comparison: {metric}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    """Vẽ chồng một chỉ số (ví dụ "val_loss", "val_macro_f1", "grad_norm") của nhiều thí nghiệm
    trên cùng một trục, mỗi thí nghiệm một đường, chú thích bằng exp_id.

    Dùng cho ảnh figures/compare_<nhóm>.png (ví dụ compare_optimizer.png).
    """
