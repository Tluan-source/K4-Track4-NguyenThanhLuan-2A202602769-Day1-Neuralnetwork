from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_run(result: dict, path: str) -> None:
    """Create a 3-panel plot for one experiment."""
    cfg = result["cfg"]
    history = result["history"]
    epochs = history["epoch"]
    if not epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax1, ax2, ax3 = axes

    ax1.plot(epochs, history["train_loss"], label="train loss", color="tab:blue")
    ax1.plot(epochs, history["val_loss"], label="val loss", color="tab:orange")
    ax1.set_title("Loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend()

    ax2.plot(epochs, history["val_acc"], label="val acc", color="tab:green")
    if "val_macro_f1" in history and history["val_macro_f1"]:
        ax2.plot(epochs, history["val_macro_f1"], label="val macro-F1", color="tab:red", linestyle="--")
    ax2.set_title("Validation accuracy / macro F1")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("metric")
    ax2.legend()

    ax3.plot(epochs, history["grad_norm"], color="tab:purple")
    ax3.axhline(0, color="black", linewidth=0.5)
    ax3.set_title("Gradient norm")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("grad norm")

    best_epoch = result["summary"].get("best_epoch", epochs[-1])
    best_idx = max(0, min(best_epoch - 1, len(history["val_loss"]) - 1))
    ax1.scatter([best_epoch], [history["val_loss"][best_idx]], color="black", s=20, zorder=3)
    fig.suptitle(f"exp_id={cfg['exp_id']} | opt={cfg['optimizer']} | lr={cfg['lr']} | batch={cfg['batch']} | epochs={cfg['epochs']} | dropout={cfg['dropout']}")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Persist a result payload to a per-experiment JSON file."""
    exp_id = result["cfg"]["exp_id"]
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{exp_id}.json"
    payload = {"cfg": result["cfg"], "history": result["history"], "summary": result["summary"]}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return str(file_path)


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    cfg = result["cfg"]
    summary = result["summary"]
    row = {
        "exp_id": cfg["exp_id"],
        "group": cfg["group"],
        "description": cfg["description"],
        "loss": cfg["loss"],
        "optimizer": cfg["optimizer"],
        "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"],
        "batch": cfg["batch"],
        "epochs": cfg["epochs"],
        "hidden": str(cfg["hidden"]),
        "dropout": cfg["dropout"],
        "clip_norm": cfg["clip_norm"],
        "precision": cfg["precision"],
        "init": cfg["init"],
        "seed": cfg["seed"],
        "step0_loss": summary.get("step0_loss"),
        "best_val_loss": summary.get("best_val_loss"),
        "best_epoch": summary.get("best_epoch"),
        "final_train_loss": summary.get("final_train_loss"),
        "final_val_loss": summary.get("final_val_loss"),
        "val_acc": summary.get("val_acc"),
        "val_macro_f1": summary.get("val_macro_f1"),
        "time_per_epoch_s": summary.get("time_per_epoch_s"),
        "peak_mem_MB": summary.get("peak_mem_MB"),
        "diverged": summary.get("diverged", False),
        "eval_acc": None if eval_scores is None else eval_scores.get("accuracy"),
        "eval_macro_f1": None if eval_scores is None else eval_scores.get("macro_f1"),
        "figure_file": f"figures/{cfg['exp_id']}.png",
        "notes": notes,
    }
    return row
