from __future__ import annotations

import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import EXPECTED_PARAMS, MLP, count_params
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",
    optimizer="sgd_momentum",
    lr=None,
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,
    precision="fp32",
    seed=1,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    preds = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size]
        logits = model(xb)
        preds.append(logits.argmax(dim=1))
    if not preds:
        return torch.empty((0,), dtype=torch.long, device=X.device)
    return torch.cat(preds)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds = []
    for start in range(0, X.shape[0], batch_size):
        xb = X[start:start + batch_size]
        yb = y[start:start + batch_size]
        logits = model(xb)
        loss = compute_loss(logits, yb, loss_name)
        total_loss += float(loss.item()) * yb.numel()
        total_count += yb.numel()
        all_preds.append(logits.argmax(dim=1))
    pred = torch.cat(all_preds) if all_preds else torch.empty((0,), dtype=torch.long, device=X.device)
    true = y.to(pred.device)
    acc = float((pred == true).float().mean().item()) if pred.numel() > 0 else 0.0
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (true.cpu().numpy(), pred.cpu().numpy()), 1)
    return {"loss": total_loss / max(total_count, 1), "acc": acc, "macro_f1": macro_f1_from_confusion(cm)}


def compute_loss(logits, y, loss_name: str):
    loss_name = loss_name.lower()
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        y_onehot = F.one_hot(y, num_classes=logits.size(-1)).float()
        return ((logits - y_onehot) ** 2).mean()
    raise ValueError(f"Unsupported loss name: {loss_name!r}")


def run_experiment(cfg: dict, data: dict) -> dict:
    set_seed(cfg["seed"])
    hidden = tuple(cfg["hidden"])
    if hidden not in EXPECTED_PARAMS:
        raise ValueError(f"Unsupported hidden size {hidden!r}; must be one of {tuple(EXPECTED_PARAMS.keys())}")
    device = data["X_tr"].device
    model = MLP(hidden=hidden, dropout=float(cfg["dropout"]), init=cfg["init"], in_features=data["X_tr"].shape[1], num_classes=7)
    model.to(device)
    assert count_params(model) == EXPECTED_PARAMS[hidden], (count_params(model), EXPECTED_PARAMS[hidden])

    optimizer = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], momentum=cfg["momentum"])
    scaler = None
    if cfg["precision"] == "fp16" and device.type == "cuda":
        scaler = torch.amp.GradScaler(device=device.type)

    step0 = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"])
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "grad_norm": [], "epoch_time_s": []}
    diverged = False

    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_start = time.time()
        model.train()
        grad_norm_vals = []
        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], int(cfg["batch"]), shuffle=True):
            if cfg["precision"] == "fp16" and scaler is not None:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, cfg["loss"])
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if cfg["clip_norm"] is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, cfg["loss"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if cfg["clip_norm"] is not None:
                    gn = clip_gradients(model.parameters(), cfg["clip_norm"])
                else:
                    gn = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            gn += float(p.grad.detach().norm(2).item() ** 2)
                    gn = float(np.sqrt(gn))
                optimizer.step()
            grad_norm_vals.append(float(gn))
            if not torch.isfinite(loss):
                diverged = True
                break
        if diverged:
            break

        train_metrics = evaluate(model, data["X_tr"], data["y_tr"], loss_name=cfg["loss"], batch_size=max(4096, int(cfg["batch"])))
        val_metrics = evaluate(model, data["X_val"], data["y_val"], loss_name=cfg["loss"], batch_size=max(4096, int(cfg["batch"])))
        epoch_time = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(float(np.mean(grad_norm_vals)) if grad_norm_vals else 0.0)
        history["epoch_time_s"].append(epoch_time)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = epoch
            best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}

    final_train_loss = history["train_loss"][-1] if history["train_loss"] else step0["loss"]
    final_val_loss = history["val_loss"][-1] if history["val_loss"] else step0["loss"]
    final_val_acc = history["val_acc"][-1] if history["val_acc"] else step0["acc"]
    final_val_macro_f1 = history["val_macro_f1"][-1] if history["val_macro_f1"] else step0["macro_f1"]
    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

    summary = {
        "step0_loss": float(step0["loss"]),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "val_acc": float(final_val_acc),
        "val_macro_f1": float(final_val_macro_f1),
        "time_per_epoch_s": float(np.mean(history["epoch_time_s"])) if history["epoch_time_s"] else 0.0,
        "peak_mem_MB": float(peak_mem_mb),
        "diverged": bool(diverged),
    }
    return {"cfg": cfg, "history": history, "summary": summary, "best_state": best_state}


def write_predictions(row_id, preds, path: str) -> None:
    import pandas as pd
    row_id = np.asarray(row_id, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    out = pd.DataFrame({"row_id": row_id, "pred": preds})
    out.to_csv(path, index=False)


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    hidden = tuple(cfg["hidden"])
    model = MLP(hidden=hidden, dropout=float(cfg["dropout"]), init=cfg["init"], in_features=data["X_eval"].shape[1], num_classes=7)
    model.load_state_dict(result["best_state"])
    model.to(data["X_eval"].device)
    model.eval()
    preds = predict(model, data["X_eval"], batch_size=8192)
    write_predictions(data["eval_row_id"], preds.cpu().numpy(), pred_path)
