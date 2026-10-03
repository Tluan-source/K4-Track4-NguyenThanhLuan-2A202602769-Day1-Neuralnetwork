from __future__ import annotations

import csv
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients


DEFAULT_CFG = dict(
    exp_id="base-s1",
    group="baseline",
    description="Baseline M-base",
    loss="ce",
    optimizer="sgd_momentum",
    lr=None,
    weight_decay=0.0,
    momentum=0.9,
    batch=512,
    epochs=20,
    hidden=(256, 128),
    dropout=0.0,
    init="he",
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
    cm = np.asarray(cm, dtype=np.float64)
    f1s = []

    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        p = tp / (tp + fp) if tp + fp > 0 else 0.0
        r = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0

        f1s.append(f1)

    return float(np.mean(f1s))


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    model.eval()
    preds = []

    for i in range(0, len(X), batch_size):
        logits = model(X[i:i + batch_size])
        preds.append(logits.argmax(dim=1))

    if not preds:
        return torch.empty(0, dtype=torch.int64, device=X.device)

    return torch.cat(preds)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    model.eval()

    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset")

    total_loss = 0.0
    all_preds = []

    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]

        logits = model(xb)

        if loss_name == "ce":
            loss = F.cross_entropy(logits, yb, reduction="sum")
        elif loss_name == "mse":
            target = F.one_hot(
                yb,
                num_classes=logits.shape[1]
            ).to(logits.dtype)

            loss = F.mse_loss(
                logits,
                target,
                reduction="sum"
            ) / logits.shape[1]
        else:
            raise ValueError("loss must be 'ce' or 'mse'")

        total_loss += loss.item()
        all_preds.append(logits.argmax(dim=1))

    preds = torch.cat(all_preds)

    acc = (preds == y).float().mean().item()

    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(
        cm,
        (
            y.cpu().numpy(),
            preds.cpu().numpy()
        ),
        1
    )

    return {
        "loss": total_loss / n,
        "acc": acc,
        "macro_f1": macro_f1_from_confusion(cm),
    }


def compute_loss(logits, y, loss_name: str):
    if loss_name == "ce":
        return F.cross_entropy(logits, y)

    if loss_name == "mse":
        target = F.one_hot(
            y,
            num_classes=logits.shape[1]
        ).to(logits.dtype)

        return F.mse_loss(logits, target)

    raise ValueError("loss must be 'ce' or 'mse'")


def run_experiment(cfg: dict, data: dict) -> dict:
    cfg = dict(cfg)

    if cfg["lr"] is None:
        raise ValueError("cfg['lr'] cannot be None")

    set_seed(cfg["seed"])

    X_tr = data["X_tr"]
    y_tr = data["y_tr"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    device = X_tr.device
    hidden = tuple(cfg["hidden"])

    model = MLP(
        hidden=hidden,
        dropout=cfg["dropout"],
        init=cfg["init"],
        in_features=X_tr.shape[1],
        num_classes=7,
    )

    assert count_params(model) == EXPECTED_PARAMS[hidden]

    model = model.to(device)

    optimizer = build_optimizer(
        model.parameters(),
        name=cfg["optimizer"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        momentum=cfg["momentum"],
    )

    precision = cfg["precision"]

    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Invalid precision")

    use_fp16 = precision == "fp16"

    if use_fp16 and device.type != "cuda":
        raise ValueError("fp16 requires CUDA")

    scaler = (
        torch.amp.GradScaler("cuda")
        if use_fp16
        else None
    )

    if precision == "fp16":
        autocast_dtype = torch.float16
    elif precision == "bf16":
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = None

    amp_enabled = (
        precision != "fp32"
        and device.type == "cuda"
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])

    step0 = evaluate(
        model,
        X_val,
        y_val,
        cfg["loss"]
    )

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    diverged = False

    for epoch in range(1, cfg["epochs"] + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        model.train()
        grad_norms = []

        for xb, yb in iterate_batches(
            X_tr,
            y_tr,
            cfg["batch"],
            generator
        ):
            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype
                ):
                    logits = model(xb)
                    loss = compute_loss(
                        logits,
                        yb,
                        cfg["loss"]
                    )
            else:
                logits = model(xb)
                loss = compute_loss(
                    logits,
                    yb,
                    cfg["loss"]
                )

            if not torch.isfinite(loss):
                diverged = True
                break

            if use_fp16:
                scaler.scale(loss).backward()

                if cfg["clip_norm"] is not None:
                    scaler.unscale_(optimizer)
            else:
                loss.backward()

            gn = clip_gradients(
                model.parameters(),
                cfg["clip_norm"]
            )

            gn = (
                gn.item()
                if torch.is_tensor(gn)
                else float(gn)
            )

            grad_norms.append(gn)

            if not np.isfinite(gn):
                diverged = True
                break

            if use_fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        if diverged:
            break

        if device.type == "cuda":
            torch.cuda.synchronize()

        train_metrics = evaluate(
            model,
            X_tr,
            y_tr,
            cfg["loss"]
        )

        val_metrics = evaluate(
            model,
            X_val,
            y_val,
            cfg["loss"]
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - start
        mean_gn = (
            float(np.mean(grad_norms))
            if grad_norms
            else 0.0
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["grad_norm"].append(mean_gn)
        history["epoch_time_s"].append(epoch_time)

        if (
            not np.isfinite(train_metrics["loss"])
            or not np.isfinite(val_metrics["loss"])
        ):
            diverged = True
            break

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_val_acc = val_metrics["acc"]
            best_val_macro_f1 = val_metrics["macro_f1"]

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if history["epoch"]:
        final_train_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        time_per_epoch_s = float(
            np.mean(history["epoch_time_s"])
        )
    else:
        final_train_loss = float("nan")
        final_val_loss = float("nan")
        time_per_epoch_s = float("nan")

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
        }

        best_val_loss = step0["loss"]
        best_val_acc = step0["acc"]
        best_val_macro_f1 = step0["macro_f1"]

    peak_mem_MB = (
        torch.cuda.max_memory_allocated(device)
        / 1024 ** 2
        if device.type == "cuda"
        else 0.0
    )

    summary = {
        "step0_loss": float(step0["loss"]),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_train_loss),
        "final_val_loss": float(final_val_loss),
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(time_per_epoch_s),
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": bool(diverged),
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id, preds, path: str) -> None:
    row_id = np.asarray(row_id).reshape(-1)
    preds = np.asarray(preds).reshape(-1)

    if len(row_id) != len(preds):
        raise ValueError("row_id and preds length mismatch")

    if len(np.unique(row_id)) != len(row_id):
        raise ValueError("Duplicate row_id")

    if not np.all((preds >= 0) & (preds <= 6)):
        raise ValueError("Predictions must be 0..6")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row_id", "pred"])

        for rid, pred in zip(row_id, preds):
            writer.writerow([rid, int(pred)])


def final_eval(
    cfg: dict,
    result: dict,
    data: dict,
    pred_path: str,
) -> None:
    X_eval = data["X_eval"]

    model = MLP(
        hidden=tuple(cfg["hidden"]),
        dropout=cfg["dropout"],
        init=cfg["init"],
        in_features=X_eval.shape[1],
        num_classes=7,
    )

    model.load_state_dict(result["best_state"])
    model = model.to(X_eval.device)

    preds = predict(model, X_eval)

    write_predictions(
        data["eval_row_id"],
        preds.cpu().numpy(),
        pred_path,
    )