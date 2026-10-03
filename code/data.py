from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

N_NUMERIC = 10


def load_split(processed_dir: str = "data/processed"):
    """Load the pre-split train/eval arrays from the processed .npz files."""
    train_arr = np.load(f"{processed_dir}/train.npz")
    eval_arr = np.load(f"{processed_dir}/eval.npz")

    X_train = np.asarray(train_arr["X"], dtype=np.float32)
    y_train = np.asarray(train_arr["y"], dtype=np.int64)
    X_eval = np.asarray(eval_arr["X"], dtype=np.float32)
    y_eval = np.asarray(eval_arr["y"], dtype=np.int64)
    eval_row_id = np.asarray(eval_arr["row_id"], dtype=np.int64)

    assert X_train.ndim == 2 and X_train.shape[1] == 54, X_train.shape
    assert X_eval.ndim == 2 and X_eval.shape[1] == 54, X_eval.shape
    assert y_train.shape == (X_train.shape[0],), y_train.shape
    assert y_eval.shape == (X_eval.shape[0],), y_eval.shape
    assert X_train.dtype == np.float32 and X_eval.dtype == np.float32
    assert y_train.dtype == np.int64 and y_eval.dtype == np.int64
    assert eval_row_id.dtype == np.int64
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Split train data into train/validation while preserving the class balance."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=val_fraction,
        random_state=seed,
        stratify=y,
        shuffle=True,
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Compute mean/std for the first 10 numeric columns on the training split only."""
    X_tr = np.asarray(X_tr, dtype=np.float32)
    mean = X_tr[:, :N_NUMERIC].mean(axis=0)
    std = X_tr[:, :N_NUMERIC].std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X, mean, std):
    """Standardize the numeric columns and keep the one-hot part unchanged."""
    X = np.asarray(X, dtype=np.float32).copy()
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    std_safe = np.where(std == 0, 1.0, std)
    X[:, :N_NUMERIC] = (X[:, :N_NUMERIC] - mean) / std_safe
    return X.astype(np.float32)


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Prepare normalized train/val/eval tensors on the requested device."""
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)
    mean, std = fit_standardizer(X_tr)

    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    device_obj = torch.device(device)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device_obj)
    y_tr_t = torch.tensor(y_tr, dtype=torch.int64, device=device_obj)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device_obj)
    y_val_t = torch.tensor(y_val, dtype=torch.int64, device=device_obj)
    X_eval_t = torch.tensor(X_eval, dtype=torch.float32, device=device_obj)
    y_eval_t = torch.tensor(y_eval, dtype=torch.int64, device=device_obj)

    majority = int(np.bincount(y_val).argmax())
    majority_acc = float((y_val == majority).mean())
    print(f"train X={X_tr_t.shape}, val X={X_val_t.shape}, eval X={X_eval_t.shape}")
    print(f"val majority-class accuracy = {majority_acc:.4f}")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Yield batches of (X, y) until the complete dataset is covered."""
    n = X.shape[0]
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if shuffle:
        perm = torch.randperm(n, device=X.device, generator=generator) if generator is not None else torch.randperm(n, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        yield X[idx], y[idx]
