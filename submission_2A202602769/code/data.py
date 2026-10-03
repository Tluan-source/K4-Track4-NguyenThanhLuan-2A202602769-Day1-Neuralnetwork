"""data.py — data loading, validation split, normalization, and batching.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    Các bước:
      1. np.load(f"{processed_dir}/train.npz") -> khoá "X", "y"
      2. np.load(f"{processed_dir}/eval.npz")  -> khoá "X", "y", "row_id"
      3. assert shape/dtype đúng quy ước ở đầu file
    """
    train_arr = np.load(f"{processed_dir}/train.npz")
    eval_arr = np.load(f"{processed_dir}/eval.npz")
    X_train = np.asarray(train_arr["X"], dtype=np.float32)
    y_train = np.asarray(train_arr["y"], dtype=np.int64)
    X_eval = np.asarray(eval_arr["X"], dtype=np.float32)
    y_eval = np.asarray(eval_arr["y"], dtype=np.int64)
    eval_row_id = np.asarray(eval_arr["row_id"], dtype=np.int64)
    assert X_train.ndim == 2 and X_train.shape[1] == 54
    assert X_eval.ndim == 2 and X_eval.shape[1] == 54
    assert y_train.shape == (len(X_train),) and y_eval.shape == (len(X_eval),)
    assert eval_row_id.shape == (len(X_eval),)
    assert np.all((y_train >= 0) & (y_train < 7))
    assert np.all((y_eval >= 0) & (y_eval < 7))
    return X_train, y_train, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    Gợi ý: sklearn.model_selection.train_test_split(..., stratify=y, random_state=seed)
    Dùng CÙNG seed và val_fraction cho mọi thí nghiệm để so sánh công bằng.
    """
    from sklearn.model_selection import train_test_split
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    return train_test_split(
        X, y, test_size=val_fraction, random_state=seed, stratify=y, shuffle=True
    )


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    Câu hỏi: vì sao không được tính trên toàn bộ dữ liệu hay trên eval?
    """
    X_tr = np.asarray(X_tr, dtype=np.float32)
    mean = X_tr[:, :N_NUMERIC].mean(axis=0)
    std = X_tr[:, :N_NUMERIC].std(axis=0)
    return mean.astype(np.float32), np.where(std == 0, 1.0, std).astype(np.float32)


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên.

    Chú ý: không sửa X tại chỗ nếu bạn còn dùng lại nó; chú ý std = 0 (nếu có).
    """
    X_out = np.asarray(X, dtype=np.float32).copy()
    mean = np.asarray(mean, dtype=np.float32)
    std = np.where(np.asarray(std, dtype=np.float32) == 0, 1.0, std)
    X_out[:, :N_NUMERIC] = (X_out[:, :N_NUMERIC] - mean) / std
    return X_out


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    Các bước:
      1. load_split -> make_val_split -> fit_standardizer (chỉ trên X_tr)
      2. apply_standardizer cho X_tr, X_val, X_eval bằng CÙNG mean/std
      3. torch.tensor(..., device=device); X là float32, y là int64
      4. in ra kích thước các tập và accuracy của chiến lược "luôn đoán lớp đa số" trên val
    """
    X_train, y_train, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, X_val, y_tr, y_val = make_val_split(
        X_train, y_train, val_fraction=val_fraction, seed=seed
    )
    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)
    device_obj = torch.device(device)
    data = {
        "X_tr": torch.tensor(X_tr, dtype=torch.float32, device=device_obj),
        "y_tr": torch.tensor(y_tr, dtype=torch.int64, device=device_obj),
        "X_val": torch.tensor(X_val, dtype=torch.float32, device=device_obj),
        "y_val": torch.tensor(y_val, dtype=torch.int64, device=device_obj),
        "X_eval": torch.tensor(X_eval, dtype=torch.float32, device=device_obj),
        "y_eval": torch.tensor(y_eval, dtype=torch.int64, device=device_obj),
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
    }
    majority = int(np.bincount(y_val).argmax())
    print(f"train X={data['X_tr'].shape}, val X={data['X_val'].shape}, eval X={data['X_eval'].shape}")
    print(f"val majority-class accuracy = {float((y_val == majority).mean()):.4f}")
    return data


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader.

    Các bước:
      1. nếu shuffle: perm = torch.randperm(len(X), generator=generator, device=X.device); ngược lại arange
      2. for i in range(0, N, batch_size): idx = perm[i:i+batch_size]; yield X[idx], y[idx]
    Chú ý: batch cuối có thể nhỏ hơn batch_size; hãy quyết định bạn xử lý thế nào và ghi lại.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, device=X.device, generator=generator)
    else:
        perm = torch.arange(n, device=X.device)
    for start in range(0, n, batch_size):
        idx = perm[start:start + batch_size]
        yield X[idx], y[idx]
