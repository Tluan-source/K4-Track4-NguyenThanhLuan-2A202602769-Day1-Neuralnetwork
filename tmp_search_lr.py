import sys
from pathlib import Path
sys.path.insert(0, "code")
import torch
from data import prepare_data
from train import DEFAULT_CFG, run_experiment

repo = Path(".").resolve()
print("repo", repo)
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device", device, "torch", torch.__version__)

data = prepare_data(device, val_fraction=0.2, seed=42, processed_dir=str(repo / "data" / "processed"))
results = []
for lr in [0.01, 0.03, 0.05, 0.07, 0.1]:
    cfg = {**DEFAULT_CFG, "lr": lr, "seed": 42, "exp_id": f"lr{lr}"}
    result = run_experiment(cfg, data)
    summary = result["summary"]
    print("LR", lr, "val_macro_f1", summary["val_macro_f1"], "val_acc", summary["val_acc"], "best_epoch", summary["best_epoch"])
    results.append((lr, summary["val_macro_f1"], summary["val_acc"]))
print("BEST", sorted(results, key=lambda x: x[1], reverse=True)[:5])
