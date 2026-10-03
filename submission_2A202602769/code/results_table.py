"""results_table.py — JSON result persistence and Excel report generation.

Nhiệm vụ: lưu kết quả từng lần chạy ra JSON, rồi điền vào experiments.xlsx từ mẫu
templates/experiment_table_template.xlsx (đừng gõ tay hàng chục dòng, rất dễ sai).

Tên cột của sheet "Experiments" (giữ nguyên, đúng thứ tự mẫu):
    exp_id, group, description, loss, optimizer, lr, weight_decay, batch, epochs, hidden, dropout,
    clip_norm, precision, init, seed, step0_loss, best_val_loss, best_epoch, final_train_loss,
    final_val_loss, val_acc, val_macro_f1, time_per_epoch_s, peak_mem_MB, diverged,
    eval_acc, eval_macro_f1, figure_file, notes
(các cột công thức ở cuối bảng mẫu tự tính, đừng ghi đè)
"""
from __future__ import annotations

import json
from pathlib import Path
from openpyxl import Workbook, load_workbook
from openpyxl.formula.translate import Translator


def save_result(result: dict, results_dir: str = "../results") -> str:
    """Ghi result["cfg"], result["history"], result["summary"] (KHÔNG ghi best_state) ra
    <results_dir>/<exp_id>.json. Trả về đường dẫn file. Tạo thư mục nếu chưa có."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['cfg']['exp_id']}.json"
    payload = {key: result[key] for key in ("cfg", "history", "summary")}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def load_results(results_dir: str = "../results") -> list[dict]:
    """Đọc mọi file *.json trong results_dir, trả về danh sách dict (sắp theo exp_id)."""
    out_dir = Path(results_dir)
    if not out_dir.exists():
        return []
    results = [json.loads(path.read_text(encoding="utf-8")) for path in out_dir.glob("*.json")]
    return sorted(results, key=lambda result: result.get("cfg", {}).get("exp_id", ""))


def to_row(result: dict, eval_scores: dict | None = None, notes: str = "") -> dict:
    """Biến một kết quả thành một dòng của bảng: gộp cfg + summary (+ eval_acc, eval_macro_f1 nếu có)
    + figure_file = f"figures/{exp_id}.png". Khoá phải trùng tên cột ở đầu file.
    Chỉ truyền eval_scores cho baseline và cấu hình cuối cùng."""
    cfg = result["cfg"]
    summary = result["summary"]
    return {
        "exp_id": cfg["exp_id"], "group": cfg["group"], "description": cfg["description"],
        "loss": cfg["loss"], "optimizer": cfg["optimizer"], "lr": cfg["lr"],
        "weight_decay": cfg["weight_decay"], "batch": cfg["batch"], "epochs": cfg["epochs"],
        "hidden": str(tuple(cfg["hidden"])), "dropout": cfg["dropout"], "clip_norm": cfg["clip_norm"],
        "precision": cfg["precision"], "init": cfg["init"], "seed": cfg["seed"],
        "step0_loss": summary.get("step0_loss"), "best_val_loss": summary.get("best_val_loss"),
        "best_epoch": summary.get("best_epoch"), "final_train_loss": summary.get("final_train_loss"),
        "final_val_loss": summary.get("final_val_loss"), "val_acc": summary.get("val_acc"),
        "val_macro_f1": summary.get("val_macro_f1"), "time_per_epoch_s": summary.get("time_per_epoch_s"),
        "peak_mem_MB": summary.get("peak_mem_MB"), "diverged": summary.get("diverged", False),
        "eval_acc": None if eval_scores is None else eval_scores.get("accuracy"),
        "eval_macro_f1": None if eval_scores is None else eval_scores.get("macro_f1"),
        "figure_file": f"figures/{cfg['exp_id']}.png", "notes": notes,
    }


def write_xlsx(rows: list[dict], template_path: str, out_path: str) -> None:
    """Điền các dòng vào sheet "Experiments" của mẫu, từ dòng 2 trở xuống, rồi lưu thành out_path.

    Các bước (openpyxl):
      1. wb = openpyxl.load_workbook(template_path)   # KHÔNG dùng data_only=True (sẽ mất công thức)
      2. ws = wb["Experiments"]; đọc tiêu đề dòng 1 để biết cột nào ứng với khoá nào
      3. với mỗi row: ghi giá trị vào đúng cột; BỎ QUA các cột công thức (step0_gap_vs_lnC, gap_val_minus_train,
         delta_val_f1_vs_base, beyond_noise)
      4. wb.save(out_path)
    Sau khi lưu, mở file bằng Excel/LibreOffice để các công thức tính lại.
    """
    wb = load_workbook(template_path) if Path(template_path).exists() else Workbook()
    if "Experiments" not in wb.sheetnames:
        wb.create_sheet("Experiments")
    ws = wb["Experiments"]
    headers = [cell.value for cell in ws[1]]
    formula_templates = {
        col: cell.value for col, cell in enumerate(ws[2], start=1)
        if isinstance(cell.value, str) and cell.value.startswith("=")
    } if ws.max_row >= 2 else {}
    if ws.max_row > 1:
        ws.delete_rows(2, ws.max_row - 1)
    for row_idx, row in enumerate(rows, start=2):
        for col_idx, header in enumerate(headers, start=1):
            if header in row:
                ws.cell(row_idx, col_idx).value = row[header]
            elif col_idx in formula_templates:
                origin = ws.cell(2, col_idx).coordinate
                target = ws.cell(row_idx, col_idx).coordinate
                ws.cell(row_idx, col_idx).value = Translator(formula_templates[col_idx], origin=origin).translate_formula(target)
    if "Seeds" in wb.sheetnames:
        seeds = wb["Seeds"]
        baseline_ids = [row["exp_id"] for row in rows if row.get("group") == "baseline"]
        for row_idx in range(2, seeds.max_row + 1):
            seeds.cell(row_idx, 1).value = baseline_ids[row_idx - 2] if row_idx - 2 < len(baseline_ids) else None
    if "Summary" in wb.sheetnames:
        summary = wb["Summary"]
        notes_by_group = {}
        for row in rows:
            notes_by_group.setdefault(row.get("group"), []).append(row.get("description", ""))
        for row_idx in range(2, summary.max_row + 1):
            group = summary.cell(row_idx, 1).value
            if group in notes_by_group:
                summary.cell(row_idx, 8).value = "; ".join(notes_by_group[group])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
