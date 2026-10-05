# README — Hướng dẫn full bài Lab Day 1 và các checkpoint cần làm để nộp được

## 1. Mục tiêu của lab

Lab Day 1 yêu cầu bạn xây dựng một mạng nơ-ron đơn giản để phân loại Forest CoverType (7 lớp) từ dữ liệu tabular. Mục tiêu không phải chỉ chạy code cho xong, mà là bạn phải:

- hiểu đúng quy trình huấn luyện mạng nơ-ron,
- kiểm chứng dữ liệu và mô hình trước khi huấn luyện,
- chạy các thí nghiệm theo đúng logic khoa học,
- chọn cấu hình dựa trên validation, không dùng eval để chọn model,
- xuất ra đầy đủ sản phẩm cần thiết để nộp bài.

Điểm quan trọng: lab này đánh giá cả quá trình kiểm tra và cách bạn phân tích kết quả, không chỉ số điểm cuối cùng.

---

## 2. Dữ liệu và quy tắc bắt buộc

### 2.1 Dữ liệu gốc

- Dữ liệu được lưu trong thư mục `data/`
- Có sẵn:
  - `data/covtype.csv.gz`
  - `data/split_metadata.csv`
- Bạn không được sửa `split_metadata.csv`.
- Bạn không được tự chia lại train/eval theo cách khác so với quy định.

### 2.2 Quy tắc chia dữ liệu

- `train`: 464 809 mẫu
- `eval`: 116 203 mẫu
- Bạn phải chạy script `scripts/split_data.py` để tạo:
  - `data/processed/train.npz`
  - `data/processed/eval.npz`

### 2.3 Validation

- Sau khi có `train.npz`, bạn phải tự tách validation từ `train`.
- Khuyến nghị: tách 20% validation, phân tầng theo nhãn, seed 42.
- Validation phải dùng để chọn learning rate, checkpoint và cấu hình cuối.
- Eval chỉ nên dùng ở bước cuối cùng để đánh giá tổng thể, không dùng để chọn mô hình.

### 2.4 Chuẩn hoá dữ liệu

- Chỉ chuẩn hoá 10 cột số liên tục.
- Tính mean và std chỉ trên phần train còn lại sau khi tách validation.
- Không tính mean/std trên toàn bộ dữ liệu.
- Eval phải dùng cùng mean/std của train.

> Nếu làm sai bước này, lab có thể bị trừ điểm nặng vì vi phạm quy tắc chuẩn hoá và chia tập.

---

## 3. Cấu trúc bài lab

Bài lab được chia theo 4 phần chính:

### Part 0 — Dữ liệu

Nhiệm vụ:

- chạy `scripts/split_data.py`,
- load dữ liệu,
- tách validation từ train,
- chuẩn hoá,
- kiểm tra shape và phân bố lớp,
- kiểm tra baseline “luôn đoán lớp đa số” trên validation.

Kết quả mong đợi:

- `X_tr`, `X_val`, `X_eval` có shape hợp lệ,
- validation có kích thước khoảng 20% của train,
- mean/std của 10 cột số trên train gần bằng 0/1,
- “luôn đoán lớp đa số” trên validation khoảng 0.4876 accuracy.

### Part 1 — Model và kiểm tra "sức khoẻ" ban đầu

Nhiệm vụ:

- định nghĩa model `MLP` của riêng bạn,
- kiểm tra số tham số,
- kiểm tra shape logits `(B, 7)`,
- kiểm tra loss ở trạng thái ban đầu gần `ln(7) ≈ 1.946`,
- chạy overfit trên 20 mẫu để xem loss giảm về gần 0,
- kiểm tra gradient không bằng 0 sau backward.

Kết quả mong đợi:

- model có đúng số tham số như quy định,
- logits xuất ra 7 lớp,
- loss ban đầu ở mức hợp lý,
- gradient có thể cập nhật tham số.

### Part 2 — Pipeline và baseline

Nhiệm vụ:

- chọn learning rate bằng validation,
- chạy baseline với ít nhất 2 seed, khuyến nghị 3 seed,
- lưu log/history của mỗi chạy,
- vẽ đồ thị train/val loss và các metric,
- so sánh với mốc “luôn đoán lớp đa số”.

Kết quả mong đợi:

- baseline chạy ổn định,
- val loss giảm và val accuracy vượt mốc 0.4876,
- các seed không quá nhiễu.

### Part 3 — Thí nghiệm

Nhiệm vụ:

- chọn ít nhất một thí nghiệm để kiểm tra một yếu tố khác biệt so với baseline,
- chỉ thay đổi đúng 1 yếu tố trong mỗi thí nghiệm,
- viết dự đoán trước khi chạy,
- so sánh với baseline sau khi chạy,
- có đồ thị riêng cho từng thí nghiệm.

Ví dụ các biến để thử:

- learning rate,
- optimizer (SGD, Adam),
- dropout,
- weight decay,
- batch size,
- khởi tạo,
- clipping gradient,
- loss function.

### Part 4 — Đánh giá cuối trên eval, bảng và nộp bài

Nhiệm vụ:

- chọn cấu hình cuối cùng chỉ bằng validation,
- chạy model final trên toàn bộ eval,
- lưu file dự đoán `predictions_eval.csv`,
- chạy script đánh giá `scripts/evaluate.py`,
- lưu `eval_result.json`,
- phân tích lỗi theo lớp,
- tạo `experiments.xlsx`, báo cáo và kiểm tra full submission.

---

## 4. Các checkpoint bắt buộc để hoàn thành bài lab

Dưới đây là các mốc kiểm tra mà bạn nên làm theo thứ tự. Nếu bỏ qua một checkpoint thì rất dễ sai hoặc không nộp được.

### Checkpoint 0 — Khởi tạo môi trường và repo

Bạn phải chắc chắn:

- [X] repo đã được copy vào đúng thư mục `submission_<MSSV>/code/`
- [X] `REPO_ROOT` và `OUT_DIR` đã được thiết lập đúng
- [X] `device` đã chọn đúng (`cuda` nếu có GPU, otherwise `cpu`)
- [X] folder `figures/` và `results/` đã được tạo
- [X] đọc kỹ `README.md`, `GUIDE.md`, `RUBRIC.md`

Nếu chưa làm xong phần này thì không nên chạy các phần sau.

### Checkpoint 1 — Dữ liệu đã sẵn sàng

Bạn cần xác nhận:

- [X] chạy `scripts/split_data.py` thành công
- [X] `data/processed/train.npz` và `eval.npz` đã tạo ra
- [X] `X_tr`, `X_val`, `X_eval` có shape đúng
- [X] validation đã tách từ `train` theo đúng quy tắc
- [X] 10 cột số đã chuẩn hoá bằng mean/std của train
- [X] “luôn đoán lớp đa số” trên val khoảng 0.4876

> Đây là checkpoint cực kỳ quan trọng vì nếu dữ liệu sai, toàn bộ lab sẽ sai.

### Checkpoint 2 — Model đã “khỏe” và có thể học

Bạn cần kiểm tra:

- [ ] model được định nghĩa bằng class `nn.Module` của bạn
- [ ] số tham số khớp với `EXPECTED_PARAMS`
- [ ] logits shape đúng `(B, 7)`
- [ ] loss ban đầu gần `ln(7)`
- [ ] overfit 20 mẫu cho loss gần 0
- [ ] gradient sau backward không bằng 0

Nếu model không học được, bạn phải sửa mô hình hoặc pipeline trước khi đi tiếp.

### Checkpoint 3 — Baseline chạy ổn định

Bạn cần có:

- [ ] learning rate được chọn bằng validation
- [ ] ít nhất 2 seed chạy baseline, khuyến nghị 3 seed
- [ ] lưu `results/<exp_id>.json` cho mỗi chạy
- [ ] lưu `figures/<exp_id>.png` cho mỗi chạy
- [ ] thống kê mean ± std cho `val_macro_f1` và `val_acc`
- [ ] so sánh với mốc dự đoán lớp đa số

Nếu baseline không ổn định, bạn chưa đủ điều kiện để làm thí nghiệm nâng cấp.

### Checkpoint 4 — Có ít nhất một thí nghiệm hợp lệ

Bạn cần:

- [ ] có dự đoán trước rõ ràng
- [ ] chỉ đổi đúng 1 yếu tố so với baseline
- [ ] lưu ảnh và JSON cho thí nghiệm
- [ ] so sánh với baseline bằng val loss / val_acc / val_macro_f1
- [ ] có nhận xét về vì sao khác biệt xảy ra

> Một thí nghiệm hợp lệ tốt hơn nhiều so với chạy quá nhiều thí nghiệm nhưng không có logic.

### Checkpoint 5 — Final eval và chuẩn bị nộp bài

Bạn cần chắc chắn:

- [ ] chọn cấu hình cuối cùng chỉ bằng validation
- [ ] chạy `final_eval` và lưu `predictions_eval.csv`
- [ ] chạy `python scripts/evaluate.py --pred ... --out ...`
- [ ] lưu `eval_result.json`
- [ ] đọc kết quả accuracy và macro-F1
- [ ] phân tích lớp nào khó nhất và viết nhận xét
- [ ] tạo `experiments.xlsx`
- [ ] tạo `REPORT.md`
- [ ] thư mục nộp đúng cấu trúc yêu cầu

---

## 5. Cấu trúc thư mục tối thiểu để nộp bài

Thư mục nộp cần có dạng tương tự:

- `submission_<MSSV>/`
  - `REPORT.md`
  - `eval_result.json`
  - `predictions_eval.csv`
  - `experiments.xlsx`
  - `figures/`
  - `results/`
  - `code/`
    - `lab.ipynb`
    - `data.py`
    - `model.py`
    - `optimizer.py`
    - `train.py`
    - `plots.py`
    - `results_table.py`

> Lưu ý: toàn bộ code phải nằm trong `code/`. Không nộp checkpoint, dữ liệu thừa hoặc file tạm.

---

## 6. Tiêu chí để được xem là “đủ điều kiện nộp”

Một bài lab được coi là đủ điều kiện nộp khi bạn đạt đủ các điều kiện sau:

- [ ] chạy đúng quy trình dữ liệu và không sửa `split_metadata.csv`
- [ ] validation và chuẩn hoá được thực hiện đúng
- [ ] model có kiểm tra ban đầu hợp lệ
- [ ] baseline có ít nhất 2 seed với kết quả được lưu lại
- [ ] có ít nhất 1 thí nghiệm có logic rõ ràng
- [ ] chọn cấu hình cuối theo validation, không dùng eval để chọn mô hình
- [ ] file dự đoán và score eval đã được tạo đầy đủ
- [ ] có hình ảnh và JSON lưu cho từng thí nghiệm
- [ ] báo cáo và bảng kết quả đã được hoàn thiện
- [ ] notebook chạy lại được từ đầu đến cuối nếu Restart & Run All

---

## 7. Mẹo làm bài hiệu quả

1. Làm đúng theo thứ tự: dữ liệu → model → baseline → thí nghiệm → final eval.
2. Không chạy quá nhiều thí nghiệm cùng lúc.
3. Mỗi thí nghiệm chỉ thay đổi 1 yếu tố để dễ giải thích.
4. Luôn lưu kết quả riêng từng `exp_id`.
5. Dùng validation để chọn cấu hình; eval chỉ để báo cáo cuối.
6. Đừng quên viết nhận xét bằng tiếng Việt/Anh rõ ràng cho từng phần.

---

## 8. Kết luận ngắn gọn

Nếu bạn hoàn thành đúng 5 checkpoint trên, bạn đã đi đủ xa để có thể nộp bài lab với logic và sản phẩm đầy đủ. Cái quan trọng nhất là không bỏ qua các bước kiểm tra ban đầu. Nhiều bạn mất điểm không phải vì model không học, mà vì dữ liệu, chuẩn hoá, hoặc chọn cấu hình sai.

Hãy làm theo các checkpoint như một checklist, không cần vội vàng. Khi mỗi checkpoint đạt yêu cầu, mới chuyển sang phần sau.
