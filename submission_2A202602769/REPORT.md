# Báo cáo Lab Day 1 — Track 4

## 1. Thiết lập và dữ liệu
Notebook chạy trên Colab T4 với PyTorch 2.11.0. Forest CoverType được chia cố định bằng metadata: 464809 mẫu train và 116203 mẫu eval. Từ train, validation phân tầng 20% bằng seed 42 tạo 371847 mẫu train và 92962 mẫu validation. Mười cột số được chuẩn hóa chỉ bằng thống kê train. Mốc đoán lớp đa số trên validation là accuracy 0.4876.

M-base có dạng 54 -> 256 -> 128 -> 7, logits `(B, 7)`, ReLU, He initialization và 47879 tham số. Part 1 kiểm tra loss bước 0, khả năng quá khớp 20 mẫu và gradient khác 0 của mọi tham số.

## 2. Baseline và chọn learning rate
Sweep trên validation với seed 42 được lưu trong `experiments.xlsx`: `base-lr0.001` = 0.5986, `base-lr0.01` = 0.7704, `base-lr0.05` = 0.8323, `base-lr0.1` = 0.8533 và `base-lr0.2` = 0.8581 macro-F1. Vì vậy lr=0.2 được chọn chỉ bằng validation.

Ba seed baseline có macro-F1 lần lượt 0.8581, 0.8596 và 0.8632; trung bình 0.8603, độ lệch chuẩn 0.0026 và ngưỡng nhiễu 2sigma 0.0053. Đây là phạm vi lặp lại dùng để diễn giải các thay đổi.

## 3. Thí nghiệm và kết quả trái dự đoán
Mỗi thí nghiệm giữ cùng split, batch, số epoch và seed 42, chỉ đổi một yếu tố so với baseline.

- `opt-adam`: dự đoán Adam sẽ giảm validation loss nhanh hơn SGD+momentum. Kết quả macro-F1 0.4814, delta -0.3789 so với baseline trung bình và không vượt ngưỡng 2sigma. Đây là kết quả trái dự đoán; Adam dùng cùng lr=0.2, chưa được sweep lr riêng nên so sánh optimizer chưa công bằng tuyệt đối.
- `dropout-02`: dự đoán dropout 0.2 có thể giảm overfit. Kết quả macro-F1 0.8193, delta -0.0410 và không vượt 2sigma. Trong 20 epoch baseline chưa cho thấy nhu cầu regularization mạnh; dropout làm giảm năng lực cập nhật trong phạm vi này.

Mỗi dòng có JSON trong `results/`, ảnh tương ứng trong `figures/`, và ảnh nhóm `compare_lr.png`, `compare_part3.png`.

## 4. Đánh giá cuối trên eval
Cấu hình cuối `base-s42` được chọn từ validation trước khi dùng eval. `scripts/evaluate.py` cho accuracy 0.9118 và macro-F1 0.8644. Các số được lưu nguyên trong `eval_result.json`.

### Phân tích lỗi theo lớp
| class | support | precision | recall | F1 |
|---:|---:|---:|---:|---:|
| 0 | 42368 | 0.9091 | 0.9103 | 0.9097 |
| 1 | 56661 | 0.9246 | 0.9271 | 0.9258 |
| 2 | 7151 | 0.9250 | 0.8722 | 0.8978 |
| 3 | 549 | 0.7278 | 0.8816 | 0.7974 |
| 4 | 1899 | 0.7770 | 0.7873 | 0.7821 |
| 5 | 3473 | 0.8118 | 0.8183 | 0.8150 |
| 6 | 4102 | 0.9192 | 0.9264 | 0.9228 |

Lớp khó nhất là class 4 với F1 0.7821 trên 1899 mẫu. Cặp nhầm ngoài đường chéo lớn nhất là class 0 -> class 1 với 3454 mẫu. Ma trận đầy đủ nằm trong output Part 4 và `eval_result.json`.

## 5. Câu hỏi dẫn dắt
Khi loss không giảm sau 2000 bước, ba kiểm tra đầu tiên là: (1) dữ liệu, nhãn, shape, chuẩn hóa và loss bước 0; (2) logits, gradient khác 0, learning-rate và tham số có được cập nhật; (3) train/eval mode, batch order, loss hữu hạn và đường cong train/validation. Ba kiểm tra này phân biệt lỗi dữ liệu, kiến trúc và training loop.

## 6. Giới hạn
Chỉ có ba seed baseline, hai thí nghiệm Part 3 và một learning-rate sweep cho SGD. Adam chưa được tối ưu learning-rate riêng. Chưa có CE-vs-MSE, clipping, mixed precision hoặc initialization comparison. Vì vậy kết luận về optimizer và dropout chỉ áp dụng cho phạm vi cấu hình đã chạy; số seed nhỏ cũng giới hạn độ chắc chắn. Eval chỉ dùng cho đánh giá cuối, không dùng để chọn model.
