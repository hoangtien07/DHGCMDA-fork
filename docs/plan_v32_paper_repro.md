# SPEC + PLAN — Tái hiện kết quả HMDD v3.2 của bài báo DHGCMDA

Ngày: 2026-11-23 · Trạng thái: **CHỜ DUYỆT** (chưa cài đặt, chưa chạy gì)

> **ĐÃ ĐƯỢC THAY THẾ** bởi `docs/strategy_thesis_dhgcmda.md` (bổ sung hiệu chuẩn v2.0 và phần cải tiến). Giữ lại để tham khảo.

---

## 1. Mục tiêu

Tạo ra trên bộ dữ liệu paper (SPLD MDAv3.2-3: 411 miRNA × 271 bệnh × 11.748 triplet) các con số **gần nhất** với hàng v3.2 trong bài báo:

| Bảng | Chỉ số paper v3.2 |
|---|---|
| Table 3 (CV_type) | P = 0,7915 · R = 0,9421 · F1 = 0,8600 |
| Table 4 (CV_triplet) | AUPR = 0,9271 · AUC = 0,9181 · F1 = 0,8674 |

**Tiêu chí "khớp":** |Δ| ≤ 0,02 với AUC/AUPR/F1, ≤ 0,03 với P/R. Không liên hệ tác giả hay thầy. Mặc định coi các con số trong paper là đúng như kết quả mà code của tác giả in ra; chỗ lệch nằm ở data, code hoặc cách tính.

## 2. Phát hiện làm cơ sở (đã kiểm chứng bằng đọc code/log, chưa chạy mới)

1. **Hàm Top-1 phát hành không chấm được v3.2.** `compute_top1_metrics` (commit `06f9ba3`) chỉ chấp nhận vector dự đoán dài 4 hoặc 5 và nhãn 1..4. Với v3.2 (5 loại), mọi mẫu đều bị `continue` và khối "Top-1 Metrics" in ra 0.
2. **Mã phát hành in chỉ số nhị phân dưới tiêu đề "CV_type".** Khối `📊 CV_type Style Metrics Details` in AUC/AUPR/F1/Accuracy/Recall/Specificity/Precision. Đây là chỉ số **existence nhị phân** trên các cặp test của chia fold theo cặp (chính là CV_type), với 1:1 mẫu âm và ngưỡng chọn để F1 lớn nhất trên test.
3. **Cặp (P, R) của paper bất khả thi với mọi quy ước "loại"** (định lý Track R; tôi kiểm thêm macro-P kiểu sklearn: cũng không đạt). Ngược lại, nó **hoàn toàn khả thi** với chỉ số nhị phân: P=0,79, R=0,94 ở 1:1 cho specificity 0,75 và accuracy 0,85, tức một điểm hợp lý trên đường ROC có AUC ≈ 0,92. Mẫu R > P là đặc trưng của ngưỡng max-F1.
4. **Log cũ đã ở cùng "vùng" số.** Các lần chạy pipeline phát hành trên v3.2 cho chỉ số nhị phân:
   - v3.2_wang: P 0,790 · R 0,869 · F1 0,827 · AUC 0,903
   - MDAv3.2-3 (`logs/r3_leaky_spld_exact.log`): P 0,756 · R 0,861 · F1 0,805 · AUC 0,878 · AUPR 0,881 · Acc 0,791
5. **Lần chạy r3 trên đúng data paper lệch mã phát hành ở 4 điểm:**
   - (a) view `M_GSM` là bản sao `M_FSM`, không phải GIP, nên mất view GIP. Đây là view mang tín hiệu existence mạnh nhất.
   - (b) `D_SSM2` là bản sao `D_SSM1`.
   - (c) 300 epoch thay vì 650.
   - (d) 4 head + full_bilinear + các flag fork, thay vì mặc định phát hành (8 head, diag).

   Recipe view của mã phát hành v2.0: miRNA = {M_FSM chức năng, M_GSM GIP}, bệnh = {D_SSM1, D_SSM2 ngữ nghĩa}. Trong đó GIP được tính sẵn từ **toàn bộ** ma trận liên kết.
6. **Data đúng là của paper.** Hàng SPLDHyperAWNTF trong Table 3 (0,6219/0,4624) được tái tạo khớp tới 4 chữ số trên MDAv3.2-3 (ledger F20).

## 3. Giả thuyết cần kiểm định

| ID | Giả thuyết | Dự đoán kiểm chứng được |
|---|---|---|
| **H1** (metric) | P/R/F1 "Top-1" v3.2 của paper = Precision/Recall/F1 nhị phân trong khối "CV_type Style", chia theo cặp | Khi H2+H3 được áp dụng, P/R/F1 nhị phân tiến về 0,79/0,94/0,86 |
| **H2** (input) | Tác giả dùng đúng recipe 4 view như v2.0, có GIP miRNA (và có thể GIP bệnh) tính từ toàn bộ ma trận v3.2 | Thêm GIP làm AUC tăng rõ từ 0,878 lên ≥ 0,90 |
| **H3** (config) | Tác giả chạy cấu hình mặc định phát hành (650 epoch, 8 head, K=13, diag predictor, exist/type weight gốc) | Cấu hình phát hành ≥ cấu hình fork r3 |
| **H4** (CV_triplet) | Hàng Table 4 v3.2 là khối nhị phân của một lần chạy (cùng pipeline hoặc chia theo triplet) | AUC/AUPR của chạy tốt nhất ≈ 0,918/0,927 |

**Tiêu chí bác bỏ H1:** nếu đã có GIP và 650 epoch mà AUC vẫn < 0,89 và R < 0,88, ghi nhận H1 yếu. Khi đó chuyển sang kiểm tra phương án phụ: "P paper" = Accuracy nhị phân (r3 có Acc = 0,7911 ≈ 0,7915).

## 4. Nguyên tắc thực thi

- **Không sửa** file gốc của repo/fork. Mọi code mới nằm trong `paper_v32_repro/`; data mới nằm trong `paper_v32_repro/data/`.
- Code nền = **mã phát hành nguyên bản** (`git show 06f9ba3:<file>`), chỉ port tối thiểu cho 5 loại và kích thước 411×271. Mọi diff được ghi trong `PORT_DIFF.md`.
- Giữ nguyên các "đặc điểm" của tác giả, kể cả rò rỉ (GIP/feature từ ma trận đầy đủ, ghi đè nhãn đa loại, ngưỡng max-F1 trên test). Mục tiêu là tái hiện, không làm sạch.
- Mỗi lần chạy lưu: config, SHA256 input, log đầy đủ, JSON kết quả (cả 3 khối: binary, CV_type style, Top-1 đã sửa cho K=5 để tham khảo).
- Không tinh chỉnh ngoài ma trận thí nghiệm dưới đây. Nếu cần thêm biến thể, quay lại xin duyệt.

## 5. Các giai đoạn

### P0 — Môi trường (≈ 30–45 phút, cần mạng ~2,5–3 GB)
- `uv venv .venv --python 3.12` trong worktree (đã có trong `.gitignore`).
- Cài `torch==2.5.1` (đúng phiên bản paper). GPU GTX 1650 4 GB, driver 462.30 (CUDA 11.2), nên thử bản **cu118** (dựa vào minor-version compatibility). Nếu `torch.cuda.is_available()` lỗi thì dùng **CPU** (`torch==2.5.1+cpu`).
- Cài `torch-geometric 2.6.1`, numpy, pandas, scipy, scikit-learn, openpyxl, xlrd.
- Smoke test: mã phát hành trên v2.0, 3 epoch × 2 fold.
- ⚠ Rủi ro: driver cũ có thể không chạy cu118. Fallback CPU chậm hơn khoảng 3–5×. Phương án phụ: nâng driver NVIDIA (cần anh tự làm).

### P1 — Kiểm chứng tĩnh H1 (≈ 30 phút, không train)
- Đưa vector dự đoán giả dài 6 (existence + 5 loại) và nhãn 1..5 vào evaluator phát hành. Xác nhận Top-1 = 0 và khối "CV_type Style" in số nhị phân.
- Đối chiếu v2.0: khối nhị phân của run phát hành v2.0 so với hàng CV_triplet v2.0 của paper (0,9738/0,9669/0,9278), để xác nhận paper lấy CV_triplet từ khối này.

### P2 — Dựng data `paper_v32_repro/data/` (≈ 1 giờ)

| File (theo format v2.0) | Nguồn |
|---|---|
| `multi_all_mirna_disease_pairs_without_negative.csv` | `triplets.csv` của MDAv3.2-3, mã loại 1..5, **sắp tăng theo mã** như v2.0 (để tái tạo hành vi ghi đè "mã lớn thắng") |
| `M_FSM.txt` | M_MISIM (bitwise như bundle) |
| `M_GSM.txt` | **GIP miRNA** tính từ ma trận nhị phân đầy đủ 411×271, công thức `Gauss_M` của mã phát hành |
| `D_SSM1.txt` | D_MESH |
| `D_SSM2.txt` | biến thể V-a: bản sao D_MESH · **V-b: GIP bệnh** từ ma trận đầy đủ |

Kèm theo: manifest SHA256 và thống kê (density, số cặp/loại sau ghi đè).

### P3 — Port tối thiểu 5 loại (≈ 1–2 giờ)
- Từ `06f9ba3`: thêm nhánh dataset 411×271; số loại = 5; trọng số lớp lấy từ số đếm 5 loại; predictor đầu ra existence + 5; ánh xạ nhãn 1..5.
- **Evaluator giữ nguyên**. Thêm một hook chỉ để *ghi* (không thay đổi tính toán) cho Top-1 K=5 đã sửa, để tham khảo.
- Biến thể chia **theo triplet** (cho H4): fold theo triplet đúng mô tả CV_triplet, mẫu âm 1:1. Cờ riêng, mặc định tắt.
- Kiểm thử trước khi train: shape, mapping nhãn, cache hypergraph theo dataset (tránh cache chéo v2.0/v3.2).

### P4 — Ma trận thí nghiệm (5 fold, seed 1234, 650 epoch trừ khi ghi khác)

| Run | View | Config | Mục đích |
|---|---|---|---|
| E0 | như r3 (bản sao) | mặc định phát hành | Tách hiệu ứng config với r3 (H3) |
| **E1** | V-a: GIP miRNA + D_MESH×2 | mặc định phát hành | Kiểm H2 + H1 (run chính) |
| **E2** | V-b: GIP miRNA + GIP bệnh | mặc định phát hành | Biến thể H2 |
| E3 | view tốt nhất E1/E2 | 4 head (theo paper text) | Độ nhạy head |
| E4 | view tốt nhất | chia theo triplet | Kiểm H4 (Table 4) |
| E5 | cấu hình tốt nhất | seed {0, 42} thêm | Độ ổn định (báo mean ± std) |

**Ước tính thời gian:**
- Trên GPU: khoảng 6–12 phút/fold, tức 30–60 phút/run và **~4–7 giờ** cho cả 7 run.
- Nếu phải chạy CPU (i5-9300H): có thể **20–35 giờ**. Khi đó đề xuất chạy E1, E2, E4 trước rồi xin duyệt tiếp.

**Stop rule:**
- Nếu E1 và E2 đều cho AUC < 0,89 thì dừng, báo cáo, chưa chạy E3–E5.
- Nếu một run đã khớp tiêu chí ở mục 1 thì chỉ chạy thêm E5 để xác nhận.

### P5 — Báo cáo (≈ 1 giờ)
- `paper_v32_repro/REPORT.md` gồm 2 bảng:
  - Bảng A: paper vs từng run, Δ theo từng chỉ số.
  - Bảng B: Top-1 K=5 đã sửa (để thấy năng lực phân loại thật).
- Kết luận H1–H4 ở dạng: xác nhận / bác bỏ / chưa đủ chứng cứ.
- Nếu H1 được xác nhận, phát biểu đúng mức: "số 0,86 của paper khớp với chỉ số existence nhị phân mà code tác giả in dưới nhãn CV_type; năng lực phân loại loại thực tế là X".

## 6. Rủi ro & giới hạn

- GPU/driver không tương thích, phải chạy CPU nên tốn thời gian.
- D_SSM2 gốc của tác giả (ngữ nghĩa kiểu 2) cho v3.2 không có sẵn; V-a/V-b chỉ là xấp xỉ. Các view sequence/gene tự dựng (`faithful_repro/ext/seqgene`) không có trên máy này, chỉ còn SHA.
- Chưa biết tác giả chạy bao nhiêu seed hay chọn fold nào. Ngưỡng max-F1 trên test làm số liệu có độ lệch dương; đó là đặc điểm của tác giả và được giữ nguyên.
- Kể cả khi khớp số, đây là tái hiện **cách tính** của paper, không phải bằng chứng mô hình phân loại loại liên kết đạt 0,86.

## 7. Cần anh duyệt

1. Đồng ý cài môi trường P0 (~3 GB, `.venv` trong worktree) với thứ tự thử cu118 rồi CPU?
2. Đồng ý ma trận E0–E5 và stop rule?
3. Nếu chỉ có CPU: chạy đủ E0–E5 (~1–1,5 ngày máy) hay chỉ E1, E2, E4 trước?
