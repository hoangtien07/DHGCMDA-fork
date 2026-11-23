# CHIẾN LƯỢC LUẬN VĂN — Hiểu, tái hiện và cải tiến DHGCMDA

Ngày: 2026-11-23 · Trạng thái: **CHỜ DUYỆT** (chưa cài đặt, chưa huấn luyện gì)
Thay thế: `docs/plan_v32_paper_repro.md` (bản đó chỉ phủ v3.2 và thiếu bước hiệu chuẩn v2.0).

---

## 0. Kết luận ngắn

1. **Bảng 4 của paper (gọi là "CV_triplet") thực chất là khối chỉ số nhị phân** mà mã phát hành in ra sau khi chia fold theo cặp. Mã gốc chạy v2.0 cho AUC 0,9738 / AUPR 0,9671 / F1 0,9295. Paper ghi AUPR 0,9738 / AUC 0,9669 / F1 0,9278. Hai bộ số trùng nhau nếu coi **hai cột AUC và AUPR trong paper bị đổi chỗ**.
2. **Bảng 3 hàng v3.2 (0,7915 / 0,9421 / 0,8600) nhiều khả năng cũng là chỉ số nhị phân** (Precision / Recall / F1 của bài toán "có liên kết hay không"). Lý do: hàm Top-1 phát hành trả 0 với 5 loại, và bộ số này bất khả thi với mọi cách chấm theo loại nhưng khả thi với chấm nhị phân. Đây vẫn là giả thuyết, chưa có lần chạy xác nhận.
3. **v2.0 là mỏ neo hiệu chuẩn.** Cần xác định cấu hình mã gốc khớp paper trên v2.0, rồi áp nguyên cấu hình đó cho v3.2 mà không tinh chỉnh thêm. Làm vậy thì kết quả v3.2 là một phép kiểm định, không phải "chỉnh tới khi khớp".
4. **Cải tiến mô hình phải đo trên một giao thức sạch, cùng quy ước với các baseline.** Hàng baseline trong Bảng 3 dùng quy ước TDRC (đã khớp SPLD tới 4 chữ số). Dưới quy ước đó, DHGCMDA trên v3.2 chỉ ở mức 0,3–0,5, thấp hơn SPLD (0,53). Đây chính là dư địa cải tiến thật.

---

## 1. Review độc lập quy trình đã làm

### 1.1. Sai lầm của quy trình trước (Round 1 → P8.1)

| # | Sai lầm | Hệ quả |
|---|---|---|
| 1 | Viết lại mô hình theo phương trình thay vì bám mã phát hành | v2.0 chỉ đạt 0,388, trong khi mã gốc đạt 0,55–0,60 |
| 2 | Không dùng v2.0 làm mốc hiệu chuẩn để hiểu paper lấy số từ khối in nào của code | Không phát hiện Bảng 4 là khối nhị phân, cũng không phát hiện AUC/AUPR đổi cột |
| 3 | Kết luận "bế tắc vĩnh viễn" từ định lý bất khả thi | Định lý đúng, nhưng chỉ loại các cách chấm theo loại. Chưa ai thử đọc bộ số như chỉ số nhị phân |
| 4 | Lần chạy duy nhất của mã gốc trên đúng data paper (`r3_leaky_spld_exact`) lệch mã gốc 4 điểm: mất view GIP, 300 thay vì 650 epoch, 4 thay vì 8 head, full_bilinear | AUC 0,878 bị hiểu là "không tái hiện được" |
| 5 | Mục tiêu trôi sang kiểm toán rò rỉ và macro-F1 đa nhãn (P5, G7, P8) | Ba tháng không tiến gần paper, và cũng chưa có đề xuất cải tiến nào được xác nhận |
| 6 | Quy trình nặng thủ tục (gate, errata, "0 optimizer step") | Bản port 5 loại đã thiết kế nhưng không được phép chạy |

### 1.2. Sai sót của chính tôi trong đoạn chat này

- **Lượt 1:** tôi kết luận "0,86 không thể tái hiện nếu thiếu artifact tác giả". Kết luận này quá sớm, vì tôi dựa vào định lý Track R mà chưa tự đọc vòng đánh giá của mã gốc.
- **Lượt 3 (plan cũ):**
  - Thiếu bước hiệu chuẩn v2.0.
  - Ma trận E0–E5 thử nhiều biến thể ngay trên v3.2, tức có nguy cơ chọn biến thể theo đích.
  - Khẳng định `M_GSM` là GIP khi chưa kiểm. Nay đã kiểm thống kê: ma trận dày, không có phần tử 0, giá trị dạng hàm mũ, phù hợp với GIP.
  - Không có phần cải tiến mô hình.
- **Phần còn giá trị:** các hướng cải tiến có bằng chứng sơ bộ (mục 5), bộ dữ liệu MDAv3.2-3 đã khôi phục đúng, kiểm toán rò rỉ, và định lý bất khả thi (dùng làm bằng chứng rằng Bảng 3 v3.2 không phải Top-1).

---

## 2. Bằng chứng đã có (đọc code, log, paper; chưa chạy mới)

| ID | Bằng chứng | Nguồn |
|---|---|---|
| B1 | Paper Fig. 3 tự gọi hai nhóm số là "binary classification metrics (AUC, AUPR, F1)" và "Top-1 metrics". Đây đúng là hai khối mà mã gốc in ra | `_pdf_text/p26.txt` |
| B2 | Mã gốc v2.0 (Phase A): AUC 0,9738 / AUPR 0,9671 / F1 0,9295. Paper Bảng 4 v2.0: AUPR 0,9738 / AUC 0,9669 / F1 0,9278 | `results/plan_c_comparison.json` |
| B3 | Mã gốc không có chia fold theo triplet. Chỉ có chia theo cặp, 1:1 mẫu âm, ngưỡng max-F1 chọn trên test, lặp 5 seed lấy mẫu âm | `main_experiments_hetero1.py` @ `06f9ba3` |
| B4 | `compute_top1_metrics` chỉ nhận 4 loại nên v3.2 ra 0. Khối "CV_type Style Metrics Details" in Precision / Recall / F1 nhị phân | `Calculate_Metrics.py` @ `06f9ba3` |
| B5 | Cặp (P = 0,7915, R = 0,9421) bất khả thi với mọi cách chấm theo loại trên tensor 8.735 cặp. Với chấm nhị phân 1:1 thì khả thi (specificity ≈ 0,75) | Track R + tính lại |
| B6 | Chỉ số nhị phân các lần chạy v3.2 cũ: P 0,756–0,790, R 0,861–0,869, F1 0,805–0,827, AUC 0,878–0,903 | `logs/r3_leaky_spld_exact.log`, `v32_wang_correct_*.log` |
| B7 | View mã gốc v2.0: miRNA = `M_FSM` (chức năng) + `M_GSM` (GIP); bệnh = `D_SSM1` + `D_SSM2` (hai loại ngữ nghĩa). Chữ "Sequence/Gene" trong code chỉ là nhãn in | `prepareData.py`, `介绍.txt`, thống kê file |
| B8 | Lần chạy v3.2 có similarity GIP (`v3.2_processed`, 3 fold) đạt AUC 0,9217. Lần không có GIP đạt 0,878–0,903 | `results/v32_baseline_partial.json` |
| B9 | Hàng SPLDHyperAWNTF của Bảng 3 (0,6219 / 0,4624) tái tạo khớp 4 chữ số trên MDAv3.2-3 theo quy ước TDRC (P = đúng/số cặp, R = đúng/số triplet) | `forensics/REPRO_LEDGER.md` F20 |
| B10 | Trên v2.0 không có cấu hình đơn lẻ nào khớp cả hai bảng: `exist_weight` 0,3 khớp Bảng 4 (Top-1 = 0,549); `exist_weight` 0,1 khớp Bảng 3 (Top-1 = 0,600) nhưng AUC tụt còn 0,964 | `plan_c_comparison.json` |

**Diễn giải B10:** paper có thể báo mỗi bảng từ một lần chạy khác nhau. Điều này cũng giải thích vì sao hai F1 của v3.2 khác nhau (0,8600 và 0,8674).

### Bản đồ "số trong paper ← nguồn trong code" (giả thuyết làm việc)

| Số trong paper | Nguồn đề xuất | Độ chắc |
|---|---|---|
| Bảng 4, v2.0 | Khối nhị phân, chia theo cặp, AUC↔AUPR đổi cột | Cao (B2) |
| Bảng 3, v2.0 | Khối Top-1 (accuracy + macro-recall), có thể từ cấu hình khác Bảng 4 | Cao (trong ±7%) |
| Bảng 4, v3.2 | Khối nhị phân của lần chạy v3.2, AUC↔AUPR đổi cột | Trung bình–cao |
| Bảng 3, v3.2 | Precision / Recall / F1 nhị phân của một lần chạy v3.2 khác | Trung bình (cần kiểm định) |
| Bảng 3, baseline | Quy ước TDRC trên MDAv3.2-3 | Cao với SPLD (B9), chưa kiểm các hàng khác |

---

## 3. Mục tiêu và tiêu chí

| Trụ | Mục tiêu | Tiêu chí đạt |
|---|---|---|
| **A. Tái hiện** | Sinh lại 4 hàng số của paper (v2.0 và v3.2, Bảng 3 và 4) bằng mã gốc | Lệch ≤ 0,02 với AUC / AUPR / F1; ≤ 0,03 với P / R |
| **B. Hiểu đúng** | Đo năng lực thật của DHGCMDA theo cùng quy ước với baseline; lượng hóa phần điểm đến từ rò rỉ | Bảng phân rã có số, mỗi yếu tố bật/tắt riêng |
| **C. Cải tiến** | Biến thể tốt hơn DHGCMDA gốc và các baseline tái tạo được, trên giao thức sạch | Thắng có ý nghĩa thống kê (paired, ≥ 3 seed × 5 fold) trên cả v2.0 và v3.2 |

---

## 4. Kế hoạch chi tiết

### Giai đoạn 0 — Môi trường (30–45 phút, tải ~3 GB)
- `uv venv .venv --python 3.12` trong worktree (đã có trong `.gitignore`).
- `torch==2.5.1` (đúng bản paper). Thử bản cu118 cho GTX 1650 (driver 462.30); nếu lỗi thì dùng bản CPU.
- `torch-geometric`, numpy, pandas, scipy, scikit-learn, openpyxl, xlrd.
- Mọi code mới đặt trong `paper_repro/`. Không sửa file gốc.

### Giai đoạn 1 — Kiểm chứng tĩnh, không huấn luyện (2–3 giờ)
1. Trích mã gốc `06f9ba3` ra `paper_repro/release/` kèm hash từng file.
2. Đưa dự đoán giả 6 chiều và nhãn 1..5 vào evaluator gốc. Xác nhận Top-1 = 0 và khối nhị phân vẫn in bình thường.
3. Xác định `M_GSM` v2.0 là GIP của ma trận nào: 5.429 liên kết nhị phân hay 1.498 cặp có loại, và tham số γ. Cách làm: tính lại rồi so từng phần tử. Kết quả này quyết định công thức dựng GIP cho v3.2.
4. Xác định quan hệ `D_SSM1` / `D_SSM2` (ngữ nghĩa kiểu 1 và 2). So với `DSSM3.2_3.csv` và `mi_fun_sim_3.2_3.csv` của bundle MDAv3.2-3 để biết bundle đã có sẵn view nào.
5. Tính ngược từ số paper: với chấm nhị phân 1:1, bộ (P, R, F1) của Bảng 3 v3.2 có tồn tại nghiệm nguyên nhất quán theo cỡ fold không.

**Cổng G1:** nếu bước 3 cho thấy `M_GSM` không phải GIP của ma trận liên kết thì dừng và báo lại. Giả thuyết về view phải sửa.

### Giai đoạn 2 — Hiệu chuẩn trên v2.0 (3 lần chạy)
| Run | Cấu hình | Dự đoán ghi trước |
|---|---|---|
| C1 | Mã gốc nguyên bản, seed 1234 | Khối nhị phân ≈ AUC 0,974 / AUPR 0,967 / F1 0,93 (khớp Bảng 4 sau khi đổi cột) |
| C2 | Mã gốc + `exist_weight` 0,1 (thay đổi duy nhất) | Top-1 ≈ 0,58 / 0,63 / 0,60 (khớp Bảng 3) |
| C3 | Lặp lại C1 với seed khác | Đo độ dao động giữa các seed, dùng làm thước đo cho v3.2 |

Kết quả giai đoạn này là **cấu hình hiệu chuẩn** K* (cho Bảng 4) và T* (cho Bảng 3), được khóa trước khi đụng tới v3.2.

**Cổng G2:** nếu C1 lệch Bảng 4 quá 0,01 thì tìm nguyên nhân (phiên bản thư viện, cache) trước khi đi tiếp.

### Giai đoạn 3 — Kiểm định trên v3.2 (data paper MDAv3.2-3)
1. **Dựng data theo đúng công thức v2.0:**
   - `M_FSM` = chức năng (bundle).
   - `M_GSM` = GIP theo công thức đã xác định ở Giai đoạn 1.
   - `D_SSM1` = ngữ nghĩa (bundle).
   - `D_SSM2` = view thứ hai cho bệnh. Nguồn duy nhất được chọn theo kết quả bước 4 của Giai đoạn 1. Nếu bundle không có sẵn thì dùng bản sao của `D_SSM1` và ghi rõ.
   - File liên kết sắp theo mã loại tăng dần (để tái tạo đúng hành vi ghi đè nhãn).
2. **Port 5 loại tối thiểu:** kích thước 411×271, 5 loại, trọng số lớp, đầu ra existence + 5. Evaluator giữ nguyên. Mọi thay đổi ghi vào `PORT_DIFF.md`.
3. **Chạy đúng hai cấu hình đã khóa, không tinh chỉnh:**

| Run | Cấu hình | So với | Dự đoán ghi trước |
|---|---|---|---|
| V1 | K* | Bảng 4 v3.2 | AUC ≈ 0,927 / AUPR ≈ 0,918 / F1 ≈ 0,867 |
| V2 | T* | Bảng 3 v3.2 | Precision ≈ 0,79 / Recall ≈ 0,94 / F1 ≈ 0,86 (nhị phân) |
| V3 | Cấu hình khớp hơn, thêm 2 seed | — | Dao động giữa các seed |

**Cổng G3 — ba kịch bản:**
- **Khớp** (trong ngưỡng ở mục 3): trụ A xong. Chuyển Giai đoạn 4.
- **Gần** (AUC ≥ 0,90 nhưng Recall < 0,91): cho phép đúng **một** vòng chẩn đoán với danh sách đóng. Thử 4 head thay cho 8; thử thêm GIP bệnh làm view thứ hai. Sau vòng này dừng dù kết quả thế nào.
- **Xa** (AUC < 0,89): giả thuyết Bảng 3 là nhị phân bị bác. Ghi nhận trung thực. Trụ A dừng ở mức "v2.0 tái hiện, v3.2 chỉ tái hiện phần nhị phân ở mức X". Vẫn làm tiếp trụ B và C.

### Giai đoạn 4 — Đo năng lực thật và phân rã rò rỉ (trụ B)
Dùng lại dự đoán đã lưu ở Giai đoạn 2–3, chỉ huấn luyện thêm cho các biến thể tắt rò rỉ.

1. **Chấm lại theo nhiều quy ước** trên cùng dự đoán:
   - Top-1 kiểu TDRC (quy ước của baseline).
   - Top-1 kiểu mã gốc mở rộng 5 loại.
   - Macro-F1 đa nhãn.
   - Nhị phân.

   Kết quả là bảng "0,86 nghĩa là gì".
2. **Phân rã rò rỉ**, mỗi lần tắt một yếu tố:
   - GIP tính lại chỉ từ train.
   - Che nhãn test khỏi đặc trưng ghép `[S, A]`.
   - Chọn ngưỡng và epoch bằng tập validation thay cho test.
   - Giữ đa nhãn thay cho ghi đè.
3. **Baseline tái tạo cùng giao thức:** TDRC, SPLDHyperAWNTF (đã có code), cùng hai baseline đơn giản là prior theo lớp và logistic. Lý do: Track N cho thấy baseline đơn giản rất mạnh.

Kết quả: bảng so sánh công bằng duy nhất của luận văn. Bảng này cũng là **đường cơ sở cho trụ C**.

### Giai đoạn 5 — Cải tiến mô hình (trụ C)

**Giao thức đánh giá (khóa trước khi thử ý tưởng):**
- Dữ liệu: v2.0 (495×383) và MDAv3.2-3 (411×271).
- Chia: 5 fold theo cặp × 3 seed, fold cố định, dùng chung cho mọi phương pháp. Similarity phụ thuộc nhãn tính lại theo từng fold. Chọn epoch và ngưỡng bằng inner-validation.
- Chỉ số chính: Top-1 F1 kiểu TDRC. Chỉ số phụ: macro-F1 đa nhãn, AUC / AUPR.
- So sánh: paired theo fold × seed, báo trung bình ± độ lệch chuẩn và kiểm định dấu / Wilcoxon.
- Báo thêm một cột "theo giao thức tác giả" (có rò rỉ) để đối chiếu với paper.

**Ứng viên, xếp theo bằng chứng sơ bộ đã có** (mọi bằng chứng mới là thăm dò, phải xác nhận lại trên giao thức trên):

| Ưu tiên | Ý tưởng | Bằng chứng sơ bộ | Rủi ro |
|---|---|---|---|
| 1 | **Cân bằng gradient giữa các loss**: reconstruction dùng mean thay sum, sau đó mở rộng thành trọng số học được (uncertainty weighting / GradNorm) | Gradient reconstruction lớn gấp ~10⁵ lần gradient loss loại; đổi sang mean tăng macro-F1 +0,04 trên 4/4 fold; v2.0 từ 0,388 lên 0,518 | Thấp |
| 2 | **Decoder full bilinear** (ma trận d×d riêng cho từng loại) | v2.0 +5,9%; v3.2 +23% tương đối | Thấp |
| 3 | **Tinh giản kiến trúc**: bỏ hoặc giảm contrastive learning và HGT | Trên v2.0, bỏ CL +4,5%, bỏ HGT +7,0% (4 seed, 8/8 dương); bản dựng lại thật +8–10% | Trung bình: cần kiểm trên v3.2 |
| 4 | **Huấn luyện và giải mã đa nhãn** cho v3.2 (24% cặp có nhiều loại; ghi đè làm mất 26% triplet) | Mất nhãn đã được đo; nhưng BCE thuần kém softmax (A2 −0,035) | Trung bình: cần thiết kế loss cẩn thận |
| 5 | **Ensemble nhiều seed** | 4 seed: Top-1 0,69 trên chia cố định | Thấp, nhưng đóng góp khoa học nhỏ |

**Trình tự:**
1. Ablation cộng dồn: gốc → +1 → +1+2 → +1+2+3.
2. Chọn tổ hợp bằng inner-validation.
3. Chạy xác nhận một lần trên test.
4. Ý tưởng 4 chỉ thử trên v3.2.

**Mô hình đề xuất dự kiến:** "DHGCMDA cân bằng gradient + decoder bilinear + kiến trúc tinh giản".

**Tiêu chí thành công:** thắng DHGCMDA gốc và baseline mạnh nhất tái tạo được, trên cả hai bộ dữ liệu, theo chỉ số chính.

**Tiêu chí dừng:** nếu sau bước 1–3 không thắng baseline đơn giản (logistic / prior) trên v3.2 thì báo cáo trung thực là kết quả âm, và chuyển trọng tâm đóng góp sang trụ A và B.

### Giai đoạn 6 — Cấu trúc luận văn
1. Tổng quan và phương pháp DHGCMDA.
2. Tái hiện: v2.0 khớp; v3.2 theo cách tính của tác giả; bản đồ số ↔ code; lỗi đổi cột và nhãn chỉ số.
3. Đánh giá lại: năng lực thật theo quy ước baseline, phân rã rò rỉ.
4. Cải tiến: mô hình đề xuất, ablation, so sánh.
5. Hạn chế và hướng phát triển.

---

## 5. Ước tính thời gian

| Giai đoạn | Việc tay | Thời gian máy (GPU ước tính / CPU) |
|---|---|---|
| 0 | 0,5 giờ | — |
| 1 | 2–3 giờ | — |
| 2 (3 lần chạy v2.0) | 0,5 giờ | 1,5–3 giờ / 4–6 giờ |
| 3 (3–5 lần chạy v3.2) | 3–4 giờ | 3–6 giờ / 10–25 giờ |
| 4 | 4–6 giờ | 4–8 giờ / 15–30 giờ |
| 5 | 2–3 ngày | 1–2 ngày / 4–7 ngày |

Căn cứ: các lần chạy CPU cũ mất 8–14 phút/fold với v2.0 (650 epoch) và 10–28 phút/fold với v3.2 (300 epoch). Tốc độ GPU chưa đo trên máy này.

---

## 6. Rủi ro

- **Driver GPU cũ (CUDA 11.2):** nếu bản cu118 không chạy thì phải dùng CPU. Giai đoạn 5 khi đó nên chạy trên Colab hoặc VM.
- **`D_SSM2` cho v3.2 không có bản gốc của tác giả:** phải xấp xỉ, và đây là nguồn sai lệch khó loại bỏ nhất ở Giai đoạn 3.
- **Giả thuyết Bảng 3 là nhị phân có thể sai:** đã có kịch bản "Xa" ở cổng G3. Trụ B và C không phụ thuộc vào giả thuyết này.
- **Bằng chứng cho các ý tưởng cải tiến đều là thăm dò**, đo trên fold đã lộ hoặc pipeline có rò rỉ. Mức cải thiện có thể co lại trên giao thức sạch.
- **Nếu giả thuyết đúng,** luận văn sẽ phải viết rằng 0,86 là chỉ số nhị phân, không phải năng lực phân loại loại liên kết. Cần cách diễn đạt trung tính: "nhãn chỉ số trong paper không khớp với cách tính trong mã phát hành".

---

## 7. Cần anh quyết định

1. Duyệt cài môi trường ở Giai đoạn 0 (~3 GB, `.venv` trong worktree; thử GPU trước, lỗi thì CPU)?
2. Duyệt thứ tự **hiệu chuẩn v2.0 trước, rồi kiểm định v3.2 không tinh chỉnh**, kèm các cổng G1–G3?
3. Chỉ số chính cho phần cải tiến: Top-1 F1 kiểu TDRC (đề xuất) hay macro-F1 đa nhãn?
4. Giai đoạn 5 chạy trên máy này hay trên Colab / VM anh đang có?
