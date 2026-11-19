# THESIS_RESULTS_AND_LIMITATIONS — kết quả khoa học và giới hạn (báo cáo tiếng Việt)

## Kết quả chính

**1. Verdict tái hiện: PARTIAL_REPRODUCTION_WITH_EXTERNAL_BLOCKERS.**

Không tái hiện được F1 = 0.8600 của paper. Trên tensor strict
paper-first (411×271×5, đã thắt provenance), dưới PAIR_CVTYPE sạch
leakage với metric đã khóa, U1 đạt harmonic Top-1 ≈ **0.4558
(dynamic) / 0.4924 (static)**. Đây là con số trung thực nhất có thể
đạt trên dữ liệu công khai hiện nay — được ghi nhãn
`OUR_INTERPRETATION`, không tuyên bố tương đương paper.

**2. Metric release tác giả không chấm được v3.2.**

`OptimizedMetricCalculator.compute_top1_metrics` hardcode 4 loại:
bỏ hết ground-truth type-5 và dự đoán length-6 → Top-1 F1 = 0.0
trên mọi mô hình v3.2, kể cả predictor hoàn hảo. Mọi kết quả
"v3.2 collapse" trong lịch sử repo là artifact metric, không phải
kiến trúc. Đã chứng minh byte-level (fixture K4) + trên dữ liệu
thật.

**3. Định nghĩa metric di chuyển kết quả ~0.05.**

Cùng một bộ logits: harmonic-of-aggregates 0.4558 vs macro-F1
0.4045 (dynamic). Fixture chứng minh hai công thức khác nhau về
mặt toán học (counterexample 0.2154 vs 0.0971). Không một con số
"F1" nào tồn tại độc lập với convention.

**4. Split semantics không tương đương.**

CVtype (pair-fold) và CVtriplet (triplet-fold) là hai phân hoạch
khác nhau của cùng tập dữ liệu — fixture chứng minh không phải
bijection. Triplet-fold straddle pairs → không dùng được cho câu
hỏi CVtype. Errata E-1 vô hiệu protocol cũ.

**5. Mô hình học gì trên task chuẩn.**

Count priors (log1p(mi)+log1p(dis)−log1p(global)) đạt harmonic
0.448 / micro 0.539 / any-hit 0.631 — cạnh tranh mạnh với U1 trên
metric aggregate. Lợi thế U1 tập trung ở macro-F1 (lớp thiểu số).
Static ≥ dynamic 14/15 ô paired — refresh machinery không cải
thiện trong cấu hình của ta (không suy rộng sang paper).

**6. Đảo chiều ablation v2.0 là finding thật.**

Ablation w/o CL, w/o HGT đánh bại baseline trên shipped v2.0 —
giữ nguyên sau 4 lần verify (additive + rebuild + multi-seed + 2
loss modes). Hướng ngược paper Fig. 4.

**7. Data filter của paper không reverse-engineer được.**

SPLD filter `<3 pairs` cho 11,748 links — khớp số, nhưng paper viết
`<2`, và stratum thực tế là 10 miRNA <3 + 2 miRNA <2 + 1 disease
<3 → cả hai đọc literal đều không tái lập bộ lọc. UNRESOLVED.

## Giới hạn

- **Retrospective, không blind:** mọi fold bundle đều đã qua vòng
  phát triển; fold0 test bị nhìn sớm; horizon 650ep chọn hậu
  nghiệm. Không p-value, không tuyên bố confirmatory.
- **Density khác paper:** tensor strict ~4.6% vs 10.5% paper — bài
  toán khó hơn; 0.49 không nằm trên cùng phân phối với 0.86.
- **Label space khác:** ta chấm 5 loại multi-hot, release chấm 4
  loại scalar — không có bridge trực tiếp.
- **Provenance similarity đã thắt nhưng khác pipeline:** 4 view
  rebuild độc lập, hợp lệ về leakage nhưng không byte-identical
  với feature paper.
- **Provenance drift đã có và được ghi nhận** (crash artifacts,
  SHA drift sau sửa baseline) — lịch sử giữ nguyên.
- **Adaptation không tránh khỏi:** exist head, triplet scoring,
  multi-label loss là của ta — paper không đủ spec để "author-
  faithful" cho v3.2.
- **Không suy ra nguyên nhân gap:** G1–G4 đều chưa cô lập; nói
  "gap do X" là overclaim — chỉ liệt kê blocker theo tác động.

## Phát biểu dùng được trong luận văn / ReScience note

- "Trên bản tái cấu trúc strict của tensor paper-first, mô hình
  họ DHGCMDA đạt F1 hài hòa Top-1 ≈ 0.46–0.49 dưới CVtype
  pair-disjoint sạch leakage — đáng kể dưới 0.8600 của paper, với
  phần gap còn lại nằm ở các artifact chưa public (tensor, fold,
  eval code) thay vì sai lệch kiến trúc đã đo được."
- "Metric đính kèm release không chấm được tập v3.2 của chính
  tác giả — evidence của một release-eval disconnect."
- "Priors tần suất đạt tới ~90% kết quả model trên aggregate
  metric — gợi ý task pair-fold mang degree prior mạnh, và
  nghi ngờ lợi thế contrastive ở tầng lớp thiểu số."


---

> **[ERRATUM 2026-11-19, REV4]** Mọi phát biểu "density ~4.6% vs 10.5%" trong file này đã RETRACTED — strict tensor đo thực tế 10.5476% typed-link (khớp paper). Xem `DENSITY_ERRATUM.md`.
