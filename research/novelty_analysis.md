# Novelty Analysis — ứng viên đóng góp (SESSION 5, 2026-07-05)

> Áp R7: mỗi ứng viên chạy ≥3 truy vấn phủ định "đã ai làm chưa". Novelty = *có bằng chứng search phủ định*, KHÔNG phải "model không nhớ ra".
> Kết quả: **không ứng viên nào bị giết**, nhưng đều có **prior art kề cận phải trích dẫn & phân biệt**. Đây là tín hiệu lành mạnh (R7 hoạt động).

---

## 1. PRIOR ART KỀ CẬN PHÁT HIỆN QUA R7 (phải cite & phân biệt)

| Bài kề cận | Cái nó ĐÃ làm | Cái nó CHƯA làm (khe của ta) | Trạng thái |
|---|---|---|---|
| **"A consistent evaluation of miRNA-disease association prediction models"** (bioRxiv 2020.05.04.075754) | Chỉ ra **data leakage** từ similarity precomputed trên toàn mạng; released common framework so 12 model **BINARY** | KHÔNG đụng **type prediction**, KHÔNG đụng DHGCMDA, KHÔNG metric-bug/predictor-faithfulness/K-sparsity | ⚠ **prior art nguy hiểm nhất cho angle A** — [SH] biorxiv chặn bot, PHẢI verify Session 6 |
| **Benchmark of computational methods for MDA** (PMC6781296) + "36 methods, HMDD v3.1" | Benchmark **binary** MDA quy mô lớn | KHÔNG phải type prediction; không audit code/metric | [SH] cần fetch nếu dùng |
| **MassSpecGym in the Wild** (arXiv 2606.19624) | *"implementation-level divergence in metric computation can be sufficient to reorder leaderboard rankings on identical predictions"* — molecule discovery | Khác domain; **precedent hoàn hảo** cho finding metric-bug của ta | [SH] cần verify (arXiv ID 2606 nghi — kiểm ID) |
| **PROXI: Challenging the GNNs for Link Prediction** (arXiv 2410.01802, 2024) + **Reconsidering GAE in Link Prediction** (2411.03845) | Cho thấy method đơn giản sánh/vượt GNN cho link prediction (general) | KHÔNG phải MDA/hypergraph/type | precedent cho angle B ("simplification wins") |
| **Untrained Message Passing** (VERIFIED, 2406.16687) | Bỏ tham số MP vẫn competitive/superior | general link pred, không MDA type | anchor angle B |
| Hypergraph-KNN MDA có sẵn (HFHLMDA, MSCHLMDA, MSCHLMDA…) | Dùng KNN param K để dựng hyperedge trong MDA | KHÔNG ai nghiên cứu **K-sensitivity/over-param như ĐÓNG GÓP**, không "sparse thắng dense" cho type | narrows G2, không giết |

**Không tìm thấy prior art cho:** (i) metric-bug hardcode #types trong code MDA phát hành; (ii) diagonal-vs-full bilinear faithfulness trong MDA; (iii) uncertainty/conformal cho MDA **type** prediction; (iv) LLM/foundation model cho MDA (chưa có bài nào).

---

## 2. BẢNG CHẤM ỨNG VIÊN (thang 1–5; Risk cao = tệ)

| ID | Ứng viên | Novelty | Khả thi với repo | Bằng chứng | Risk | Venue tiềm năng |
|---|---|:--:|:--:|:--:|:--:|---|
| **A-metric** | Metric bug: hardcode 4 types → bỏ 100% mẫu v3.2 → Top-1=0.0 giả | **4.5** | **5** (đã chứng minh 3 cách) | **5** | **1.5** | ReScience / GigaScience / Brief Bioinf (correspondence) |
| **A-pred** | Predictor faithfulness: diagonal (degenerate) → full bilinear +6.4% | 4 | **5** (multi-seed) | **5** | 2.5 | ↑ (cùng bài A) |
| **A-K** | K-sparsity right-sizing: K=3 (thưa) +15.3% vs paper, K=13 tệ nhất | 3.5 | **5** | **5** (paired t, adversarial) | 3 | ↑ / methods venue |
| **A-abl** | Ablation reversal: bỏ HGCN/HGT/CL +8–11% (over-param) | 3.5 | **5** (5 cách verify) | **5** | 3 | ↑ |
| **A-data** | v3.2 0.86 không tái hiện (curated 411×271 ẩn) + baseline lệch có hệ thống (O2) | 3 | **5** | 4 | 3 | ↑ |
| **B-bundle** | "Right-sizing hypergraph GNN cho MDA thưa" (gộp A-pred+A-K+A-abl thành methods insight) | 3.5 | 4 (cần +1-2 dataset MDA khác) | 4 | 3 | Brief Bioinf / BMC Bioinf / Bioinformatics |
| **C-uq** | Uncertainty/conformal-aware type prediction cho MDA | **4.5** | 2.5 (thí nghiệm MỚI, additive được) | 1.5 (chưa làm) | 4 | (đột phá — rủi ro) |
| **C-llm** | LLM/foundation-model-augmented MDA | 5 | 1.5 (ngoài repo) | 1 | 4.5 | (không khuyến nghị lần này) |

---

## 3. XẾP HẠNG & KHUYẾN NGHỊ ĐÓNG GÓI

### 🥇 Hạng 1 — **Bài A (reproducibility/critique)** = combo A-metric + A-pred + A-K + A-abl + A-data
- **Vì sao:** khả thi cao nhất (ta ĐÃ có toàn bộ kết quả, kiểm định đối kháng), bằng chứng mạnh nhất, novelty đủ (4 sub-finding không có prior art trực tiếp). Rủi ro thấp nhất.
- **Đóng góp trục:** (1) **metric bug** trong code phát hành làm sập đánh giá v3.2 (instance cụ thể của Kapoor taxonomy + precedent MassSpecGym); (2) **predictor faithfulness** — reimplementation trung thực (full bilinear) vượt cả bản gốc; (3) **K-sparsity + ablation reversal** — DHGCMDA over-parameterized cho MDA thưa (nền: Oversmoothing + Untrained MP); (4) **v3.2 không tái hiện + baseline lệch có hệ thống** — thiếu chuẩn hoá benchmark (nền: Kapoor + consistent-eval-2020).
- **Headline số:** v2.0 Top-1 F1 0.5970(paper) → **0.688±0.011** (ta, faithful reimpl) — VƯỢT mọi public baseline.
- **Venue:** ReScience C (đúng format replication) HOẶC Briefings in Bioinformatics/GigaScience (critique + benchmark). ReScience = rủi ro thấp, uy tín reproducibility; Brief Bioinf = impact cao hơn, khó hơn.

### 🥈 Hạng 2 — **Bài B (methods insight)**, NẾU muốn tách một bài venue mạnh hơn
- Gộp A-pred + A-K + A-abl thành luận điểm "right-sizing hypergraph GNN cho đồ thị sinh học nhỏ/thưa; khi nào CL/transformer là nhiễu".
- **Cần bổ sung:** chạy trên ≥1-2 dataset MDA khác (hoặc lncRNA/circRNA-disease) để cho thấy finding tổng quát, không riêng HMDD v2.0. → thí nghiệm additive, map về K-sweep + ablation sẵn có.
- Precedent venue: PROXI, GAE-reconsidered, Untrained MP (đều được nhận) → có "khẩu vị" cho negative/simplification result.

### 🥉 Hạng 3 — **C-uq (uncertainty-aware type prediction)** = hướng đột phá, để "future work" hoặc bài 2
- Novelty cao (không thấy prior MDA-type). Conformal prediction post-hoc trên score model sẵn có của ta → **feasible additive** nhưng là công việc MỚI, chưa có kết quả. Rủi ro cao hơn. Khuyến nghị: nêu như extension trong Bài A, theo đuổi sau nếu Bài A ổn.

### ❌ KHÔNG khuyến nghị lần này — C-llm
Ngoài repo, không kết quả, không khớp thế mạnh critique. Chỉ nhắc trong "future directions".

---

## 4. PHA ĐỐI KHÁNG — reviewer giả cố bác từng novelty claim (R6)

| Claim | Reviewer công kích | Phản biện của ta (bằng chứng) |
|---|---|---|
| Metric bug là đóng góp | "Chỉ là 1 bug lặt vặt 1 repo, không đáng đăng" | (a) Bug làm **sập hoàn toàn** đánh giá v3.2 (0.0 giả) → ảnh hưởng kết luận khoa học; (b) precedent MassSpecGym/Kapoor cho thấy metric-audit publishable; (c) ta chứng minh 3 cách độc lập (synthetic-perfect, tương đương 4-type, đo lại) |
| Full bilinear "vượt paper" | "Anh chỉ tune predictor, không phải lỗi paper" | Code phát hành dùng bilinear ĐƯỜNG CHÉO (degenerate, SimplE cho thấy kém expressive) — reimplementation TRUNG THỰC với mô tả "bilinear" của paper lại tốt hơn → faithfulness fix, không phải tuning |
| K=3 chỉ là hyperparameter tuning | "Tuning K không phải khoa học" | K=13 (default paper) là K **tệ nhất** trên diag; full_bilinear thừa hưởng K=13 do tình cờ; finding = **hệ thống** (monotone K13<K7<K3) + cơ chế (over-smoothing) + kiểm định (paired t=3.62, 3 reviewer đối kháng refuted=false) |
| Ablation reversal = noise | "Single-seed noise" | 5 cách verify độc lập (additive + true rebuild + 4-seed + 2 loss mode + full_bilinear); 8/8 delta dương; nền Untrained-MP/Oversmoothing |
| v3.2 không tái hiện = anh làm sai | "Anh thiếu kỹ năng, không phải paper có vấn đề" | (a) TDRC reproduce ~98% cùng pipeline → kỹ năng OK; (b) không dòng phương pháp nào (tensor/neural) đạt gần 0.86 trên public data; (c) baseline lệch có hệ thống (O2) là vấn đề lĩnh vực (Kapoor) |
| "Consistent-eval-2020 đã làm rồi" | "Đã có bài audit MDA 2020" | Bài đó **binary + similarity-leakage**; ta **type prediction + metric-bug + predictor-faithfulness + K/over-param** — trục hoàn toàn khác, ta cite & mở rộng |

---

## 5. SELF-AUDIT (R17) SESSION 5
- Web search: **8** (R7 phủ định). Fetch: 0 (session suy luận; các prior-art kề cận để verify ở Session 6).
- Ứng viên bị GIẾT bởi prior art: **0**. Ứng viên bị THU HẸP (có prior art kề cận): 3 (A-data, A-K, B).
- Prior art mới cần verify (SEARCH-HIT): consistent-eval-2020, benchmark-36-methods, MassSpecGym, PROXI, GAE-reconsidered.

## 6. Provenance & Limitations
- **consistent-eval-2020 (bioRxiv)** là rủi ro novelty lớn nhất cho Bài A — biorxiv chặn bot 3 lần; **BẮT BUỘC verify** (mirror/published version) trước khi nộp, để phân biệt chính xác.
- MassSpecGym arXiv ID "2606.19624" NGHI (định dạng 2606 = 2026-06) — kiểm ID thật trước khi cite.
- Điểm chấm là **định tính** (đánh giá của trợ lý), không phải phép đo — user cân nhắc lại.
- Chưa verify số Top-1 của TFLP/SPLDHyperAWNTF (SH) — nếu Bài A liệt kê bảng SOTA đầy đủ, cần fetch.
- C-uq/C-llm: novelty dựa trên "không tìm thấy trong ~10 search" — CHƯA đủ mạnh để tuyên bố "chưa ai làm"; cần R7 sâu hơn nếu theo đuổi.
