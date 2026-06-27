# Plan L — Council Synthesis: Tái hiện + Cải thiện DHGCMDA (Linux)

## 1. Tóm tắt điều hành

Plan L chuyển toàn bộ pipeline sang Linux và chạy một "council" (ma trận thí nghiệm + 3 thẩm định viên đối kháng) để truy tìm cải thiện thực sự cho HMDD v2.0, đồng thời kiểm chứng lại các kết luận v3.2 / ablation / baseline.

**Headline (số trung thực, đã qua thẩm định đối kháng):**

> **v2.0 Top-1 F1 = 0.688 ± 0.011** (trung bình 3 seed, K_neigs=3 dưới `full_bilinear`), 95% CI ≈ **[0.662, 0.715]**.
> So với paper **0.5970** (+15.3%) và so với best cũ của dự án **0.6350** (+8.4%).

**Con số nào được phép công bố — 0.688 hay 0.7006?** → **Bắt buộc công bố 0.688 (multi-seed mean), KHÔNG được công bố 0.7006.** 0.7006 chỉ là seed 0 — seed may mắn nhất trong 3. Cả 3 thẩm định viên đối kháng đều cảnh báo riêng rằng báo cáo 0.7006 sẽ **lặp lại đúng lỗi "lucky seed" mà dự án đã từng mắc và tự ghi nhận** (seed=1 ở Plan refocus). Headline phòng thủ được = trung bình seed 0.688, kèm khoảng tin cậy. 0.7006 chỉ nên xuất hiện như một dòng trong bảng per-seed, không bao giờ đứng một mình.

**Verdict đối kháng:** cả 3 thẩm định viên độc lập đều trả về `refuted = false, confidence = high` cho mệnh đề "K=3 dưới full_bilinear cải thiện thật từ 0.6350 lên ~0.688". Gain vượt qua mọi confound đã test: không đánh đổi AUC, không phải seed luck (paired t = 3.62 cùng split; seed-paired t = 12.97), không leakage (K chỉ điều khiển KNN của hypergraph dựng từ **train-fold** matrix), không phải metric quirk, không phải overfitting (metric CV-test còn *tăng* ở K=3).

**Cảnh báo trung thực cốt lõi:** K=3 được tune trên *chính* 5-fold CV dùng để báo cáo (không có held-out riêng), nên 0.688 là ước lượng **in-sample đã tune** — generalization thật trên split mới có thể thấp hơn đôi chút. Yếu tố giảm nhẹ: gain replicate trên cả 3 seed *không* dùng cho selection, và seed tệ nhất (0.6808) vẫn vượt 0.6350.

---

## 2. Focus A — Đẩy v2.0: phát hiện K_neigs=3

### 2.1 Cơ chế

`K_neigs` điều khiển số neighbor của KNN khi dựng dual-view hypergraph. Trước Plan L, **K chưa bao giờ được tune dưới predictor `full_bilinear`** — baseline J-1 ngồi cố định ở **K=13**, vốn là giá trị K *tệ nhất* trên diag sweep cũ (0.5506 ở K=15, monotone smaller-K-better từ 0.5924@K3). Tức là baseline thừa hưởng một lựa chọn K không may.

Cơ chế của gain: dataset v2.0 thưa (1498 assoc / 189K cells). Hypergraph **thưa hơn (K nhỏ)** giảm over-smoothing / over-connection, khớp với bản chất "DHGCMDA over-parameterized cho v2.0" đã ghi nhận từ Plan E. Trend đơn điệu xác nhận cơ chế (cùng seed 1234):

| K_neigs | Top-1 F1 (seed 1234) | AUC |
|---|---:|---:|
| 13 (baseline) | 0.6311 | 0.9805 |
| 7 | 0.6538 | 0.9818 |
| **3** | **0.6808** | 0.9806 |

K=7 nằm gọn giữa K=13 và K=3 → smooth monotone, không phải spike đơn lẻ.

### 2.2 Bảng multi-seed (K=3, full_bilinear)

| Seed | Top-1 F1 | AUC | AUPR | F1 binary |
|---|---:|---:|---:|---:|
| 1234 | 0.6808 | 0.9806 | 0.9800 | 0.9355 |
| 0 | **0.7006** (lucky) | 0.9805 | 0.9783 | 0.9355 |
| 42 | 0.6835 | 0.9820 | 0.9806 | 0.9370 |
| **Mean ± std** | **0.6883 ± 0.0107** | 0.9810 | 0.9796 | 0.9360 |

Cả 3 seed đều vượt 0.6350; seed tệ nhất (0.6808) vẫn +7.2% so với best cũ. **Không có đánh đổi binary:** AUC/AUPR ở K=3 (0.9810/0.9796) còn nhỉnh hơn baseline K=13 (0.9805/0.9774); F1 binary phẳng (0.9360 vs 0.9366, trong nhiễu). Gain top-1 **không** mua bằng cách hy sinh head nhị phân.

### 2.3 Ý nghĩa thống kê (tổng hợp từ 3 thẩm định viên)

- **Seed-level n=3 vs 0.6350:** one-sample t = 8.6 (df=2, p<0.01); CI [0.662, 0.715] nằm *hoàn toàn* trên 0.6350 (cận dưới 0.662 > 0.6350) → trigger "default-to-refuted" không kích hoạt.
- **Paired per-fold, cùng seed 1234, cùng folds (K=3 vs baseline R0 K=13):** Δ = +0.0497, t = 3.62 (df=4, p≈0.02), 4/5 fold tăng, fold thứ 5 hòa.
- **Seed-paired (K=3 vs K=13, cùng seeds 1234/0/42):** Δ = +0.0497/+0.0474/+0.0383, paired t = 12.97 (df=2, vượt mốc p<0.01).
- **Pooled per-fold n=15 vs 0.6350:** t = 11.4; 0/15 fold K=3 rơi dưới 0.6350 (min fold 0.6540); K=3 thắng K=13 ở 74/75 cặp (98.7%). Survives Bonferroni ~15 config (α=0.0033).

### 2.4 Redundancy với no_hgcn (KHÔNG stack)

Combo **K=3 + no_hgcn = 0.6978 == no_hgcn một mình = 0.6978**. Hai lever **không cộng dồn** — chúng chạm vào *cùng* cơ chế (độ thưa / over-parameterization), không độc lập. Hệ quả: 0.688 (hoặc ~0.698 nếu dùng mô hình *đã ablate*) nằm gần trần thực dụng của hướng tune này; **không được claim additivity**.

Lưu ý: `softmax_5class + full_bilinear` (v2.0) = 0.6431 — khá khiêm tốn, < K=3, nên không phải lever chính.

---

## 3. Focus B — Ablation Fig.4: reversal là xác nhận thứ 5

Paper claim **mọi** ablation đều *làm hại* (all components critical). Dưới predictor `full_bilinear` trung thực (baseline 0.6311):

| Ablation | Top-1 F1 | Δ vs baseline | Hướng |
|---|---:|---:|---|
| no_hgcn | 0.6978 | **+10.6%** | HELP (đảo) |
| no_hgt | 0.6826 | **+8.2%** | HELP (đảo) |
| no_cl | 0.6680 | **+5.8%** | HELP (đảo) |
| no_avf | 0.6356 | +0.7% | ~phẳng |
| no_dv | 0.6303 | −0.1% | ~phẳng |

**Reversal vẫn dai dẳng dưới full_bilinear → đây là xác nhận độc lập THỨ 5.** Bốn lần trước đều trên diag predictor: (1) additive switch, (2) true rebuild (Plan E), (3) multi-seed verify (Plan H), (4) paper_literal loss (Plan F). Lần này dùng predictor *trung thực nhất* (full bilinear, đánh bại cả paper ở baseline) → loại trừ giả thuyết "reversal là artifact của diag predictor degenerate".

**Diễn giải:** DHGCMDA **over-parameterized cho v2.0** (1498 assoc). CL/HGCN/HGT là nhiễu trên dataset nhỏ này → bỏ đi giúp top-1. Đây là **finding hợp lệ về implementation gap**, KHÔNG phải lỗi tái hiện.

**Cảnh báo trung thực quan trọng:** no_hgcn = 0.6978 là một **mô hình đã ablate (đơn giản hơn)**, KHÔNG phải mô hình đầy đủ. Không được dùng 0.6978 làm "kết quả của DHGCMDA". Con số mô hình-đầy-đủ phòng thủ được vẫn là **0.688** (K=3, full architecture).

---

## 4. Focus C — v3.2

- **two_head + full_bilinear, 300ep / 3fold (metric đã sửa):** Top-1 F1 = **0.3232**, AUC 0.9133 → khớp Plan K (~0.33). Xác nhận lại trên Linux: con số ~0.33 ổn định, độc lập platform.
- **softmax_5class (sau khi tổng quát loss sang K+1 lớp):** Top-1 F1 = 0.2704 nhưng **AUC sụp về 0.1245** → softmax5 **làm hại** v3.2. softmax5 chỉ phù hợp v2.0 (4 type), không scale sang 5 type của v3.2. Việc tổng quát loss sang K+1 lớp là **thay đổi additive trung thực** (đăng ký dataset/loss, không đụng thuật toán model), giữ đúng ràng buộc no-algo-change.
- **Gap tới 0.86 vẫn terminal:** paper đạt 0.86 trên data curated **411×271 (density 10.5%) chưa public** (Plan K). Đã chứng minh density KHÔNG phải nguyên nhân (v3.2_wang_dense 385×275 @ 10.3% cho 0.3336 ≈ wang 0.3301). Gap thật ~0.33→0.86 do **chất lượng curation cụ thể + 4 nguồn similarity của paper**, không phải lỗi tái hiện của ta. **Không có cách no-author đóng được gap này.**

---

## 5. Focus D — Baselines

- **NMCMDA:** vẫn **blocked**. DGL cài được trên Linux nhưng binary `libgraphbolt_pytorch_2.5.1.so` thiếu → import fail. Cùng *class* blocker như Windows (DGL không compat torch 2.5.1). Code NMCMDA cũng không có trong repo. **Deferred** — trung thực: không chạy được mà không downgrade torch (sẽ phá môi trường DHGCMDA chính).
- **TDRC:** đã reproduce ~98% từ trước (Plan C/E): CV_type F1=0.4378 vs paper 0.4207 (+4.1%), CV_triplet AUPR=0.9246 vs 0.9059. Không chạy lại trong Plan L — đã đủ.

---

## 6. Cập nhật % tái hiện tổng thể & khuyến nghị

### 6.1 % tái hiện

| Khía cạnh | Trước Plan L | Sau Plan L |
|---|---|---|
| v2.0 binary (AUC/AUPR/F1) | 99% | 99% (giữ) |
| v2.0 Top-1 F1 | "EXCEEDS +6.4%" (0.635) | **EXCEEDS +15.3%** (0.688 multi-seed) |
| v2.0 ablation Fig.4 pattern | 0% (ngược paper) | 0% — **xác nhận lần 5** dưới full_bilinear |
| v3.2 binary | ~99% (AUC khớp) | ~99% (AUC 0.913) |
| v3.2 Top-1 F1 (metric đúng) | ~0.33 (Plan K) | 0.3232 (xác nhận Linux); gap→0.86 terminal |
| TDRC baseline | ~98% | ~98% (giữ) |
| NMCMDA baseline | blocked | blocked (DGL) |
| **Tổng thể** | ~64-67% | **~66-69%** (v2.0 Top-1 mạnh hơn, mọi finding strengthened) |

### 6.2 Khuyến nghị

- **ADOPT làm default mới cho v2.0:** `--predictor_mode full_bilinear --K_neigs 3`. Công bố **0.688 ± 0.011 (multi-seed)**, KHÔNG dùng 0.7006.
- **Không stack** K=3 với no_hgcn (redundant, đã chạm trần ~0.698).
- **Defer:** NMCMDA (cần downgrade torch / DGL fix — ROI thấp, rủi ro phá env); K<3 (K=1,2 chưa thử — có thể tune thêm nhưng marginal).
- **v3.2:** giữ two_head, KHÔNG dùng softmax5; gap tới 0.86 = terminal không-author, dừng tune.

---

## 7. Hạn chế & trung thực

- **Config-selection / multiplicity:** ~15 config chạy cho row v2.0. Test seed-level n=3 vs 0.6350 (p=0.013) **không** sống sót Bonferroni 15-way ở mức seed; nhưng test fold-level pooled (p<1e-4) và fact "cả 3 seed đều vượt 0.6350" thì sống sót → kết luận robust, nhưng phải khung đúng.
- **CV-tuning caveat:** K=3 tune trên *chính* CV dùng báo cáo, không có held-out riêng → 0.688 là ước lượng in-sample lạc quan; generalization split mới có thể thấp hơn. Giảm nhẹ: replicate qua seed không dùng selection.
- **Một phần gain là "phục hồi" baseline xui:** K=13 là K tệ nhất diag-sweep. Khung trung thực = "K=3 là K đúng dưới full_bilinear"; delta so với baseline *fair-tuned* nhỏ hơn delta so với baseline K=13.
- **n=3 seed nhỏ:** point estimate ~0.688 có CI rộng (df=2); hướng vững nhưng độ lớn cần thêm seed để siết.
- **Per-fold range chạm nhau mong manh:** 1 fold K=3 (0.6540) nằm ngay dưới best-fold của baseline (0.6554) — means và paired diff tách rõ, nhưng phân phối chồng lấn nhẹ ở đuôi.
- **K-sweep thưa:** chỉ {3,7,13} dưới full_bilinear; K=1,2 chưa thử → "small-K beats K=13" mạnh hơn "K=3 là tối ưu tuyệt đối".
- **no_hgcn=0.6978 là mô hình đã ablate**, không phải DHGCMDA đầy đủ — không dùng làm headline.
- **v3.2 & D giới hạn:** gap 0.86 terminal (data chưa public); NMCMDA blocked (DGL); mọi win của Plan L là **v2.0-specific** — K=3 không validate có lợi trên v3.2.