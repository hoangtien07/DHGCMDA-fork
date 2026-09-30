# BÁO CÁO TIẾN ĐỘ — Tái hiện bài DHGCMDA

**Học viên:** [điền tên] — **Người hướng dẫn:** [điền] — **Ngày:** 26/09/2026
**Bài báo:** DHGCMDA — a dual-view heterogeneous graph contrastive learning framework for miRNA–disease association prediction (Sun Y. et al., BMC Bioinformatics 2026). Repo gốc: github.com/CDMBlab/DHGCMDA; repo fork: github.com/hoangtien07/DHGCMDA-fork.

## I. Lý thuyết cơ bản

**Bài toán:** dự đoán liên kết miRNA–bệnh không chỉ nhị phân mà theo **loại bằng chứng** (association type): genetics, epigenetics, circulation, target, tissue. Một cặp (miRNA, bệnh) có thể mang nhiều loại nhãn → bài toán multi-label, metric chính là Top-1 precision/recall/F1 chấm theo cặp.

**Các loại độ tương đồng (similarity) dùng trong bài:**

- **DSSM — Disease Semantic Similarity:** độ tương đồng ngữ nghĩa giữa hai bệnh, tính từ đồ thị có hướng không chu trình (DAG) của MeSH theo kiểu Wang — bệnh chia sẻ tổ tiên trong ontology thì gần nhau. Đây là nguồn ngoài (ontology), không sinh từ nhãn.
- **MISIM — miRNA functional similarity:** độ tương đồng chức năng giữa hai miRNA, được suy ra từ các bộ bệnh mà chúng liên kết (định nghĩa kiểu MISFUGE): hai miRNA gắn với các bệnh ngữ nghĩa giống nhau thì giống nhau về chức năng. **Lưu ý:** vì được suy từ nhãn, MISIM là label-derived — đưa vào như input tĩnh là một kênh rò rỉ.
- **GIP — Gaussian Interaction Profile kernel:** hai miRNA (hoặc bệnh) có profile liên kết giống nhau trong ma trận liên kết thì tương đồng cao. GIP phải tính lại mỗi fold từ phần train, nếu tính trên ma trận đầy đủ là rò rỉ.

**Siêu đồ thị (hypergraph):** khác đồ thị thường (cạnh nối 2 đỉnh), mỗi siêu cạnh nối nhiều đỉnh → biểu diễn được quan hệ bậc cao. G = (V, E, W) với ma trận liên thuộc Y: Y(v,e)=1 nếu v ∈ e.

- **KNN hypergraph:** với mỗi đỉnh, chọn K láng giềng gần nhất theo similarity tạo thành một siêu cạnh → siêu đồ thị có |V| siêu cạnh, mang thông tin cục bộ (local).
- **GMM/cluster hypergraph:** phân cụm các đỉnh theo đặc trưng (Gaussian Mixture Model hoặc K-means), các đỉnh cùng cụm tạo siêu cạnh → mang thông tin toàn cục (global).

**HGCN (Hypergraph Convolutional Network):** lan truyền phổ trên siêu đồ thị, biểu diễn đỉnh tại lớp t: X^(t) = D_v^(−1/2) · H · W · D_e^(−1) · H^T · D_v^(−1/2) · X^(t−1) · Θ — tức đi qua siêu cạnh rồi quay lại, nắm quan hệ bậc cao mà bipartite graph không làm được.

**Contrastive learning (học tương phản):** pipeline 4 bước: data augmentation → encoder → projection head → loss. Với hai view KNN/GMM: hai embedding của **cùng một nút** là cặp dương; **intra-view negative** = cặp trong cùng view, **inter-view negative** = cặp chéo view. Loss dạng InfoNCE với temperature τ, hai chiều anchor được cân bằng α=0.5. Mục tiêu: embedding nhất quán giữa view, giảm phụ thuộc vào số nhãn đã biết ít.

**HGT (Heterogeneous Graph Transformer):** attention nhiều đầu trên đồ thị không đồng nhất, phân biệt loại nút/loại cạnh — trong bài này để tổng hợp thông tin giữa miRNA và bệnh theo các loại liên kết khác nhau.

**Attentional feature fusion:** thay vì trung bình hai view, học trọng số: trích thông tin local (PWConv) + global (GAP) → sigmoid ra attention map ∈[0,1] → nhân với đặc trưng gốc rồi hợp nhất.

**Predictor hai đầu (two-head):** đầu existence (cặp có liên kết không — binary) + đầu type (loại nào — per-type bilinear miᵀ·W_t·dis). Loss tổng = existence + type + contrastive + reconstruction.

## II. Nội dung paper DHGCMDA

**Ý tưởng chính:** end-to-end framework dự đoán loại liên kết miRNA–bệnh bằng học tương phản trên **hai siêu đồ thị** (dual-view): một view cục bộ (KNN) và một view toàn cục (GMM), để vừa nắm lân cận gần vừa nắm cấu trúc cụm.

**Pipeline:**

1. Tính integrated similarity cho miRNA (functional + GIP) và disease (semantic DSSM + GIP).
2. Xây hai siêu đồ thị mỗi phía: KNN-hypergraph (K láng giềng) và GMM-hypergraph (phân cụm).
3. HGCN trên từng view → biểu diễn bậc cao.
4. Contrastive learning giữa hai view (intra + inter negative) → embedding nhất quán.
5. Attentional view fusion → biểu diễn hợp nhất.
6. HGT trên đồ thị không đồng nhất miRNA–bệnh (cập nhật động mỗi 5 epoch).
7. Predictor hai đầu: existence + loại.

**Bộ dữ liệu:**

| Dataset | miRNA | Bệnh | Số loại | Triplets | Ghi chú |
|---|---:|---:|---:|---:|---|
| HMDD v2.0 | 495 | 383 | 4 | ~5.4k | đi kèm repo |
| HMDD v3.2 (bài) | 411 | 271 | 5 | 11.748 | **không phát hành** — per-type: target 3.997, circulation 2.293, epigenetics 403, genetics 1.155, tissue 3.900 |

**Parameters (theo code phát hành):**

| Tham số | Giá trị |
|---|---|
| K_neigs (KNN) | 13 |
| clusters (GMM) | 9 |
| hidden dim | 256 |
| HGT layers / heads | 2 / 4 |
| learning rate | 0,0001 |
| epochs | 650 |
| dropout | 0,3 |
| cl_temperature / inter_view_weight | 0,5 / 0,3 |
| update_graph_frequency | 5 epoch |
| focal_gamma | 2,5 |
| exist_weight | 0,3 (paper thực tế ≈ 0,1 — xem mục III.2) |
| validation | 5-fold |

**Kết quả paper claim:** v2.0 Top-1 F1 = 0,5970, AUC = 0,9669; v3.2 P = 0,7915, R = 0,9421, **F1 = 0,8600**, AUC = 0,9181, AUPR = 0,9271.

## III. Tái hiện

### III.1. Môi trường

CPU-only (torch 2.5.1+cpu, torch-geometric 2.7.0), mỗi fold v2.0 ~10 phút. Code phát hành chạy được ngay v2.0; v3.2 không chạy được vì evaluator hardcode 4 loại (mục III.3).

### III.2. HMDD v2.0 — tái lập được

Sau khi sửa 3 điểm lệch code–paper (HGT heads=4, dynamic graph update mỗi 5 epoch, λ_recon=1.0) và predictor diag→full bilinear đúng mô tả:

| Metric (v2.0, 5-fold) | Paper | Reproduced | Lệch |
|---|---:|---:|---:|
| Top-1 F1 | 0,5970 | **0,6350** | +0,038 (vượt paper) |
| AUC | 0,9669 | 0,9805 | +0,014 |
| Binary acc/P/R/F1 | ~0,95 | khớp ~99% | |

**Ablation — pattern NGƯỢC paper (đã verify multi-seed):**

| Variant | Top-1 F1 | Δ vs baseline |
|---|---:|---:|
| baseline | 0,5521 | — |
| w/o contrastive loss (no_cl) | 0,6206 | **+12,4%** |
| w/o HGT (no_hgt) | 0,6415 | **+16,2%** |
| w/o dual-view (no_dv) | 0,5705 | +3,3% |

Paper báo các ablation *làm giảm* hiệu năng; code phát hành cho thấy bỏ hai module chính lại *tăng*. Đã kiểm chứng 4 cách (additive, true rebuild, 4 seeds, 2 loss mode) → reversal là **legitimate finding**, không phải nhiễu. Với exist_weight=0.1 (đúng spec paper hơn): no_avf −3,7%, no_dv −2,0% → 2/5 ablation đúng hướng.

### III.3. HMDD v3.2 — hành trình forensic

**a) Metric phát hành không chấm được v3.2.** `compute_top1_metrics` hardcode 4 loại → mọi mẫu v3.2 (5 loại) bị bỏ → F1 = 0,0000 bất kể model. Chứng minh bằng predictor hoàn hảo nhân tạo (1,0 ở 4 loại, 0,0 ở 5 loại) + metric tổng quát khớp v2.0. Đo lại đúng: v3.2 thật cho **F1 ≈ 0,27–0,36** — không phải collapse, nhưng cách xa 0,86.

**b) Khôi phục đúng bộ dữ liệu của bài.** Repo đồng tác giả `Ouyang-Dong/SPLDHyperAWNTF_Model` (đã xoá) còn trong Software Heritage → thư mục `MDAv3.2_3` **khớp tuyệt đối** fingerprint bài (411×271×11.748, đúng từng per-type count). Đã commit vào fork kèm SHA256 canonical — mọi run v3.2 sau chạy trên đúng data bài. Density-control trước đó đã phủ định "độ dày là nguyên nhân" (385×275 @10,3% → 0,3336 ≈ baseline 0,3301).

**c) Neo evaluator bằng hàng SPLD của Bảng 3.** Tái lập evaluator SPLD (pair-fold, 1 dự đoán/cặp) và chạy mô hình SPLDHyperAWNTF gốc trên artifact:

| | P | micro-R | macro-R | F1 |
|---|---:|---:|---:|---:|
| SPLD trong Bảng 3 bài DHGCMDA | 0,6219 | — | — | — |
| Golden replay | **0,6219** | **0,4624** | **0,5069** | **0,5585** |

Precision khớp đến 4 chữ số → pipeline forensic đúng, baseline của bài được chấm bằng eval SPLD.

**d) Chứng minh bất khả thi của claim.** Protocol 1-dự-đoán/cặp: recall tối đa kể cả oracle là macro-R ≤ **0,8642** (micro-R ≤ 0,7435), vì cặp đa nhãn chỉ ghi 1 loại. Bài báo **R = 0,9421 > trần** → eval thật của tác giả khác protocol đã document; eval code chưa tìm được public (đã quét Software Heritage, forks, các repo cùng lab).

**e) Kiểm toán nguồn similarity + kênh rò rỉ.** `mi_fun_sim` = MISIM 2.0 chính xác → label-derived → leaky nếu dùng tĩnh; `DSSM` ontology → sạch. Code phát hành còn có 3 kênh leakage: GIP trên full matrix, full association trong node features/hypergraph, test edges trong HGT. Đo trực tiếp: MISIM tĩnh thổi phồng vote **+0,020** (CI95 [0,013;0,030]) nhưng **không ảnh hưởng SPLD** (static 0,5585 ≈ fold 0,5570 ≈ no 0,5594).

**f) Leaderboard đóng băng** (golden protocol, honest = train-only sims):

| Mô hình | Track | F1 |
|---|---|---:|
| vote miRNA-GIP only (0 tham số) | honest | **0,5910** |
| vote MISIM+DSSM | leaky | 0,5799 |
| vote fold-MISIM+DSSM | honest | 0,5733 |
| SPLD (3 cấu hình MISIM) | legacy | 0,5570–0,5594 |
| vote GIP+DSSM(+J) | honest | 0,5599–0,5623 |
| DHGCMDA encoder (honest) | honest | 0,4481 |
| vote DSSM only / co-occur | honest | 0,4173 / 0,3941 |
| bilinear / DistMult | honest | 0,283±0,006 / 0,252±0,007 |
| **Paper claim** | — | **0,8600** |

Bootstrap 10k: `simknn_gip` − SPLD = +0,0014, CI95 [−0,0075;+0,0104] → **vote 0-tham-số ngang SOTA công bố**; encoder public thua vote −11pt; ablation nhỏ: chỉ phía miRNA là đủ, thêm DSSM-side *giảm* −3,1pt → **hiệu năng nằm ở lan truyền cục bộ lân cận miRNA, không phải machinery multi-view/contrastive**.

### III.4. Parameters em dùng

| Cấu hình | Giá trị |
|---|---|
| v2.0 | epoch 650, lr 1e-4, hidden 256, K=13, clusters=9, exist_weight 0,1, full_bilinear, seed 0/1234 |
| v3.2 pipeline gốc (leaky, metric đúng) | 300 epoch, full_bilinear, exist_weight 0,1 |
| Encoder honest | 300 epoch × 5-fold, multilabel BCE, per-fold GIP/features/KNN |
| Vote baselines | không train (deterministic), J prior λ=0,2 hoặc inner-val |
| Seeds | 0/1/2 cho learned models; rep-CV seeds cho vote |

## IV. Kết quả tái hiện tổng hợp

| Bài | Dataset | Paper | Reproduced | Verdict |
|---|---|---:|---:|---|
| DHGCMDA | v2.0 binary | ~0,95 | ~99% | ✅ khớp |
| DHGCMDA | v2.0 Top-1 | 0,5970 | **0,6350** | ✅ vượt |
| DHGCMDA | v2.0 ablation pattern | giảm khi bỏ module | **ngược** | ❌ không khớp (verified) |
| DHGCMDA | v3.2 Top-1 | 0,8600 | honest ≤0,36; SPLD-protocol ≤0,59 | ❌ **không tái hiện được — claim vượt trần toán học của protocol** |
| SPLDHyperAWNTF | v3.2 (artifact) | P=0,6219 | **P=0,6219, F1=0,5585** | ✅ khớp 4 chữ số |
| TDRC | v3.2 | F1=0,4207 | 0,4378 | ✅ khớp +4% |

**Kết luận:** v2.0 tái lập tốt; v3.2 claim 0,86 **không thể** sinh ra từ protocol paper mô tả (R=0,9421 > trần 0,8642 đã chứng minh) và eval code thật chưa phát hành. Đóng góp mới của đợt này: (1) khôi phục artifact v3.2 chính xác, (2) neo hàng SPLD của Bảng 3 vào thực nghiệm, (3) chứng minh bất khả thi của claim, (4) baseline vote 0-tham-số không rò rỉ đạt ngang SOTA. Khung manuscript + figure/bảng publication đã dựng sẵn trong `paper/`.
