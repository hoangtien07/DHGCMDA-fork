# B3 + B4 — Findings (branch breakthrough-imbalance-features) — IN PROGRESS

> Ngày bắt đầu: 2026-07-12. Additive-only (flag mới, default = hành vi cũ). KHÔNG sửa Calculate_Metrics.
> Mục tiêu (user chọn): B3 imbalance-aware type loss (phục hồi minority Tissue v3.2) + B4 feature ngoài association / chặn leakage.

## Phát hiện phát sinh trong lúc implement (đều verify-in-code)

### F1 — Bug mislabel Tissue trong type-loss `two_head` (nguyên nhân TRỰC TIẾP của collapse T5)
[main_experiments_hetero1.py `_compute_type_loss`] bản gốc hardcode:
```python
type_indices[pos_targets == 1] = 0 ... == 4] = 3   # KHÔNG map type 5
# type 5 (Tissue) giữ nguyên 0 (default zeros_like) → HỌC như class 0 (circulation)
```
→ Với v3.2 (5 type), mọi mẫu Tissue bị gán nhãn train = circulation → model không bao giờ học Tissue.
Đây là **một nguyên nhân độc lập** của "Tissue collapse" (bên cạnh metric-bug ở eval). Đã sửa tổng quát `k→k-1`
(tương đương HỆT cho type 1–4 của v2.0 → không đổi kết quả Plan A–M).

**A/B ĐỊNH LƯỢNG (200ep×3fold v3.2_wang, `--legacy_type_map` vs fixed, per-class từ dump):**
| | LEGACY (bug) | FIXED | Δ |
|---|---:|---:|---:|
| **Tissue recall (T5)** | **0.000** | **0.479** | phục hồi hoàn toàn |
| macro-F1 | 0.1885 | 0.3254 | +72.6% |
| accuracy | 0.2626 | 0.3564 | +35.7% |
| macro-recall | 0.3415 | 0.4283 | +25.4% |
→ **HAI bug độc lập cùng đè Tissue**: (a) metric-bug ở eval (Plan K), (b) mapping-bug ở TRAIN (Tissue→circulation).
Fix (b): Tissue recall 0.000→0.479. Files: `logs/b3_perclass_{legacy,fixed}.log`, `results/b3/dump_{legacy,fixed}/`.
Lưu ý: imbalance-loss (logit_adjust/ldam) trên NỀN mapping-fix chỉ ≈ trung tính (ldam≈ce ở diagnostic) →
**đòn bẩy thật là mapping-fix, KHÔNG phải loss cầu kỳ**. Honest finding.

### F2 — Leakage: ma trận association ĐẦY ĐỦ nối thẳng vào feature hypergraph
[main:942/946/958/962] mọi view = `cat([association_matrix, similarity])`, với
`association_matrix = train_data[4] = md_p` = **ma trận đầy đủ, chia sẻ giữa các fold, còn nguyên nhãn test**
([trainData.py:106]). Không có chỗ nào mask test-positive trước khi dựng hypergraph.
→ Khi đoán cặp test (i,j), feature của miRNA i đã chứa `association[i,j]` = nhãn cần đoán → **feature/similarity leakage**
(đúng taxonomy Kapoor / consistent-eval-2020). Giải thích AUC luôn ~0.98 (vượt paper) ở mọi cấu hình Plan A–M.

**KẾT QUẢ FULL (650ep×5fold, A/B sạch — CHỈ khác cờ leakage_free):**
| Thiết lập | Top-1 F1 | AUC |
|---|---:|---:|
| Leaked (baseline K=2 s1234) | 0.6940 | 0.9816 |
| **Leakage-free** (mask 268 test-pos/fold) | **0.6075** | **0.9402** |
| **Mức tụt do leakage** | **−12.5%** | **−4.2%** |

→ Leakage thổi phồng đáng kể Top-1 (−12.5%) và AUC (−4.2%). Giải thích AUC ~0.98 kỳ lạ ở Plan A–M.
**Tuy nhiên** leakage-free VẪN 0.608/0.940 ≥ paper 0.597/0.9669(*) → cải thiện right-sizing (K=2, full_bilinear)
sống sót qua de-leak. (*) so sánh chỉ định hướng: số paper cũng đo dưới protocol leaky (cùng code) → so công bằng
cần de-leak cả paper. File: `results/b4/leakagefree_v2_K2_s*.json`, log `logs/b4_leakagefree_v2_K2_s*.log`.

**MULTI-SEED (robust, K=2 full_bilinear):**
| Seed | Leaked F1 | LF F1 | Leaked AUC | LF AUC |
|---|---:|---:|---:|---:|
| 1234 | 0.6940 | 0.6075 | 0.9816 | 0.9402 |
| 0 | 0.7008 | 0.6272 | 0.9814 | 0.9367 |
| 42 | 0.6974 | 0.6131 | 0.9823 | 0.9335 |
| **Mean** | **0.6974** | **0.6159** | **0.9818** | **0.9368** |

→ Leakage thổi phồng **Top-1 F1 −11.7%, AUC −4.6%** nhất quán qua 3 seed. Headline TRUNG THỰC (leakage-free) =
**0.616 ± 0.010** (so paper 0.597 leaky: vẫn +3%, nhưng phải nhớ paper cũng đo leaky). Đây là finding critique mạnh
nhất: hiệu năng báo cáo (kể cả của chính ta ở Plan A–M) bị leakage nâng ~1 điểm phần mười.

### F3 — "Heterogeneous similarity" của paper thực chất association-derived (bác đóng góp #1)
Paper đóng góp #1: *"dual-view hypergraphs ... avoiding excessive reliance on association-derived similarity"*.
Thực tế theo `介绍.txt` + code:
- **M-GSM = "miRNA高斯相互作用谱核相似度" = GIP (association-derived)** — nhưng code gán nhãn sai là
  *"miRNA-sequence features"* ([prepareData.py:390]). M_GSM range [0,1], diag=1.0 = kernel similarity, KHÔNG phải sequence.
- miRNA View 1 = [assoc, M_GSM(GIP)]; View 2 = [assoc, M_FSM(functional, semi-assoc)] → **cả 2 view miRNA bị
  association-derived chi phối**. Chỉ phía disease có semantic MeSH thật (D_SSM1/2).
→ Claim "tránh phụ thuộc association-derived similarity" **mâu thuẫn với chính code**. Đây là động lực cho B4-features:
thay M_GSM(GIP giả-"sequence") bằng **feature sequence miRNA thật (miRBase k-mer)** → hiện thực hoá đúng tiền đề paper + inductive.

**B4-features ĐÃ BUILD (user cấp `mature.fa` miRBase):** `build_mirna_seq_features.py` → `v2.0_495m383D/M_SEQ.txt` (k=4).
- **Khớp 487/495 (98.4%)**; 8 tên legacy (mir-720, mir-189, mir-1273a...) → mean vector.
- **corr(M_SEQ, M_GSM=GIP) off-diag = −0.027 ≈ 0** → feature sequence THẬT **độc lập hoàn toàn** với GIP.
  Bằng chứng định lượng: "sequence" của paper (M_GSM) KHÔNG phải sequence.
- Flag `--mirna_seq_sim_path` nạp M_SEQ thay M_GSM cho View 1 (additive). Smoke v2.0 ✓ (exit 0).
- Ma trận thí nghiệm B4-features (chờ 2 job hiện tại xong): {GIP,real-seq} × {leaked, leakage_free}.

## Trạng thái thí nghiệm
- [running] B3 diagnostic ce/logit_adjust/ldam — v3.2_wang full_bilinear two_head, 60ep×2fold, correct-metric.
  Logs `logs/b3_v32_{ce,logitadjust,ldam}_ep60f2.log`.
- [running] B4 leakage-free full — v2.0 K=2 s1234 650ep×5fold. So baseline leaked = Top-1 F1 0.6940 / AUC 0.9816.
  Log `logs/b4_leakagefree_v2_K2_s1234.log`.

## Cần user xác nhận
- **B4-features (miRBase)**: tải `mature.fa` từ miRBase (public, nhỏ) → k-mer freq cho 495 miRNA → thay M_GSM.
  Cần network + map tên miRNA. Chưa làm (chờ xác nhận). Disease-side MeSH đã có sẵn (D_SSM1/2).
