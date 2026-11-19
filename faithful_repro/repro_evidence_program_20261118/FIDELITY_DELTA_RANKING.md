# FIDELITY_DELTA_RANKING — WP-A3 (REV4)

Mỗi discrepancy: cơ chế ảnh hưởng, ref code, test cụ thể, trạng thái
(đã audit vs mới), và liệu có đáng một bounded experiment.

## Đã audit xong (fixture/measurement-backed)

| ID | Discrepancy | Cơ chế ảnh hưởng | Bằng chứng | Test | Trạng thái |
|---|---|---|---|---|---|
| D-metric | k4 release bỏ type-5 + len-6 preds | Headline F1 0.0 bất kể model | byte-equivalence fixture | test_metric_contracts | CLOSED |
| D-denom | U1 multi-hot denominator vs author scalar-mean CE | Trọng số per-sample đổi khi 24.2% cặp multi-label | micro-fixture denominator diverge | test_fidelity_microfixtures | CLOSED (đo được, không sửa) |
| D-pred | diag vs full bilinear | +5.9% trên v2.0 khi thay diag→full | j1 results | measured | CLOSED |
| D-refresh | paper: recon→G update; release: chỉ edge_index HGT | HGCN G-matrix không bao giờ refresh | FIDELITY_MATRIX + graph fixture | CLOSED |
| D-fusion | fixed 0.6/0.4 vs learned gate | attention học được; direction unknown | declared adaptation | CLOSED (không đo isolate) |
| D-label | scalar overwrite vs multi-hot truth | 24.2% cặp mất nhãn phụ khi scalar | tensor measure | CLOSED |
| D-neg | PU zero_index ≤10× có thể chứa held-out positives | existence head học label nhiễu | prior manifest | CLOSED (declared) |
| D-leak | full-A features (X=[S,A], GIP, hypergraph, HGT edges) | test pairs lọt vào features | paired diag_leak runs (xem dưới) | CLOSED, kết quả dưới |

## MỚI: kết quả paired leakage isolation (B1 đã được trả lời)

Cùng codebase, cùng config (recon_reduction=sum), seed 1234, 5 fold,
chỉ toggle `--diag_leak` (full-A vs train-masked features):

| Dataset | CLEAN (per-fold) | LEAK | Δ leak |
|---|---|---|---|
| v2.0 type | [.396,.367,.483,.406,.393] → 0.409 | [.370,.548,.463,.521,.390] → 0.458 | **+0.049** |
| v3.2_ref type | [.291,.328,.246,.283,.293] → 0.288 | [.282,.287,.230,.300,.293] → 0.279 | −0.009 |

→ Trên v2.0, leakage inflate ~+0.05 F1 (cùng chiều với
baseline upstream leaky 0.5568 vs clean ~0.41–0.54 tùy config).
Trên v3.2_ref, leakage KHÔNG giúp (Δ≈0, mixed sign per fold).
Câu hỏi B1 của roadmap — "isolate performance influence of the
release leakage path" — đã có câu trả lời paired, không cần
chạy lại. (Single-seed caveat: 1 seed × 5 folds; directional.)

## Chưa giải quyết — xếp hạng thông tin

| Rank | Discrepancy | Vì sao chưa đóng | Giá trị isolate | Khuyến nghị |
|---|---|---|---|---|
| 1 | Eval convention của paper (scalar vs multi-hot vs any-hit) | cần eval script tác giả | quyết định tính so sánh được của 0.86 | EXTERNAL_ARTIFACT — không train được |
| 2 | Filter tensor (<2 vs <3, boundary rows) | cần pipeline curation | quyết định dataset identity | EXTERNAL_ARTIFACT |
| 3 | recon loss raw-Gram vs normalized-MSE trong author-vs-paper | paper/code/paper-text lệch nhau; effect size không đo được trên v3.2 | trung bình | NO_TRAIN — câu hỏi kỹ thuật, không đổi verdict |
| 4 | exist_weight 0.3/0.7 vs paper Eq.32 | spec gap, không resolve được từ trong | thấp | NO_TRAIN |
| 5 | checkpoint/early-stop rule | chính sách chọn model | thấp — đã pin fixed-650 | NO_TRAIN |

## Gate A3 → NO_TRAIN_NECESSITY

Discrepancy thông tin nhất còn mở (rank 1–2) là **ngoại sinh** —
không thể đóng bằng training. B1 đã answered bằng paired runs có
sẵn. Không bounded experiment nào được biện minh → Phase B SKIP.
