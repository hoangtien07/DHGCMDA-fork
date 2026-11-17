# Đề xuất protocol U1-CVtriplet (chưa chạy — spec + đánh giá chi phí)

Ngày soạn: 2026-11-18. Trạng thái: **đề xuất, chưa phê duyệt, chưa chạy**. Mục đích: lấp khoảng trống CVtriplet — hiện chỉ có baseline count/logistic (`triplet_baseline.json`: AUROC 0,9324, F1 nhị phân 0,8565 trên split mô phỏng), chưa có kết quả model DHGCMDA-style nào cho task existence.

## Định nghĩa task theo mã tác giả công bố (v2.0, commit 06f9ba3)

- Predictor `SimplifiedTypePredictor`: đầu ra `[mi, dis, 1+K]`; kênh 0 = `sigmoid(miᵀ·diag(r_exist)·dis)` (existence), kênh 1..K = `softmax(logits/T)` theo type, T học được clamp [0,5; 5,0].
- `SimplifiedMultiTypeAssociationLoss`: existence focal loss trên tập `one_index` (positive cells) và `zero_index` (zero cells, subsampled theo `neg_sample_ratio`), trọng số alpha cho phần negative; type CE chỉ trên positive cells. Tổng `exist_weight·exist + type_weight·type`.
- Eval CVtriplet (test_optimized): mỗi fold, positives test + số unknown bằng nhau lấy từ zero-cells (`np.random.seed(seed)`, shuffle), binary metrics trên kênh existence.

## Điểm cần quyết định (cần pin trước khi chạy)

1. **Split:** CVtriplet chia theo triplet, không theo pair — cần 5 train-mask mới theo triplet-fold (mô phỏng seed0 như `triplet_baseline`, vì fold tác giả không public). Đây là bundle build mới: functional/semantic phải rebuild theo train-mask triplet; `build.py` hiện pin pair-fold, cần tổng quát hoá entry-point, KHÔNG sửa bundle cũ.
2. **Existence supervision:** positive = triplet train; negative = cells zero của A_train. Lấy mẫu mỗi epoch hay bộ cố định, ratio nào — upstream subsample theo `neg_sample_ratio`; pin trước giá trị và seed. False-negative trong zero-cells đã biết (bank-size analysis) — ghi trong giới hạn.
3. **Objective:** giữ nguyên toàn bộ objective U1 hiện tại + thêm existence focal loss với hệ số pin trước (upstream `exist_weight`; paper không nêu cho v3.2). Đây là arm MỚI, không phải cap10/cap1; đối chứng matched-control = count baseline trên cùng split + tuỳ chọn arm existence-học-mà-không-contrastive.
4. **Eval:** chỉ chấm trên triplet test của fold tương ứng (positives + unknown sampled); metric AUROC/AP/F1@0.5 binary trên kênh existence — KHÔNG phải Top-1 F1 CVtype. Transductive caveat (danh sách positive đầy đủ định nghĩa unknown) giữ nguyên từ triplet_baseline.
5. **Provenance:** mọi file mới trong namespace riêng (`triplet1/`), protocol này hash-pin, source băm trước/sau, ghi rõ existence head là diễn giải từ `SimplifiedTypePredictor` v2.0 — không phải mã v3.2 của tác giả.

## Đánh giá chi phí

- Code mới: existence channel trong U1 predictor (~60 dòng, pattern có sẵn), existence focal loss (~40 dòng), triplet-split bundle builder + eval (~200 dòng), fixture/hash.
- Compute: 5 bundle builds (phút mỗi cái) + tối thiểu 5 lượt train 650 epoch (~11-16 phút/lượt CPU, 2 worker → ~40-80 phút) cho seed1234; 15 lượt nếu đủ 3 seed (~2-3 giờ).
- Rủi ro: thay objective làm kết quả không so trực tiếp với cap10/cap1; kết luận chỉ ở mức "U1+existence vs count baseline trên split mô phỏng" — không phải tái hiện paper CVtriplet (fold/unknown-sampling tác giả chưa biết).

## Tiêu chí dừng

- Dừng nếu preflight hash không đạt, bundle build lệch contract, hoặc existence channel không phân biệt được positive/unknown trên fold0 sanity run.
- Báo cả âm tính: nếu AUROC model ≤ baseline count, đó là kết quả hợp lệ cần báo, không chỉnh để vượt baseline.
