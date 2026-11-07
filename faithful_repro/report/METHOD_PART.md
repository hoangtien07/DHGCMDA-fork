## 3. Đối chiếu paper và code tác giả: những gì đã sửa

**Nguyên tắc:** chỗ nào paper ghi rõ thì làm theo paper. Chỗ nào paper không ghi thì giữ nguyên hành vi code tác giả và gắn nhãn [giả định]. Không có tham số nào được tune.

| # | Nội dung | Paper | Code tác giả | faithful.py |
|---|---|---|---|---|
| 1 | Rò rỉ dữ liệu test | CV: các cặp test phải chưa biết | Ma trận `association_matrix` đầy đủ (gồm cặp test) đi vào feature `[S,A]`, hypergraph, cạnh HGT; `M_GSM` là GIP tính trên toàn bộ ma trận | Chỉ dùng A_train; GIP tính lại theo từng fold |
| 2 | Hai view (Eq.1–2) | Nhánh v nhận X^(v) | Cả 2 nhánh nhận `concat_mi_tensor` (feature view 1) | Mỗi nhánh nhận view của mình |
| 3 | KNN (Sec 2.3) | Cosine, K=13 | Euclidean (`Eu_dis`), K=13 hardcode | Cosine, `--K` mặc định 13 |
| 4 | AVF (Eq.16–18) | Gate α_v và β_v học được | `0.6*z1 + 0.4*z2` cố định | Gate SE theo kênh + softmax β [giả định: hệ số reduction 4] |
| 5 | Reconstruction (Eq.19–21) | MLP 2 lớp, Ŝ=ZZᵀ, ‖S−Ŝ‖_F² | Decoder cosine, 0.15×MSE | Đúng như paper, λ3=1 |
| 6 | Cập nhật graph động (Sec 2.8) | Mỗi 5 epoch, ngưỡng 0.5, dựng lại hypergraph | Mỗi epoch khi MSE thay đổi > 0.01, không dựng lại hypergraph | Đúng như paper (Ŝ được min-max về [0,1] trước khi so ngưỡng [giả định]) |
| 7 | Hàm loss (Eq.32) | L_type + 1·L_intra + 0.3·L_inter + 1·L_recon | recover + cl_mi + cl_dis + 0.15·recon + 1e-4·L2, cộng label smoothing 0.1 | Đúng Eq.32, cộng thêm existence loss 0.3 (paper dự đoán existence nhưng không định nghĩa loss cho nó, nên giữ của tác giả [giả định]) |
| 8 | Multi-label (Eq.33) | Σ_k w_k y_k log p_k | Mỗi cặp chỉ giữ 1 nhãn; 161/1498 cặp v2.0 bị ghi đè nhãn | Tensor Y[m,d,C] multi-hot |
| 9 | Class weight (Eq.34) | Effective number, tính trên tập train | `counts=[367,157,293,681]` hardcode | n_k lấy từ train của từng fold |
| 10 | Số loại association | 4 (v2.0) / 5 (v3.2) | `num_types=4` hardcode; metric loại bỏ type 5 | C đọc từ data |
| 11 | HGT (Sec 2.7) | 2 lớp, 4 head, residual + LayerNorm | 8 head mặc định | 4 head, x = LN(x + Dropout(HGT(x))) |
| 12 | Optimizer | Adam, có early stopping | AdamW, không early stopping, chạy cố định 650 epoch | Adam lr 1e-4 (giữ của tác giả); early stopping trên 10% validation tách từ tập train, đánh giá mỗi 10 epoch, patience 10 [giả định] |
| 13 | Giao thức CV (Sec 3.2) | CV_triplet và CV_type là 2 giao thức riêng | 1 lần split theo cặp | Split riêng cho từng giao thức; CV_triplet có số triplet chưa biết bằng số positive |
| 14 | Lỗi bị nuốt | — | `forward()` có `except Exception` trả về điểm hằng | Bỏ |

**Giữ nguyên của tác giả vì paper không ghi rõ:**
- Inter-CL lấy toàn bộ cặp không liên kết làm negative; margin m = 0.5, trọng số margin 0.1.
- Predictor bilinear dạng đường chéo, kèm projector và temperature (paper chỉ ghi "bilinear").
- Existence loss dạng focal, negative 10:1, trọng số negative 0.5.
- Gradient clipping 1.0; hidden 256; dropout 0.3.
- Điểm của một triplet bằng p_exist(m,d) × p_type(m,d,k) [giả định].
- Metric dùng đúng định nghĩa của tác giả: AUPR = auc(recall, precision); F1 lấy ở ngưỡng cho F1 lớn nhất; Top-1 R là trung bình recall theo từng loại.
