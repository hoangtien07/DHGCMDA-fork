## 5. Ablation (Fig.4), v2.0 CV_type, seed 1234, 5 fold

| Biến thể | Top-1 F1 | Δ so với full (cùng seed 1234) | Paper: biến thể < full |
|---|---:|---:|---|
| full | 0.4091 (3 seed: 0.3878±0.0151) | — | — |
| w/o CL (λ1=λ2=0) | 0.4141 | +0.0050 | NGƯỢC |
| w/o HGCN (GCN trên đồ thị KNN cặp đôi) | 0.3968 | -0.0123 | khớp |
| w/o AVF (ghép nối [Z1‖Z2]) | 0.4759 | +0.0668 | NGƯỢC |
| w/o HGT (bỏ HGT) | 0.3728 | -0.0363 | khớp |
| w/o DV (chỉ 1 view, không có intra-CL) | 0.4202 | +0.0111 | NGƯỢC |

Ablation chỉ chạy 1 seed. Độ lệch chuẩn giữa các seed của bản full là 0.015. Chênh lệch của w/o AVF (+0.067, khoảng 4.4 sd) lớn hơn rõ mức nhiễu này. w/o HGT (−0.036, khoảng 2.4 sd) đúng chiều paper. Các chênh lệch còn lại (≤0.012, dưới 1 sd) chưa đủ bằng chứng để kết luận.

## 6. Phân bố dự đoán (CV_type, seed 1234, fold 0)

- v2.0_type_none_s1234: số dự đoán theo từng loại [137, 72, 63, 28], recall theo loại [0.71, 0.38, 0.34, 0.13], best_epoch qua các fold [300, 250, 440, 340, 440]
- v3.2_ref_MDAv3.2-3_type_none_s1234: số dự đoán theo từng loại [21, 295, 813, 364, 254], recall theo loại [0.03, 0.3, 0.51, 0.3, 0.11], best_epoch qua các fold [30, 450, 30, 420, 10]
- v3.2_cuilab_deg2_type_none_s1234: số dự đoán theo từng loại [52, 745, 1508, 116, 328], recall theo loại [0.03, 0.52, 0.62, 0.04, 0.12], best_epoch qua các fold [380, 20, 10, 30, 30]

Trên v3.2, early stopping nhiều khi dừng rất sớm (best_epoch 10–30), vì Top-1 trên tập validation dao động mạnh. Đây là hệ quả của giả định early stopping (mục 3, #12) và có thể làm kết quả v3.2 thấp hơn [suy luận]. Tôi không thay đổi giả định này để tránh tune theo kết quả.
