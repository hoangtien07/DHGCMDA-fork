# Tái hiện DHGCMDA: bản cài đặt theo đúng paper (vòng 2)

## Kết luận

**Không tái hiện được.** Đã sửa code cho đúng thuật toán của paper, bỏ rò rỉ dữ liệu test, và chạy đúng 2 giao thức CV_type và CV_triplet. Trên cả 3 bộ dữ liệu, không metric nào đạt mức paper báo.

| Claim của paper | Paper | Đo được (faithful, 3 seed × 5 fold, mean±sd giữa các seed) | Kết luận |
|---|---:|---:|---|
| v2.0 CV_type Top-1 P/R/F1 | 0.5842 / 0.6341 / 0.5970 | 0.4010±0.0112 / 0.3788±0.0236 / **0.3878±0.0151** | Không tái hiện được (F1 −0.209) |
| v2.0 CV_triplet AUPR/AUC/F1 | 0.9738 / 0.9669 / 0.9278 | 0.9218±0.0041 / 0.9237±0.0031 / 0.8600±0.0031 | Không tái hiện được (−0.04 đến −0.07) |
| v3.2 (cuilab, lọc đúng như paper ghi) CV_type | 0.7915 / 0.9421 / 0.8600 | 0.2916±0.0232 / 0.2341±0.0062 / **0.2567±0.0101** | Không tái hiện được |
| v3.2 (cuilab) CV_triplet | 0.9271 / 0.9181 / 0.8674 | 0.8964±0.0030 / 0.9025±0.0026 / 0.8322±0.0025 | Không tái hiện được (−0.016 đến −0.035) |
| v3.2 (MDAv3.2-3, bản 411×271 paper dùng) CV_type | như trên | 0.3130±0.0224 / 0.2486±0.0084 / **0.2727±0.0111** | Không tái hiện được |
| v3.2 (MDAv3.2-3) CV_triplet | như trên | 0.8701±0.0026 / 0.8798±0.0015 / 0.8100±0.0014 | Không tái hiện được |
| Bộ v3.2 có 411×271×11,748 khi tải từ cuilab và lọc theo quy tắc paper ghi | 411×271×11,748 | 721×609×17,607 | Không tái hiện được |
| Fig.4: bỏ bất kỳ module nào cũng làm kết quả giảm | mọi biến thể đều < full | So cùng seed 1234: bỏ HGT (−0.036) và bỏ HGCN (−0.012) làm giảm, đúng chiều paper; bỏ AVF (+0.067), bỏ DV (+0.011), bỏ CL (+0.005) lại làm tăng (1 seed, bảng ở mục 5) | Bị phản bác ở AVF; HGT đúng chiều; HGCN, CL, DV chưa đủ bằng chứng |
| Fig.2 / Fig.3 (sweep λ, t, K) | — | Vòng này chưa chạy | Chưa đánh giá |

**Hai yếu tố lệch so với paper đã đo được (chạy chẩn đoán 1 seed, KHÔNG phải kết quả chính):**
- **Thang đo của recon loss.** Eq.21 viết theo đúng chữ là ‖S−Ŝ‖_F², tức tổng trên khoảng 495² phần tử. Loss này lớn cỡ 4,000–28,000 và lấn át L_type: type loss chỉ giảm từ 1.27 xuống 1.19 sau 300 epoch.
  - Nếu thay tổng bằng trung bình (cùng thang với MSE trong code tác giả), Top-1 F1 trên v2.0 tăng từ 0.388 lên 0.541. Con số này vẫn thấp hơn paper (0.597) và thấp hơn code gốc của tác giả (0.557).
  - Tôi không dùng biến thể này làm kết quả chính, vì paper ghi rõ là ‖·‖_F².
- **Rò rỉ dữ liệu test.** Bật lại rò rỉ như code tác giả (`--diag_leak 1`) thì Top-1 F1 trên v2.0 tăng từ 0.388 lên 0.458, và AUPR CV_triplet tăng từ 0.922 lên 0.943. Như vậy một phần điểm cao của code gốc đến từ việc dữ liệu test lọt vào feature và graph [suy luận, dựa trên số đo này].

**Kết quả code gốc của tác giả (vòng 1) không được tính là tái hiện.** Code gốc cho AUC 0.9749 và Top-1 F1 0.5570, nhưng có rò rỉ dữ liệu test và dùng giao thức không phải CV_triplet. Vì vậy mức khớp phần binary ở vòng 1 không phải là tái hiện hợp lệ claim CV_triplet của paper.
