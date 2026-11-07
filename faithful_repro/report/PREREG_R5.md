# Đăng ký trước Round 5 (viết trước khi có kết quả, 2026-11-06)

Primary = cấu hình faithful Round 3/4, không đổi. Mọi mục dưới đây là thí nghiệm sensitivity hoặc kiểm tra claim. Không mục nào thay thế kết quả chính, kể cả khi cho điểm cao hơn.
Áp dụng chung: seeds {1234, 1, 2} × 5 fold; dùng chung split của faithful.make_splits; metric như Round 4 (macro và micro Top-1; CV_triplet lấy trung bình 10 lần rút negative), thêm f1_valthr.

| ID | Thí nghiệm | Dataset | Protocol | Câu hỏi | Tiêu chí kết luận (định trước) |
|---|---|---|---|---|---|
| B2 | ref_4view (m_ss + mi_fun, d_gs + DSSM) | ref 411×271 | type, triplet | Thêm view sequence và gene có thu hẹp gap trên chính dữ liệu của tác giả không | Kết luận "có tác dụng" chỉ khi |Δ mean| > 2×sd giữa các seed và cùng dấu ở cả 3 seed |
| B7 | TDRC (r=4, α=0.125, β=0.25, λ=0.001, tol 1e-6, max_iter 500, đúng mặc định repo) | v2.0, cuilab, ref | type, triplet | Claim "DHGCMDA vượt TDRC" còn đúng không khi chạy cùng split và cùng metric | So mean±sd của từng metric; không kiểm định ý nghĩa khi chỉ có n=3 seed |
| B3 | recon_reduction=mean | v2.0, cuilab, ref | type, triplet | Cách đọc Eq.21 theo code tác giả ảnh hưởng thế nào | Báo cáo song song với primary (sum); không chọn |
| B4 | predictor=full | v2.0, cuilab, ref | type, triplet | Cách hiểu "bilinear" = W_t đầy đủ ảnh hưởng thế nào | Như B3 |
| B9a | 5 ablation | v2.0, ref | type, triplet | Kiểm tra claim Fig.4: bỏ module nào cũng làm giảm | Xác nhận từng module khi Δ < 0 ở cả 3 seed và |Δ| > 2×sd |
| B9b | K ∈ {1,3,5,7,9,11,13,15} | v2.0 | type | Kiểm tra claim Fig.3: Top-1 đạt đỉnh tại K=13 | Có đỉnh tại 13 không (argmax của mean); hình dạng đường cong |
| B9c | λ2 ∈ {0.1,0.3,0.5} × t ∈ {0.5,1,3} | v2.0 | type | Kiểm tra claim Fig.2: tối ưu tại (t=0.5, λ2=0.3) | argmax của mean; không dùng để chọn lại tham số |
| B8 | f1_valthr | mọi job triplet | triplet | Mức lạc quan của F1 khi chọn threshold trên chính tập test | Báo cáo chênh lệch f1 − f1_valthr |

Thứ tự chạy: B2+B7 → B3+B4 → B9a → B9b/c. Nếu thiếu thời gian tính toán thì cắt từ cuối danh sách lên, và ghi rõ phần đã bị cắt.
