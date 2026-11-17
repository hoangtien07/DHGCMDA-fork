# Đối chứng thay hàng G

Sau kết quả hypergraph.json, giữ weights/input và thay riêng 13 hàng G miRNA của mỗi view từ cũ sang mới hoặc mới sang cũ; tập 13 đã chọn hậu nghiệm theo norm decoder mới. Chạy lại mi_view, decoder, Gram và residual. Ma trận G trộn có thể không đối xứng và không phải Laplacian hợp lệ: chỉ là can thiệp để định vị đường tính toán, không đề xuất áp dụng khi huấn luyện. AVF dùng pooling toàn node, vì vậy thay hàng G có thể ảnh hưởng cả các hàng khác. Hai góc cũ/mới phải khớp residual tham chiếu 1e-5. Không outer-test; chưa có H/Gram trước refresh.
