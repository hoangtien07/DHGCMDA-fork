# Kiểm tra split theo entity — predeclared

Trên mỗi bundle selection, kiểm tra pair train và validation không trùng; đếm cặp validation có miRNA, disease hoặc cả hai chưa xuất hiện trong train, và số unique entity train/validation. Bậc dùng đúng train pairs; không đọc nhãn outer-test. Từ đó phân biệt pair-level split với đánh giá cold-start; không suy hiệu năng lâm sàng hay độ giống split tác giả.
