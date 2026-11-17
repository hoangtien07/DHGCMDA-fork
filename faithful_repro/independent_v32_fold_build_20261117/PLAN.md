# Chuẩn bị đầu vào selection v3.2 cho fold 2–4

Tái sử dụng nguyên vẹn builder `build_candidate_fold_features_rev3.py`, semantic/functional/contracts và các đầu vào từ `colab_package_f0_selection_20261115`, xác minh SHA theo `package_manifest.json` và các pin dùng trong fold0/1. Chỉ xây phase selection cho fold 2, 3, 4 vào thư mục mới; không huấn luyện, không dự đoán/chấm outer-test, không chỉnh nguồn hoặc artifact cũ.

Mỗi bundle phải khớp CSV split theo từng cặp train/val; mask và bốn block association chỉ gồm train, không chứa val/test; nhãn train/val khớp tensor, array hash khớp metadata; sequence/gene khớp strict. Tái dựng độc lập semantic và functional theo train mask, đối chiếu elementwise. Kiểm tra hash, các group không cắt segment và provenance của nguồn tĩnh. Lưu log, tên/hash artifacts, kết quả; nếu lỗi thì giữ dấu vết và dừng, không tự đổi code để ép thành công.

Sau kiểm toán, xác định tình trạng baseline và các ràng buộc lựa chọn/horizon, không coi fold validation là outer-test; không mặc định 650 epoch của fold0/1 là một protocol xác nhận không thiên lệch. Các nhóm miRNA mơ hồ và nguồn đặc trưng ngoài vẫn là giới hạn đã biết.
