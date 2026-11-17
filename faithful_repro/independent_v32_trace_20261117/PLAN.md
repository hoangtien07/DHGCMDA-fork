# Kiểm toán v3.2 theo cặp–loại và đặc trưng fold0 U1

Chỉ đọc dữ liệu, nhãn và bundle train/validation; không huấn luyện, chấm test, thay split hay sửa đầu vào. Ghi SHA-256 các tệp dùng, đối chiếu elementwise tensor SPLD, CSV triplets, tên và thứ tự loại với reference và strict 4-view; xuất ledger từng triplet có tên, segment fold0, tier MISIM. Không coi tên “paper-exact” trong manifest là chứng cứ đồng nhất với dữ liệu tác giả DHGCMDA.

Kiểm tra bundle đúng file được driver U1 pin, hash từng array so với metadata/expected, mask bằng tập train và không giao val, nhãn train/val khớp Y, bốn khối association bằng mask/transpose; đối chiếu nguồn đặc trưng sequence/gene strict, S_func với MISIM tĩnh. Kiểm tra cặp thuộc input MISIM bằng ledger curate đã lưu, tính lại phân bố theo split U1 thực tế (không dùng nhầm split cũ). Kết quả tier same-MeSH và heuristic không chứng minh identity ở mức tên, hiện diện cặp không chứng minh nhãn loại hay đóng góp tới score. Dữ liệu cặp test chỉ để kiểm toán danh tính và provenance, không chấm model.

Xuất summary JSON, ledgers CSV và báo cáo tiếng Việt; kiểm tra ranh giới nguồn/uncertainty, và ghi nhận phiên bản pipeline/đường chạy khác có thể khác.
