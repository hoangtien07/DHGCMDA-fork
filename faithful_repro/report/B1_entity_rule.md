# B1: Quy tắc chọn 411 miRNA × 271 bệnh — KẾT LUẬN: không tìm ra bằng quy tắc thuộc tính đơn giản

Script: `code/entity_rule.py`, `code/entity_rule2.py`. Output: `report/entity_rule.txt`, `report/entity_rule2.txt`.
Ứng viên: toàn bộ entity trong bộ file CUILab không tiền tố (1,151 miRNA, 802 bệnh). Đây là bộ chứa đủ 11,748/11,748 triplet của artifact (xem `ref_provenance.txt`).

## Điều kiện cần (đã đo, đúng với 100% entity trong artifact)
- miRNA: tổng degree thô ≥3, có ≥1 association loại circulation, có tên trong MISIM.
- Bệnh: tổng degree thô ≥3, tên khớp MeSH 2025 (heading hoặc entry), có cây MeSH nhánh C.

## Điều kiện đủ: không tìm thấy
| Phép thử | miRNA (411 trong artifact) | Bệnh (271 trong artifact) |
|---|---|---|
| Decision tree depth 3 (train accuracy, fit trên chính dữ liệu, chưa có kiểm tra ngoài) | 0.973 (FP 25, FN 6) | 0.889 (FP 89, FN 0) |
| Decision tree depth 5 | 0.983 (FP 15, FN 4) | 0.913 (FP 15, FN 55) |
| Quét toàn bộ luật dạng "degree từng type ≥k, với mọi tập con type" + degree tối thiểu (bệnh có thêm điều kiện cây C) | tốt nhất 85 lỗi | tốt nhất 122 lỗi |

- Không có luật nào tái tạo đúng tập entity của artifact. Decision tree depth 5 đạt độ chính xác cao hơn, nhưng đó là fit trên chính dữ liệu cần giải thích: luật thu được không đơn giản và không ứng với bước xử lý nào được mô tả [suy luận].
- Có các bệnh degree rất cao vẫn bị loại, ví dụ gastric neoplasms (535), carcinoma lung non-small-cell (393), alzheimer disease (135). Một bộ lọc thuần theo degree hay độ phủ MeSH không thể loại các bệnh này.
- Vì vậy nhiều khả năng có bước lọc ngoài các thuộc tính đã thử: theo thời gian (năm PMID), theo danh sách bệnh lấy từ nguồn khác, hoặc chọn tay [suy luận].
- **Chưa thử:** lọc theo năm PMID, vì máy không có ánh xạ PMID → năm offline; cần PubMed API.
- Bản fork (CLAUDE.md, Plan K) cũng không tái tạo được bộ này: min-assoc ≥7 cho 391×332, nhất quán với kết quả trên.

**Phán quyết:** không đủ bằng chứng để tái tạo tập entity. Bằng chứng duy nhất có thể đóng gap này là tác giả công bố danh sách entity hoặc script tiền xử lý. Xem email nháp ở `report/email_authors.txt`.
