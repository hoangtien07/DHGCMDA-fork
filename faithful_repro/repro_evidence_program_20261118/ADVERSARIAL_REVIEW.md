# ADVERSARIAL_REVIEW — đối chiếu phản biện kết quả nội bộ

Second pass trên toàn bộ WP0–WP6 artifacts. Mỗi mục là claim có
thể bị tấn công và cách ta phòng thủ (hoặc chịu).

| # | Điểm tấn công | Phản biện / phòng thủ | Trạng thái |
|---|---|---|---|
| A1 | "0.49 thấp hơn 0.86 → model/paper bị bác bỏ" | Không: tensor khác (density 4.6 vs 10.5%), metric khác (5-type multi-hot vs 4-type), fold khác. Không so sánh trực tiếp. | DEFENDED |
| A2 | "Model học leakage qua feature" | Đã audit: similarity không derive từ association matrix; mask train-only cho phần association-derived. Post-hoc count-priors cao vì task mang degree prior, không phải leak. | DEFENDED (nhưng oracle prior trong triplet audit là tách riêng) |
| A3 | "Kết quả retrospective → overfit horizon" | Đúng một phần: horizon 650ep post-dev; chỉ khắc phục bằng blind fold mới — chưa có, ghi nhận. | ACCEPTED_LIMITATION |
| A4 | "Rescore tự viết metric → có thể sai" | Verified: rescored macro_f1 khớp bit-exact số reported gốc 50/50; k4 equivalence với author class trên fixture; sklearn agreement trên micro/any-hit. | DEFENDED |
| A5 | "Static ≥ dynamic → kết luận dynamic vô ích" | Chỉ trong config này (θ=.5, Gram, rebuild H); không suy rộng sang cơ chế paper (G-matrix vs edge_index khác tầng). | DEFENDED |
| A6 | "Priors ≈ model → model vô dụng" | Priors thắng aggregate, model thắng macro-F1 trên minority types. Hai metric trả lời hai câu hỏi khác nhau; kết luận cần nêu cả hai. | DEFENDED |
| A7 | "UNRESOLVED data filter = chưa tái hiện được data paper" | Đúng — đây là hard blocker G1/G2, đã liệt kê artifact request; không lấp bằng suy đoán. | ACCEPTED_LIMITATION |
| A8 | "Fold0 viewed early → contamination" | Đúng, nhưng fold0 chỉ là 1/15 ô; paired differences ổn định ở cả 5 fold → kết luận directional đứng. | DEFENDED |
| A9 | "Mismatch U1 vs author loss → không đo được paper" | Đúng — U1 declared adaptation; fidelity trace phân loại từng delta; không claim author-faithful ở đâu không chứng minh được. | DEFENDED |
| A10 | "Causal claims 'gap dominated by G1–G4'" | Đã sửa: không đo isolate → chỉ nói blocker ngoại sinh, không xếp nhân quả. FINAL_RESEARCH_VERDICT ghi rõ. | FIXED |

## Risk register còn mở

- R1: Nếu tác giả phát hành tensor/fold/eval → một số kết luận
  "external blocker" có thể cần tái đo (đã chuẩn bị protocol
  REV3 + request letter).
- R2: Density tensor 4.6% làm task khó hơn — nếu paper chứng minh
  được curation đúng, kết quả có thể cao hơn, nhưng vẫn trên
  tensor khác.
- R3: Một số artifact Windows-only (v20 runs) → crosswalk
  đánh dấu UNAVAILABLE_LOCAL, không suy ra.


---

> **[ERRATUM 2026-11-19, REV4]** Mọi phát biểu "density ~4.6% vs 10.5%" trong file này đã RETRACTED — strict tensor đo thực tế 10.5476% typed-link (khớp paper). Xem `DENSITY_ERRATUM.md`.
