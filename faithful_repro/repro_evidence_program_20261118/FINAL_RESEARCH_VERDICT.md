# FINAL_RESEARCH_VERDICT — kết luận tái hiện DHGCMDA

## Verdict

**PARTIAL_REPRODUCTION_WITH_EXTERNAL_BLOCKERS.**

- KHÔNG tái hiện được 0.8600. Không thể chứng minh tương đương
  dataset/fold/metric của paper — ba blocker ngoại sinh (tensor
  paper-first 411×271, fold split, eval code chính thức) đều chưa
  public (ARTIFACT_REQUEST_TO_AUTHORS.md soạn sẵn, chưa gửi).
- Trên tensor strict thắt provenance (elementwise-identical với
  bản reference SPLD), PAIR_CVTYPE sạch leakage, U1 đạt
  **GENERAL_K_TOP1_HARMONIC ≈ 0.46–0.49** (post-dev, retrospective).
- Định nghĩa metric di chuyển headline ~0.05 (0.40→0.46 dynamic) —
  mọi so sánh lịch sử phải tra RESULTS_CROSSWALK thay vì một con số.
- Metric k4 của release tác giả loại hết mẫu 5-type → mọi
  "v3.2 Top-1 = 0.0" trong lịch sử repo là artifact, không phải
  collapse model (đã chứng minh 3 cách độc lập).
- Đảo chiều ablation của paper (ablation > baseline trên v2.0
  shipped) giữ nguyên sau verify — finding chính đáng.
- CVtriplet-U1X là negative experiment có kiểm soát (baseline
  train-only vượt model 0.064 AUROC ở cùng info).

## Không được suy ra

- Không khẳng định paper gian lận hay đúng: thiếu artifact để phán.
- Không kết luận "gap do G1–G4" — không có isolating experiment;
  mỗi blocker đủ sức chiếm phần lớn gap một mình.
- Không nhầm 0.49 với "đã tái hiện một phần 0.86" — khác tensor
  density (~4.6% vs 10.5%), khác label space (5 vs 4-type metric),
  khác exposure.

## Điểm đóng được trong tương lai

Với 3 artifact từ tác giả (tensor 411×271 đúng pipeline, fold
indices, eval script): tái đo 0.86 trực tiếp — câu hỏi đóng hoàn
toàn, chi phí ~1 buổi tính.


---

> **[ERRATUM 2026-11-19, REV4]** Mọi phát biểu "density ~4.6% vs 10.5%" trong file này đã RETRACTED — strict tensor đo thực tế 10.5476% typed-link (khớp paper). Xem `DENSITY_ERRATUM.md`.
