# FINAL_COUNCIL_HANDOFF — DHGCMDA REV4 (2026-11-19)

Branch `devin/20261118-repro-evidence-program`, namespace
`faithful_repro/repro_evidence_program_20261118/`.

## 1. Verdict

`R_CLOSE_AS_PARTIAL_REPRODUCTION` — Phase A hoàn tất, Phase B
SKIP (Gate A3: NO_TRAIN_NECESSITY — câu hỏi B1 đã answered bằng
paired artifacts sẵn có), Phase C record tại
COUNCIL_DECISION_RECORD.md. Không training mới nào trong REV4.

## 2. Phát hiện mới của REV4 (ngoài program 20261118)

1. **Erratum density**: "4.6%" sai — strict tensor là **10.5476%**
   typed-link, khớp paper; per-type proportions khớp bảng paper.
   Mọi claim dựa trên density gap đã RETRACTED (DENSITY_ERRATUM).
2. **Dataset identity nâng cấp**: strict tensor rất có thể chính
   là benchmark paper (L1 exact match). G1 thu hẹp: còn thiếu
   eval-input (folds/features/eval script), không phải tensor.
3. **Metric feasibility proof**: oracle hoàn hảo chỉ đạt
   macro-recall 0.8263 / micro 0.7435 trên tensor này với
   1-pred-per-pair → R=0.9421 của paper **bắt buộc** convention
   khác (scalar-collapse/any-hit/đơn vị khác). Bất khả so sánh
   cấu trúc, độc lập dataset (fixture PASS).
4. **Leakage isolation (paired)**: v2.0 leak +0.049 F1, v3.2
   leak ~0 — paired same-config, không cần chạy B1.
5. **Metric bridge**: thứ hạng model-vs-prior đảo theo
   convention — không một headline số duy nhất (METRIC_BRIDGE).
6. **Upstream frozen**: main@06f9ba3, MRGBMDAT dataset khác
   (853×591×3 lớp), không artifact v3.2 mới.

## 3. Deliverables REV4 (mới)

DENSITY_ERRATUM, DATASET_IDENTITY_EVIDENCE, METRIC_FEASIBILITY
(+ `metric_feasibility_check.py`), AUTHOR_ARTIFACT_REQUEST_READY,
METRIC_BRIDGE.json + METRIC_BRIDGE_REPORT (+ `metric_bridge.py`),
FIDELITY_DELTA_RANKING, COUNCIL_DECISION_RECORD,
NEXT_RESEARCH_DECISION, EVIDENCE_MAP, FINAL_COUNCIL_HANDOFF.

## 4. Giới hạn

- Paired leak Δ chỉ 1 seed; retrospective everywhere; không
  p-values; không blind replication.
- Eval-input author chưa byte-verified; paper convention không
  biết chắc (4 khả năng trong METRIC_FEASIBILITY).
- File này không khép "scientific closure": G1–G4 phụ thuộc
  artifact ngoại sinh chưa có.

## 5. Next owner action (tối đa 2)

1. Cho phép gửi `AUTHOR_ARTIFACT_REQUEST_READY.md` (issue/email).
2. Nếu muốn Track N: duyệt việc soạn protocol preregistered
   (cold-start splits + locked metrics + baselines) —
   NEXT_RESEARCH_DECISION.md.

## 6. Final phrase

REV4_PHASE_A_COMPLETE; Phase B SKIPPED (already answered);
recommendation recorded.
