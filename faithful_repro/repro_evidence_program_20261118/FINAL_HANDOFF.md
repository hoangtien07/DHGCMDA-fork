# FINAL_HANDOFF — DHGCMDA Repro Evidence Program 2026-11-18

Branch: `devin/20261118-repro-evidence-program` (base pin `a6cf4e9`).
Namespace: `faithful_repro/repro_evidence_program_20261118/` (R/).

## 1. WHAT_WAS_DONE

WP0–WP8 thực thi đầy đủ, tự kiểm tra qua các gate:

- WP0: inventory 20 evidence rows (E01–E20), duplication matrix
  (D1–D7), STATUS/DECISION_LOG.
- WP1 (Gate B PASS): metric + split contract library + 14 fixtures
  — k4 release byte-equivalent, harmonic ≠ macro counterexample,
  pair vs triplet non-equivalence + disjointness thật.
- WP2: data lineage audit (raw→SPLD→reference→strict), feature
  provenance matrix, draft artifact request.
- WP3: fidelity trace paper↔author↔U1 + 6 micro-fixtures +
  contradiction/errata ledger (C1–C8).
- WP4 (Gate C PASS): rescore 30 eval npz + 4 priors từ row-level —
  50/50 verification match; RESULTS_CROSSWALK + REEVALUATION.
- WP5: REV3 protocol lock + necessity audit → ALREADY_ANSWERED.
- WP6 (Gate E): GATE_REPORT — E3 fails → TRAINING_SKIPPED_WITH_REASON.
- WP7: 4 báo cáo khoa học tiếng Việt + adversarial review.
- WP8: handoff này + FINAL_VALIDATION + STATUS cập nhật.

Không huấn luyện mới nào trong program này.

## 2. SCIENTIFIC_VERDICT

**PARTIAL_REPRODUCTION_WITH_EXTERNAL_BLOCKERS.**

- 0.8600 KHÔNG tái hiện và KHÔNG thể xác nhận/bác đúng nghĩa:
  tensor paper, fold, eval code đều chưa public.
- Trên tensor strict + PAIR_CVTYPE sạch + metric khóa: U1 đạt
  harmonic Top-1 ≈ 0.4558–0.4924 (retrospective).
- Metric release k4 loại hết v3.2 → mọi "collapse = 0.0" lịch sử
  là artifact, đã chứng minh.
- Một số kết quả nghịch lý thật: priors ≈ model trên aggregate;
  static ≥ dynamic; ablation đảo chiều trên v2.0 shipped.

## 3. KNOWN_LIMITATIONS

- Mọi số là post-dev (retrospective), không blind, không p-value.
- Density tensor 4.6% ≠ 10.5% paper; label space 5-type ≠ 4-type.
- Filter `<2`/`<3` của paper UNRESOLVED (cả hai đọc literal đều
  không tái lập được 11,748).
- v2.0 raw predictions UNAVAILABLE_LOCAL → not rescoreable.
- U1/U1X là declared adaptations, không author-faithful.

## 4. EVIDENCE_TRACE_TABLE

| Claim | Artifact | Key |
|---|---|---|
| Tensor integrity | data_audit.json (strict vs reference) | sha 85bb9ea4 |
| Fold legality | test_split_contracts.py | 3/3 PASS |
| Metric semantics | test_metric_contracts.py | 11/11 PASS |
| Loss fidelity | test_fidelity_microfixtures.py | 6/6 PASS |
| Rescore integrity | RESCORE_VERIFICATION.json | 50/50 PASS |
| Kết quả U1 | RESULTS_CROSSWALK.csv | 50 rows |
| Gate training | GATE_REPORT.json | TRAINING_SKIPPED |
| CVtriplet | triplet_u1/RESULTS_CVTRIPLET_U1_5X3.md | negative |

## 5. NEXT_OWNER_DECISION (tối đa 2 lựa chọn)

1. **Gửi ARTIFACT_REQUEST_TO_AUTHORS.md** (draft sẵn trong R/) —
   chờ tác giả phát hành 3 artifact để đóng G1–G4 và tái đo 0.86
   trực tiếp. Không gửi mà không có lệnh.
2. **Đóng gói ReScience-style reproduction note** từ
   THESIS_RESULTS_AND_LIMITATIONS_VI.md + FINAL_RESEARCH_VERDICT —
   công việc đã đủ chặt cho một báo cáo học thuật về partial
   reproduction + release-eval disconnect.

## 6. FILES_AND_COMMITS

- Branch: `devin/20261118-repro-evidence-program`
- Commits: `6b18ab7` (WP0+WP1), `1b9a3c3` (WP2+WP3+WP4-core),
  final (WP5–WP8, xem `git log`).
- Mọi artifact lịch sử giữ nguyên — không overwrite.

## 7. FINAL_PHRASE

REPRO_EVIDENCE_PROGRAM_COMPLETE
