# TRACK_N_GATE_REPORT — trạng thái gate sau code-only phase

Base commit: `d205031`. Protocol: REV N.1 (đã khóa tại
TRACK_N_PREREGISTERED_PROTOCOL_REV_N1.md).

| Gate | Yêu cầu | Trạng thái | Bằng chứng |
|---|---|---|---|
| G0 | Protocol revised + fixtures PASS | **PASS** | REV N.1 written (primary=multihot_macroF1 locked, threshold=inner-val grid locked, cardinality check, arm1=BCE-only/PU OFF, stopping rules); baseline contract fixtures 3/3 PASS (`test_baselines_contracts.py`) |
| G1 | Cold-start splits + canaries PASS | **PASS** | `splits/{mirna,disease}_cold_seed{1234,42,2025}.json` + MANIFEST sha256; entity disjointness re-verified from tensor; per-fold per-type support audit in each manifest (minority type support 23–251/fold — đủ để đo nhưng variance cao); `provenance_canary.py` PASS (train-only features provable; external views không phải A-derived) |
| G2 | Baselines verified cùng split/metric | **PENDING-RUN** | `baselines.py` implemented + contract fixtures PASS; chưa chạy real-data baseline cells (logistic fit = training → chờ; priors deterministic có thể chạy bất kỳ lúc) |
| G3 | 1f×1s sanity + independent metric recompute | **BLOCKED** | Cần ít nhất 1 training run (A1 hoặc logistic real run) — yêu cầu owner approval trước huấn luyện |
| G4 | T-pair fleet | **BLOCKED** | Chờ G2–G3 |
| G5 | Cold-start deep jobs | **BLOCKED** | Chờ G4 + stopping rule |

## Quyết định ghi nhận (không phải design change)

- `pair_features`/`label_prop`/`logistic` trong `baselines.py` là
  train-only theo contract signature (nhận Y_train đã mask);
  fixture chứng minh chúng hoạt động — việc cấp Y_train đúng mask
  nằm ở runner (chưa viết, thuộc G2+).
- Minority type (epigenetics) support trên cold-start test folds:
  23–251 mẫu → per-fold estimates noisy; preregistered làm
  descriptive, không cố tăng seed (rule: no seed-escalation).
- Priors là deterministic → chạy được ngay trong G2 mà không cần
  approval thêm; logistic + label-prop cũng chỉ là baseline fit,
  nhưng để an toàn tôi giữ mọi real-data compute sau owner go-ahead
  cho rõ ràng về "huấn luyện".

## Blocker bàn giao

Không blocker khoa học. Chỉ thiếu owner approval cho phần compute
(G2 trở đi). Mọi code-only deliverable đã xong:
protocol REV N.1, splits frozen+audited, canary PASS, baseline
contracts PASS, training budget §7 protocol (tách track, sequential).
