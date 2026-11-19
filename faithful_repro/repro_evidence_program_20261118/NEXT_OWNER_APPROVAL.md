# NEXT_OWNER_APPROVAL — các quyết định chờ owner (REV4.1)

## A. Gửi author request — CHỜ DUYỆT

`AUTHOR_ARTIFACT_REQUEST_READY.md` đã hoàn thiện (trung lập,
học thuật, 5 nhóm artifact: tensor+entity IDs+filter, folds+RNG,
v3.2 eval code+convention, per-fold feature masking, run config).
Hai kênh đã chuẩn bị: GitHub issue trên `CDMBlab/DHGCMDA`
(preferred — public, traceable) hoặc email tác giả.
**Không gửi trừ khi owner lệnh.**

## B. Track N training — CHỜ DUYỆT

Protocol preregistered tại TRACK_N_PREREGISTERED_PROTOCOL.md +
TRACK_N_BASELINE_AND_SPLIT_SPEC.md, đã qua internal review
(TRACK_N_INDEPENDENT_REVIEW.md, verdict: đủ chặt để duyệt).
Nếu duyệt: bắt đầu bằng build cold-start split indices + canary
(code-only), rồi baselines — không có U1 retrain cho T-pair
(reuse frozen).

## C. Không chờ duyệt — đã làm xong

- REV4.1 metric erratum (2 bug sửa + 5 fixtures + recompute +
  erratum doc): claim strata đảo — prior thắng U1 ở multi-type và
  single-type any-hit/acc; K4filtered tăng ~0.05.
- Upstream check: không artifact mới.
