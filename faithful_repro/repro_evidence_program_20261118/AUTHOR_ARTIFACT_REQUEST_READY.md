# AUTHOR_ARTIFACT_REQUEST_READY — REV4 WP-A1

Status: **READY, NOT SENT.** Chỉ owner mới được phép gửi (email
hoặc GitHub issue). Upstream kiểm tra 2026-11-19: `CDMBlab/DHGCMDA`
vẫn ở main@06f9ba3 (2026-01-16), không tag/release/mã mới;
`CDMBlab/MRGBMDAT` chứa dataset khác (853×591 binary, 3 lớp
{-1,1,2}), không phải v3.2 5-type; không tìm thấy repo BGMMDA
public của tác giả.

## Evidence matrix (public artifacts đã kiểm)

| Nguồn | Nội dung | Có v3.2 eval assets? |
|---|---|---|
| Paper BMC 2026 | số liệu + text "four types" | không có supplement data link |
| Author repo (v2.0 release) | pipeline + metric k4 | metric không chấm được v3.2 |
| SPLD MDAv3.2-3 | tensor derivation | có — là nguồn strict tensor của ta |
| MRGBMDAT repo | dataset.rar 853×591×3 lớp | khác benchmark |
| Git history | main frozen 2026-01-16 | không |

## Draft request (5 nhóm artifact)

**Subject:** Data/code request — DHGCMDA v3.2 reproduction (BMC Bioinformatics 2026)

Dear authors, we are independently reproducing DHGCMDA. Public
release covers HMDD v2.0 only; the v3.2 results in the paper
(Top-1 F1 = 0.8600) are not computable from released materials.
Could you share:

1. **Curated v3.2 tensor**: the exact 411×271×5 association tensor
   (or entity ID lists + type mapping + checksum), and the
   preprocessing/filter scripts that produced it from raw HMDD
   v3.2 — in particular the exact minimum-association filter
   (paper text suggests "<2"; SPLD suggests "<3"; neither alone
   reproduces the boundary).
2. **Fold indices**: the five CVtype pair-fold index files (train
   + validation per fold) and the RNG/seed provenance used to
   generate them.
3. **v3.2 evaluation code**: the script producing Table-3 Top-1
   P/R/F1 for v3.2 — specifically the label convention (scalar
   per pair vs multi-hot), the handling of the 5th type, and the
   recall denominator. (Context: the released
   `Calculate_Metrics.py` maps only types 1–4 and drops 5-type
   predictions, so it cannot produce 0.9421 recall on a 5-type
   multi-hot tensor — our oracle bound is 0.8263 for that
   convention.)
4. **Feature construction**: per-fold pair masking and how the
   association-derived similarity views were rebuilt (or whether
   they were) inside each training fold.
5. **Run configuration**: commit/config/epochs for the reported
   v3.2 run; checkpoints or prediction arrays if shareable.

We will register any received artifact by immutable hash and cite
it in the reproduction note.

## Điều kiện gửi

Chờ lệnh owner. Kênh đề xuất: GitHub issue trên CDMBlab/DHGCMDA
(public, traceable) hoặc email tác giả liên hệ trong paper.
