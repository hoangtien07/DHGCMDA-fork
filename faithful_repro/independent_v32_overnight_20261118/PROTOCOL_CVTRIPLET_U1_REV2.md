# PROTOCOL_CVTRIPLET_U1_REV2 — U1 + existence, task CVtriplet trên v3.2 (predeclared)

Ngày soạn: 2026-11-18. Trạng thái: **REV2 — đáp ứng review CHANGES_REQUIRED; chỉ được phép preflight + sanity fold0; 5×3 và sweep bị chặn**. Audit + phân loại adaptation nằm trong `DESIGN_REVIEW.md` (F01–F04). Đây là experiment bổ sung có kiểm soát, KHÔNG thay thế bằng chứng tương đương paper và không được gọi là tái hiện CVtriplet tác giả.

## 1. Hai task (F01)

- **Task P — pair existence**: `s_pair(i,j)=σ(exist_logit(i,j))`. Positives eval = cặp mà TOÀN BỘ kênh dương nằm ở fold-test (pair chưa thấy); negatives = `pair_zero` sampled.
- **Task T — typed-triplet existence** (primary): `s(i,j,k)=σ(exist_logit(i,j))·softmax(type_logits/T)[k]` — joint decomposition, justification tại DESIGN_REVIEW §3. Positives = triplet fold-test; negatives = unknown triplet sampled (§6).
- Primary metric: **AUROC Task T** (5 seed lấy mẫu, báo mean + per-seed). Secondary: AP, F1@0.5 cố định, metrics theo stratum. KHÔNG so paper, KHÔNG phải Top-1 F1 CVtype.

## 2. Split (pin)

- Universe `U` = `argwhere(Y)` trên `v3.2_ref_MDAv3.2-3_paperfirst_strict_4view.npz` (byte-pinned), |U|=11.748.
- Partition 5 fold: `order=default_rng(0).permutation(|U|)`, `parts=array_split(order,5)` — **giống hệt `triplet_baseline.py`** để so đối chứng. Hash `parts` ghi vào contract.
- Fold f: `test = parts[f]`; `train_all = U \ test`. Inner split cho selection U1: `inner_val = default_rng(2000+f).choice(train_all, size=ceil(0.1·|train_all|), replace=False)`; `inner_train = train_all \ inner_val`.
- `train_pairs` = tập cặp của `inner_train` (train_mask ≡ cells này — giữ schema runner). `y_train` = multi-hot CHỈ trên kênh inner_train (kênh held-out = 0 — held-out label canary).
- `val_pairs` = tập cặp của `inner_val` **trừ** train_pairs (runner gate bắt disjoint); `y_val` = multi-hot kênh inner_val. Triplet inner_val rơi vào cặp đã ở train bị loại khỏi val — đếm `straddle_innerval_dropped` vào contract.
- `test_pairs`/test triplets KHÔNG vào bundle (FORBIDDEN keys); eval đọc lại từ `split.py` + strict npz.

## 3. Bundle (pin)

- Cùng công thức views như builder REV3: `mi_v1=[S_seq|A]`, `mi_v2=[S_func(A)|A]`, `dis_v1=[S_gene|Aᵀ]`, `dis_v2=[S_sem|Aᵀ]`, A=train_mask; S_seq/S_gene/sim từ strict npz; `S_func=functional_similarity(A,sem).sim` rebuild train-only; `S_sem` từ MeSH-2019 pinned (semantic.py + HMDDv3.xlsx + meshToID.csv + mtrees2019.bin) — mọi input băm trước/sau.
- Không động vào bundle/artifact cũ; output namespace mới `triplet_u1/bundles/foldN/` kèm `split_sidecar.json` (index mảng + hash từng segment + straddle counts).

## 4. Model U1X (pin)

- `model_u1x.py`: `PaperGuidedU1X(PaperGuidedC4)` + `exist_rel ∈ R^readout`, init `randn·√(2/readout)` bên trong `torch.random.fork_rng(devices=[])` + `manual_seed(seed+4007)` ⇒ param baseline C4 + RNG toàn cục giữ bitwise như U1 (fixture G2 chứng minh: zero exist_rel ⇒ logits C4-identical).
- Forward trả thêm `exist[n_pairs] = σ(mi_proj(z_mi)[i]·(exist_rel∘dis_proj(z_dis)[j]))` — port L490–493 upstream trên cùng projections của readout.

## 5. Loss + supervision (pin)

- `L_exist`: focal upstream — pos: `-(1-p)² log p` trên train-mask cells; neg: `α·(-p² log(1-p))`, α=0.5 (param default upstream); chia (n_pos+n_neg). Pos = cells train_mask; neg = **subsample MỘT lần** `default_rng(seed+77)` size `min(10·n_pos, |mask-zero|)` — hash index trong contract (port ngữ nghĩa L653–665; rng mới — A4).
- `L_total = combine_objectives(U1 nguyên vẹn) + lam_exist·L_exist`, `lam_exist=0.3` — adaptation đơn vị (A5, upstream `exist_weight`); mọi hệ số U1 giữ nguyên từ cfg cũ (`lam_intra=1.0, lam_inter=0.3, lam_recon=1.0, lr=1e-4, wd=1e-5, 650 epoch, eval/10, β=0.99999, seeds {1234,42,2025}`, threads=2).
- Inter-bank (`inter_cap_mult`) giữ giá trị của arm đang chạy; cap1 đã âm tính nên production đề xuất `inter_cap_mult=10`; sanity dùng cfg y hệt production — KHÔNG sweep.

## 6. Eval (pin)

- Test positives Task T = `parts[f]`; Task P positives = pair có mọi kênh dương ∈ test (đếm vào contract).
- Unknown universe `U0=argwhere(~Y)` (|U0|=545.157). Eval unknowns Task T: 5 lần `default_rng(seed_eval)` với `seed_eval∈{0..4}` (port upstream 5 seeds), mỗi lần lấy `|test|` không lặp; Task P: từ `pair_zero` (Y[:,:]=0 ∀k — subset U0 theo cặp).
- Metrics: AUROC/AP/F1@0.5 trên joint score (Task T) và `s_pair` (Task P); strata per §5 DESIGN_REVIEW (`triplet_zero_on_train_pair`, `triplet_zero_on_heldout_pair`, `pair_zero`; pos `pair_seen`/`pair_unseen`).
- Logistic baseline train-only: features count của `triplet_baseline.py` (train-only counts + leave-one-out trên train), fit inner_train + unknown sample `default_rng(3000+f)` 1:1, chấm cùng eval sets — info-condition chuẩn hoá (F03).
- Canary (fail nếu vi phạm): ∀(i,j,k)∈inner_val∪test: `y_train[i,j,k]==0` và `y_val[i,j,k]==0`; ∀neg sampled: Y[i,j,k]==0; train/val/test disjoint ở mức triplet.
- Metric verification: perfect scorer→AUROC=AP=F1=1.0; đảo score→AUROC=0; 1 case tay nhỏ so khớp sklearn.

## 7. Phạm vi cho phép ở REV2

- Được: `fixtures.py` (unit), `build_bundles.py` (5 bundle), `preflight.py`, sanity `train.py` fold0-seed1234 (650 epoch, một lượt), `eval_cvtriplet.py`, `baseline_logistic.py`, `verify.py`, report.
- Bị chặn: mọi lượt khác ngoài sanity (fold≠0 hoặc seed≠1234), sweep hyperparam, mở rộng 5×3, chỉnh protocol sau khi thấy kết quả eval, dùng kết quả sanity để tuyên bố hiệu năng (sanity chỉ xác thực thiết kế: loss hữu hạn, exist channel phân tách pos/neg, pipeline chạy hết).
- Kết quả sanity BÁO NGUYÊN, kể cả âm tính/không phân biệt được — không chỉnh để đẹp.

## 8. Tiêu chí dừng

Preflight FAIL / fixture FAIL / bundle lệch contract / loss nonfinite / exist channel không vượt prior trên inner_val ở sanity ⇒ dừng và báo; không tự vá. Full run chỉ sau khi REV2 được duyệt lại.
