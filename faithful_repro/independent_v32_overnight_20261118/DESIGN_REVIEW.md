# DESIGN_REVIEW — U1 + CVtriplet (REV2)

Ngày: 2026-11-18. Phạm vi: audit chênh lệch giữa mã tác giả v2.0 đã pin (commit `06f9ba3`, bản sao tại `../independent_v32_author_review_20261117/upstream_*.py`), `test_optimized`/eval pipeline, và task CVtriplet trên tensor v3.2 đa-nhãn; phân loại từng adaptation cho arm U1+existence. Không khẳng định author-faithful ở mục nào chưa có bằng chứng.

## 1. Semantics mã tác giả v2.0 (bằng chứng từ source pin)

| Thành phần | Mã tác giả | Vị trí chứng cứ |
|---|---|---|
| Predictor | `SimplifiedTypePredictor` → `[mi, dis, 1+K]`; kênh 0 = `sigmoid(mi_feat·diag(r_exist)·dis_feat)`; kênh 1..K = `softmax(type_logits/T)`, T học được init 2.0 clamp [0.5,5.0]; `type_rel` khởi tạo trực giao × √(2/hidden); `r_exist` = randn×√(2/hidden) | `upstream_hetero_model.py` L417–515 |
| Existence loss | focal γ=2.0: `-(1-p)^γ log p` (pos) + `α·(-p^γ log(1-p))` (neg), α=`args.alpha`=0.5, tổng/(n_pos+n_neg) | `upstream_main_experiments_hetero1.py` L115–133, `param.py` L39–43 |
| Type loss | `F.cross_entropy` trên positive cells, EN weights β=0.99999, `label_smoothing=0.1`; tổng `0.3·exist + 0.7·type` | ibid. L64–78, L135–161 |
| Negative supervision | `zero_index` (ô 0 của ma trận train), subsample MỘT lần ≤10× số positive qua `torch.randperm`, cố định suốt training | ibid. L653–665 |
| Eval CVtriplet | `test_optimized`: `true = data[5][test_idx]` (giá trị ô đầy đủ); `evaluate_optimized`: 5 lần lặp seed 0–4, `np.random.seed(seed)` + shuffle, lấy zero-cells (true==0) đúng bằng số positives; binary metrics trên kênh 0 | ibid. L913–995, L1028–1093 |
| Split | fold theo Ô (one_index) của v2.0 — mỗi ô nonzero có đúng 1 type ⇒ ô ≡ pair ≡ triplet | fold data tác giả |

Hệ quả then chốt: **trong v2.0 pair-existence ≡ triplet-existence** vì mỗi cặp (i,j) có tối đa 1 type. Sang v3.2 đa-nhãn (2.110/8.735 cặp nhiều type) hai task tách nhau — đây chính là F01.

## 2. Chênh lệch U1 hiện tại vs upstream

| Mục | U1 (`DiagBilinearType`, `model.py` L160–188) | Upstream | Phân loại |
|---|---|---|---|
| Đầu ra | chỉ `logits[n_pairs,5]` — KHÔNG có kênh existence | `[…,1+K]` | thiếu kênh — cần port |
| Bilinear type | `z_mi·(r_t∘z_dis)/T`, T init 2.0 clamp [0.5,5] | cùng cấu trúc trên `mi_feat`/`dis_feat` đã project | port trung thực (đã có) |
| init r_exist | — | `randn·√(2/hidden)` | port: `randn·√(2/readout)` trong `fork_rng(seed+4007)` (bảo toàn bitwise C4 params) |
| Loss type | multi-label softmax CE chuẩn hoá theo weighted-positive-mass, EN weights β=0.99999 sum=C, KHÔNG label smoothing | CE + EN weights + LS=0.1 | **adaptation** (đa-nhãn) — đã khai báo trước đây |
| Các objective khác | intra/inter InfoNCE, recon, dynamic graphs — upstream KHÔNG có trong loss này | không có | U1 bổ sung riêng — giữ nguyên, không đổi |

## 3. F01 — Tách task pair-existence vs typed-triplet-existence

**Định nghĩa (pin):**
- Task P (pair existence): `s_pair(i,j) = σ(exist_logit(i,j))` — đúng kênh 0 upstream. Đối tượng (i,j) nguyên cặp.
- Task T (typed-triplet existence): `s(i,j,k) = σ(exist_logit(i,j)) · softmax(type_logits/T)[k]` — **không phải** pair score trần.

**Justification cho joint score** [suy luận có căn cứ]: output upstream chính là phân tích này — `cat([σ(exist), softmax_k])` tương đương `P(pair exists)·P(type k | pair exists)`. Tích này là xác suất đồng thời "cặp có liên kết ∧ loại k" theo đúng mô hình đã học — hợp lệ làm triplet-specific score. Điểm yếu đã khai báo: softmax chuẩn hoá trên K kênh nên với cặp đa-type, điểm mỗi triplet bị chia sẻ phần type-mass — hạn chế cấu trúc, báo cùng strata.

**Eval sets:** Task P positives = cặp mà MỌI kênh dương nằm trong fold-test (pair chưa từng thấy trong train — pair-seen sẽ rò existence qua mask). Task T positives = mọi triplet fold-test, phân tầng pair-seen (mask=1) vs pair-unseen.

## 4. F02 — Phân loại từng adaptation

| ID | Thay đổi | Căn cứ upstream | Lớp | Ghi chú |
|---|---|---|---|---|
| A1 | K=4 → 5 types, đa-nhãn | — | adaptation bắt buộc (dữ liệu) | y multi-hot; type CE của U1 đã đa-nhãn |
| A2 | exist head σ(mi·diag(r_exist)·dis) | L490–493 | **port trung thực** | cùng projections của readout U1 |
| A3 | focal exist γ=2, α=0.5, sum/(n_pos+n_neg) | L115–133 | **port trung thực** | α=0.5 từ param.py default |
| A4 | pos = train-mask cells; neg = mask-zero subsample ≤10×, MỘT lần, pinned rng | L653–665 | port ngữ nghĩa + rng mới | upstream dùng `torch.randperm` một lần; ta pin `default_rng(inter_seed+77)` và hash danh sách |
| A5 | `lam_exist=0.3` cộng vào objective U1 (đơn vị đã chuẩn hoá) | `exist_weight=0.3` | adaptation đơn vị | KHÔNG tương đương tỉ lệ 0.3/0.7 của upstream — khai báo |
| A6 | split theo triplet-fold thay pair-fold | fold ô v2.0 | adaptation bắt buộc | partition `default_rng(0)` giống `triplet_baseline` để so đối chứng |
| A7 | eval unknown sampling: seed 0–4, equal-count, không thay thế | L1028–1063 | **port trung thực** (đổi không gian: ô → triplet) | |
| A8 | triplet score = exist×softmax_k | — | **new construction** | justification §3; upstream chỉ chấm kênh 0 |
| A9 | strata info-condition | — | new (yêu cầu F03) | §5 |
| A10 | bundle schema giữ nguyên (`train_pairs`/`y_train` multi-hot thu hẹp) | builder REV3 | adaptation schema | y_train chỉ chứa kênh inner_train — held-out canary |

Không có mục nào được tuyên bố "author-faithful v3.2": pipeline tác giả trên v3.2 (data/fold/sampling/metric) là nguồn đóng — mọi lựa chọn trên là diễn giải có khai báo.

## 5. F03 — Điều kiện thông tin (info conditions) chuẩn hoá

Bốn phân loại ô unknown (i,j,k) với Y=0, được đếm và phân tầng trong eval:

| Stratum | Định nghĩa | Ý nghĩa |
|---|---|---|
| `triplet_zero_on_train_pair` | mask[i,j]=1, Y[i,j,k]=0 | pair đã biết — khó nhất (chỉ phân biệt theo type) |
| `triplet_zero_on_heldout_pair` | mask=0 nhưng ∃k′: Y[i,j,k′]=1 (kênh nằm ở inner_val/test) | pair thực sự positive nhưng ngoài train — rủi ro transductive |
| `pair_zero` | Y[i,j,:]=0 ∀k | unknown pair thật — dễ nhất |
| `train_mask_unknown` | mask[i,j]=0 (hợp hai lớp trên) | tập cha của 2 dòng trước |

Positive test cũng phân tầng: `pair_seen` (mask=1 — existence đã biết, đo riêng type channel) vs `pair_unseen`.

**Baseline đối chứng:** logistic train-only (count features của `triplet_baseline.py`, fit trên inner_train + unknown sample pinned) chấm TRÊN CÙNG tập eval pinned — chuẩn hoá info-condition: cả baseline lẫn U1X chỉ thấy inner_train.

## 6. F04 — Pinning + verification

Pin trước training: split partition (seed/thuật toán + hash index), inner split, negative-supervision subsample (index+hash), eval unknown samples (index+hash), loss config (`lam_exist=0.3`, γ=2, α=0.5, subsample 10×), hyperparams (cfg U1 nguyên vẹn), seeds (1234/42/2025), primary metric (AUROC trên Task T), tất cả source/bundle/input hashes. Fixture đa-nhãn + held-out label canary + kiểm chứng metric trong `fixtures.py`, chạy trong preflight.

## 7. Bảng F01–F04 → gate

| Req | Nội dung | Xử lý tại | Gate | Quyết định |
|---|---|---|---|---|
| F01 | tách pair/triplet existence | §3 + `eval_cvtriplet.py` (2 task riêng, joint score có chứng minh) | G4 sanity | đã thiết kế |
| F02 | audit + phân loại adaptation | §2,§4 | — | hoàn tất trong tài liệu này |
| F03 | info conditions + logistic train-only | §5 + `baseline_logistic.py` | G2 fixtures, G4 | đã thiết kế |
| F04 | pin split/neg/loss/hparam/seed/metric/hash + fixture + canary + metric verify | `split.py`,`preflight.py`,`fixtures.py`,`contracts/` | G1 preflight | đã thiết kế |

| Gate | Tiêu chí chấp nhận | Trạng thái |
|---|---|---|
| G1 preflight | mọi pin hash PASS; fixtures PASS; canary không rò | chờ chạy |
| G2 build | 5 bundle hash + sidecar khớp contract; không sửa artifact cũ | chờ chạy |
| G4 sanity fold0 | train hữu hạn hết horizon; exist channel phân biệt pos/neg trên inner_val; eval+baseline chạy hết; verify PASS | chờ chạy — KHÔNG phải kết quả khoa học |
| G5 full 5×3 | bị chặn — cần duyệt riêng sau khi G1–G4 xanh | BLOCKED theo review |

## 8. Không khẳng định

- Arm này **không** là tái hiện CVtriplet của paper (data/fold/sampling/metric v3.2 tác giả không public).
- `s(i,j,k)=σ·softmax_k` là diễn giải của ta, không phải cách tác giả chấm triplet.
- Không so điểm với 0.86/0.92 của paper; chỉ so U1X vs logistic train-only trên cùng split mô phỏng.
