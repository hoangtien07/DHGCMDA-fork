# Section 7.4 — Implementation Issues in Public Code (MLRC paper outline)

> Draft outline cho section thứ 4 đóng góp. Ngày: 2026-05-11.

## Goal

Document các implementation bugs + hardcoded values phát hiện trong code public DHGCMDA-fork. Đây là contribution thực tế cho cộng đồng reproducibility — community benefit từ việc identify these issues.

## Structure (target ~600 words, 1 figure or table)

### 7.4.1 Overview

Từ Plan A đến Plan E, trong quá trình debug khi metrics không khớp paper, chúng tôi phát hiện 3 bugs nghiêm trọng + 5 hardcoded values + 5 dead CLI flags trong code public. Một số bugs có ảnh hưởng đến tính reproducibility (multi-seed broken), số khác giảm flexibility cho follow-up research (hyperparameter sweep impossible without code edit).

### 7.4.2 Critical bugs (3, affect multi-seed reproducibility)

**Bug 1 — `seed_torch()` not invoked with `args.seed`**

`main_experiments_hetero1.py:65` calls `seed_torch()` at module load time với default seed=1234. Sau khi `args = parameter_parser()` ở `__main__`, KHÔNG có call `seed_torch(args.seed)` lại. Result: CLI flag `--seed` bị ignored cho training pipeline.

**Bug 2 — Hardcoded `np.random.seed(0)` trong `prepareData.py:271`**

Trước khi shuffle indices cho train/test split, code reset `np.random.seed(0)` HARDCODED. Bất kể user truyền seed nào, train/test split LUÔN dùng seed=0.

**Bug 3 — Indices cache key thiếu seed**

Cache key format: `f"indices_{hash(association_matrix)}_{validation}"`. KHÔNG include seed → cache hit prevent recomputation khi seed thay đổi (but with bug #2, this didn't matter in practice — split was fixed anyway).

**Combined effect**: Tất cả results báo cáo dùng seed=0 train/test split. Cross-seed comparisons từ `args.seed` flag are **invalid for the trained data path** (only model init weight differs by seed).

**Our fixes**: 
- Add `seed_torch(args.seed)` after parse args in `main_experiments_hetero1.py`
- Add `seed=0` parameter to `preprocess_indices()`, propagate from `args.seed`
- Include seed in cache key: `f"indices_{hash}_{validation}_seed{seed}"`

### 7.4.3 Hardcoded hyperparameters (5, reduce flexibility)

| Location | Hardcoded value | Affected CLI flag | Paper section blocked |
|---|---|---|---|
| `main_experiments_hetero1.py:817-864` | `K_neigs=[13]` | `--K_neigs` | Fig.3 K sensitivity sweep |
| `main_experiments_hetero1.py:1482` | `lr=0.0001, weight_decay=1e-5` | `--lr, --weight_decay` | Optimizer tuning |
| `main_experiments_hetero1.py:120` | `focal_gamma=2.0` | `--focal_gamma` | Loss function tuning |
| `hetero_model.py:701-702` | `num_types=4` | `--num_association_types` | v3.2 dataset adaptation |
| `hetero_model.py:639` | `temperature=0.5, margin=0.5` | `--inter_view_temperature, --inter_view_margin` | Inter-view CL tuning |

**Implication**: Paper Fig.2 (sensitivity to t, λ2) và Fig.3 (K sensitivity) KHÔNG REPRODUCIBLE WITHOUT CODE EDITS. User cannot run paper's hyperparameter analysis with public code as-is.

### 7.4.4 Dead CLI flags (5, defined but never read)

`--cl_weight, --cl_temperature, --use_focal_loss, --enable_type_prediction, --eval_metric` are all defined trong `param.py` nhưng KHÔNG được sử dụng anywhere in main pipeline. User có thể truyền các flags này NHƯNG không có effect — silent no-op.

### 7.4.5 Recommendations to upstream maintainers

1. **Critical**: Fix seed propagation (3 bug fixes) để multi-seed experiments thực sự có ý nghĩa.
2. **Important**: Replace hardcoded values với args reading. Specifically `K_neigs` (paper Fig.3 cần), `lr/wd` (training tuning), `num_types` (v3.2 adaptation).
3. **Cleanup**: Remove dead CLI flags hoặc wire them up.
4. **Test**: Add unit test verifying `args.X` propagates to corresponding code paths.

### 7.4.6 Implications for replication studies

Findings here highlight a broader issue in ML reproducibility: **CLI flags defined in scripts không guarantee functional behavior**. Other groups reproducing DHGCMDA might unknowingly run với incorrect seed → see "different" variance than paper because of wrong source of randomness. We recommend that replication studies systematically verify CLI flag → code path linkage before reporting metrics.

## Figures / tables for this section

- **Table 7.4.1**: Bugs summary (3 critical + 5 hardcoded + 5 dead flags)
- **Optional Figure 7.4.1**: Multi-seed before/after fix — shows variance was 0 (all same data split) vs proper variance (different splits per seed). Use Plan B2 multi-seed v3 results.

## Word count estimate

- 7.4.1: 80
- 7.4.2: 200
- 7.4.3: 120
- 7.4.4: 60
- 7.4.5: 80
- 7.4.6: 80
- **Total ~620 words** (within 800-word budget for §7 contribution)

## References needed

- ML Reproducibility Challenge guidelines
- "Reproducibility in ML: A practical framework" (Pineau et al. 2021)
- DHGCMDA paper (Sun Y. et al. 2026)
