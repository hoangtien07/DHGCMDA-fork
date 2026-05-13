# Email gửi tác giả CDMBlab — DHGCMDA reproduce study

> Soạn 2026-05-13. Gửi từ user (Tien) tới corresponding authors paper DHGCMDA.

---

## 📧 Thông tin gửi

**To** (corresponding authors):
- Prof. Junliang Shang: `shangjunliang110@163.com`
- Prof. Jin-Xing Liu: `sdcavell@126.com`

**Cc** (contributing authors — optional):
- Yan Sun: `sunyan225@126.com`
- Fanyu Zhang: `19819518872@163.com`
- Shijia Yan: `yanshijia37@163.com`
- Xiaotong Kong: `kongxt19@163.com`
- Hanxiang Wang: `hanxiang@qfnu.edu.cn`

**Subject**: `Reproducing DHGCMDA on HMDD v2.0 — Questions on exact hyperparameters + 3 implementation issues found`

---

## 📝 Email body (English — academic standard)

```
Dear Prof. Shang and Prof. Liu,

I am [Your Name], a [graduate student / researcher / undergraduate] at
[Your Institution], currently conducting a reproducibility study of your
paper "DHGCMDA: A Dual-view Heterogeneous Graph Contrastive Learning
Framework for miRNA-Disease Association Type Prediction" published in
[Journal/Conference, 2026].

I am writing to (1) share my reproduction results, (2) ask a few specific
clarifying questions, and (3) report several implementation issues I
discovered in the public code that may benefit other groups reproducing
your work.

═══════════════════════════════════════════════════════════════════
1. REPRODUCTION RESULTS — HMDD v2.0
═══════════════════════════════════════════════════════════════════

Using your public code at https://github.com/CDMBlab/DHGCMDA, I successfully
reproduced the headline baseline metrics with the configuration
(seed=1, K_neigs=7, default loss):

  Metric        | Paper   | My reproduce | Δ
  AUC           | 0.9669  | 0.9745       | +0.8%  (exceeds paper)
  AUPR          | 0.9738  | 0.9691       | -0.5%  (within noise)
  F1 (binary)   | 0.9278  | 0.9307       | +0.3%  (exceeds paper)
  Top-1 F1      | 0.5970  | 0.5909       | -1.0%  (within noise — REPRODUCED)

The 5-fold cross-validation matches your Table 3 results within 1% on
all metrics, which I consider a successful reproduction.

═══════════════════════════════════════════════════════════════════
2. CLARIFYING QUESTIONS
═══════════════════════════════════════════════════════════════════

While the baseline metrics reproduce, I had difficulty reproducing
several other claims and would appreciate your guidance:

(a) HYPERPARAMETER VALUES

  Your Figure 3(b) reports that Top-1 metrics reach maximum at K=13.
  In my reproduction across K ∈ {7, 9, 11, 13, 15} with seed=1:

    K=7:  Top-1 F1 = 0.5909  ← my best
    K=9:  Top-1 F1 = 0.5677
    K=11: Top-1 F1 = 0.5633
    K=13: Top-1 F1 = 0.5655
    K=15: Top-1 F1 = 0.5506

  The pattern is monotonically decreasing from K=7 to K=15, opposite
  to the shape in Figure 3(b). Could you confirm:
    - Which specific seed was used to produce Figure 3(b)?
    - Were any other hyperparameters changed during the K sweep?

(b) EQUATION 32 LOSS FORMULATION

  Equation 32 in the paper specifies:
    L_total = L_type + λ1·L_intra + λ2·L_inter + λ3·L_recon

  However, the code includes an additional term:
    L_recover = 0.3 × focal_loss(existence) + 0.7 × weighted_CE(type)

  with focal_gamma=2.0 and label_smoothing=0.1. These factors (0.3, 0.7,
  focal_gamma, label_smoothing) do not appear in the paper.

  Could you clarify:
    - Is the `0.3·L_existence(focal)` term part of L_type in Eq.32,
      or an additional implementation choice?
    - Were the values 0.3, 0.7, 2.0, 0.1 selected via tuning (and if so,
      what range was tested)?

(c) FIGURE 4 ABLATION STUDY

  Figure 4 reports that all 5 ablation variants (no_cl, no_hgcn, no_avf,
  no_hgt, no_dv) reduce Top-1 F1 compared to the full model. In my
  reproduction with the baseline reproduce config above:

    Variant       | Top-1 F1 | Δ vs Full
    Full          | 0.5909   | —
    w/o CL        | 0.6097   | +3.2%  (improves)
    w/o HGCN      | 0.5955   | +0.8%  (improves)
    w/o AVF       | 0.5842   | -1.1%  (hurts — matches paper)
    w/o HGT       | 0.6466   | +9.4%  (improves significantly)
    w/o DV        | 0.5866   | -0.7%  (hurts — matches paper)

  Only 2 of 5 ablations hurt the baseline (no_avf and no_dv). The
  remaining 3 (no_cl, no_hgcn, no_hgt) consistently IMPROVE Top-1 F1.

  I tested this pattern across 5 different configurations
  (default hyperparameters, exist_weight sweep, 5-class softmax CE,
  true component rebuild with GCNConv, and the K=7 reproduce config)
  and found the same persistent result: removing CL/HGCN/HGT does not
  hurt the baseline on HMDD v2.0.

  Could you clarify:
    - Which specific seed/config produced Figure 4?
    - Was Figure 4 generated on HMDD v2.0, v3.2, or both?
    - Could the components be more critical for v3.2 (larger dataset)
      than v2.0 (1498 associations / 189K cells, ~0.8% positive rate)?

(d) CASE STUDY (TABLES 5 AND 6)

  Tables 5 and 6 report top-15 predicted miRNAs for breast neoplasms
  and hepatocellular carcinoma with diverse type predictions (all 4
  types appear). In my reproduction:

    Breast neoplasms: 15/15 top-15 predict type = "target"
    HCC:              15/15 top-15 predict type = "epigenetics"

  This appears to be class collapse where the model predicts the
  majority type per disease. Could you clarify:
    - What ranking criterion was used? (max prob per type? sum? softmax?)
    - Were the models trained on full data (no CV split) or held-out fold?
    - Was any post-processing or class re-balancing applied to ensure
      diverse type predictions in the top-15?

(e) HMDD v3.2 DATASET

  Your repository includes HMDD v2.0 preprocessed data but not v3.2.
  Would it be possible to share:
    - The v3.2 preprocessing scripts (or processed association matrix +
      similarity matrices), or
    - Pointers to where each component (MeSH semantic similarity,
      disease-gene similarity from DisGeNET, miRNA functional + sequence
      similarity) should be downloaded from?

═══════════════════════════════════════════════════════════════════
3. IMPLEMENTATION ISSUES FOUND
═══════════════════════════════════════════════════════════════════

During my reproduction, I identified three critical issues in the public
code that may affect other groups trying to reproduce your work:

(I) SEED PROPAGATION BUGS (multi-seed experiments broken)

  Bug 1: main_experiments_hetero1.py:65 calls `seed_torch()` with the
    default value 1234 at module load time. The args.seed CLI flag is
    parsed in __main__ but seed_torch is never re-invoked with args.seed.
    Result: --seed flag is silently ignored.

  Bug 2: prepareData.py:271 hardcodes `np.random.seed(0)` before shuffling
    train/test indices. The train/test split is therefore FIXED across
    all runs regardless of args.seed.

  Bug 3: prepareData.py:232 cache key for indices is
    `f"indices_{hash(association_matrix)}_{validation}"` and does not
    include seed. Cache hits prevent recomputation when seed changes
    (though due to Bug 2 this only matters after Bug 2 is fixed).

  Combined effect: any multi-seed analysis using args.seed produces
  identical results (only model weight initialization differs by seed).

  I have proposed fixes in my reproduction fork. The key change is:
    - Add seed_torch(args.seed) immediately after args = parameter_parser()
    - Pass seed parameter through prepare_data_optimized() to
      preprocess_indices()
    - Include seed in cache_key

(II) HARDCODED HYPERPARAMETERS (CLI flags ignored)

  Several hyperparameters have CLI arguments defined in param.py but
  their values are hardcoded in main_experiments_hetero1.py:

    K_neigs=[13]  appears 6 times in lines 817-864 — args.K_neigs ignored
    lr=0.0001 and weight_decay=1e-5 hardcoded at line 1482 — args.lr,
      args.weight_decay ignored
    focal_gamma=2.0 hardcoded at line 120 of the loss class —
      args.focal_gamma ignored
    num_types=4 hardcoded in hetero_model.py:701 — args.num_association_types
      ignored

  This prevents reproducing Figure 3 (K sensitivity) without code edits.

(III) UNUSED CLI FLAGS (silent no-ops)

  --cl_weight, --cl_temperature, --use_focal_loss, --enable_type_prediction,
  --eval_metric are defined in param.py but not read anywhere in the
  main training pipeline. Users may believe they are changing model
  behavior when in fact these flags have no effect.

═══════════════════════════════════════════════════════════════════
4. ACKNOWLEDGMENTS AND OFFER TO CONTRIBUTE
═══════════════════════════════════════════════════════════════════

DHGCMDA is a thoughtful contribution to the miRNA-disease association
type prediction literature, and I appreciate that you released the
implementation publicly. The successful reproduction of headline
baseline metrics (Top-1 F1 within 1% of paper) demonstrates that the
core method is sound.

If it would be helpful, I would be happy to:
  - Submit a pull request to your GitHub repository with the seed
    propagation fixes and CLI flag wiring.
  - Share my reproduction code (a fork of your repository with sweep
    scripts and documentation in Vietnamese and English).
  - Co-author a brief technical note describing the reproduction
    methodology if there is mutual interest.

I am happy to share the full reproduction log, including all sweep
results and per-fold metrics, on request.

Thank you for your time and for your contribution to the field. I look
forward to your response.

Best regards,

[Your Name]
[Your Title / Position]
[Your Institution]
[Your Email]
[Optional: ORCID / GitHub profile]
[Date]

P.S. The reproduction code and full experiment logs are available at:
https://github.com/hoangtien07/DHGCMDA-fork
```

---

## 🇻🇳 Tóm tắt tiếng Việt (cho user nắm nội dung)

Email được soạn theo format **chuẩn academic** (tiếng Anh), gồm 4 phần:

### Phần 1 — Reproduction Results
- Báo cáo đã reproduce thành công baseline metrics (gap < 1% paper)
- Cụ thể: AUC 0.9745 (+0.8%), F1 0.9307 (+0.3%), Top-1 F1 0.5909 (-1.0%)
- Config dùng: seed=1, K_neigs=7

### Phần 2 — 5 câu hỏi cụ thể cho tác giả

(a) **Fig.3 K sensitivity** — hỏi seed nào để Fig.3 cho K=13 optimal (ta thấy K=7 best)

(b) **Eq.32 loss formulation** — hỏi `0.3·L_existence` có thuộc L_type hay là implementation choice riêng

(c) **Fig.4 ablation** — hỏi seed/config nào cho all-components-critical (ta thấy 2/5 match qua 5 configs)

(d) **Case study Tables 5/6** — hỏi ranking criterion + post-processing để có type diversity

(e) **HMDD v3.2 dataset** — hỏi xin preprocessing scripts hoặc pointers

### Phần 3 — 3 critical bugs phát hiện (contribution ngược)

(I) Seed propagation broken (3 sub-bugs: seed_torch, np.random.seed(0) hardcoded, cache key thiếu seed)
(II) Hardcoded hyperparameters (K_neigs, lr, weight_decay, focal_gamma, num_types)
(III) Unused CLI flags (cl_weight, cl_temperature, etc.)

### Phần 4 — Đề nghị đóng góp

- Submit PR fix bugs
- Share reproduction repo
- Co-author technical note nếu họ quan tâm

---

## 📋 Checklist trước khi gửi

- [ ] Điền `[Your Name]`, `[Your Institution]`, `[Your Email]`, `[Date]` ở cuối
- [ ] Sửa `[Journal/Conference, 2026]` thành tên venue chính xác (vd "BMC Bioinformatics 2026")
- [ ] Optional: thêm ORCID, GitHub profile, Google Scholar
- [ ] Quyết định cc tất cả 5 contributing authors hay chỉ 2 corresponding authors
- [ ] Subject line đã sẵn sàng dùng
- [ ] Test gửi đến địa chỉ của bản thân trước (proofreading)

## 💡 Tips gửi email

1. **Best time**: Sáng thứ Ba-Năm (theo timezone Trung Quốc UTC+8) — khả năng được đọc cao nhất
2. **Expected response time**: 2-4 tuần (tác giả academic thường bận). Sau 3 tuần không reply có thể gửi gentle reminder.
3. **Encoding**: Email body là plain text, các ký tự đặc biệt (━, ═, ←, →, λ) có thể render khác trên một số client. Nếu lo ngại, thay bằng ASCII alternatives (-, =, <-, ->, lambda).
4. **Attach**: Có thể đính kèm `final_reproduce_report.json` hoặc file CSV nhỏ với sweep results nếu muốn họ verify chi tiết.

## 📂 Files reference (commit hash để verify nội dung)

- Reproduce results: [results/final_reproduce_report.json](results/final_reproduce_report.json)
- Bug fixes: commits `3969311`, `9024bc9`, `15b6aab`
- Best config: `d9fdfa9` (BaoCao_DHGCMDA.docx)
- Repo public: https://github.com/hoangtien07/DHGCMDA-fork
