# TRACK_N_INDEPENDENT_REVIEW — phản biện nội bộ (WP4, REV4.1)

Đối chiếu protocol Track N theo các trục council yêu cầu.

## 1. Tính mới (novelty)

- **Mạnh:** benchmark leakage-controlled + metric contracts +
  strata/multi-label analysis là thiếu trong literature; phát hiện
  "degree priors ≈/hơn model trên nhiều convention" là kết quả
  có giá trị công bố độc lập với model mới.
- **Yếu:** nếu chỉ là "U1 + BCE multi-label" thì đó là ablation,
  không phải method mới — phải giữ claim ở mức "benchmark +
  honest evaluation + one preregistered arm", không oversell.
- Phán quyết: NEW-ENOUGH as benchmark+evaluation contribution;
  method novelty is modest by design (roadmap cho phép).

## 2. Độ tin cậy nhãn (label trust)

- HMDD v3.2 type labels là curated nhưng không có mechanism để ta
  verify per-label accuracy; 24.2% multi-type pairs có thể chứa
  annotation co-occurrence artifacts. → cap ceiling unknown; ghi
  là limitation cứng, không cố "fix" bằng smoothing.
- Đề xuất đã có trong protocol: co-occurrence vs independent-label
  check làm descriptive analysis (không phải claim).

## 3. Feasibility CPU

- Đo thực tế: U1-like ~10-13 phút/run (đã chạy 45+ runs); baselines
  phút-mức. 3 tracks × ~20-45 runs ≈ vài giờ — khả thi.
- Rủi ro: label propagation trên 411×271 graph trivial; logistic
  trivial. PASS.

## 4. Baseline strength

- `prior_product_laplace1` đã thắng U1 trên nhiều convention —
  floor cao, tốt cho tính trung thực nhưng có nguy cơ H1 fail
  ngay T-pair. Đã preregister: falsification = kết quả âm báo
  thẳng, benchmark vẫn là contribution.
- TDRC anchor: tái hiện được trên tensor nó (0.42-0.44 cùng
  convention khác); port sang strict tensor + metric của ta là
  công việc mới có thể lệch — đánh dấu optional.

## 5. Leakage

- Canary bắt buộc trước graph creation — đúng; nhưng cần kiểm
  tra rằng similarity ngoài (S_seq/S_gene/S_sem) không ngầm chứa
  association info: đã provenance-verified (ngoại trừ S_func
  rebuild train-only). Lỗ hổng còn lại: fold-shared entity
  embeddings — chấp nhận được cho T-pair, cold-start phải re-derive
  toàn bộ features chỉ từ train entities.

## 6. Power

- 15 cells paired, target Δ≥0.03 vs precedent leak effect +0.049 —
  phát hiện được nếu hiệu ứng thật cỡ đó; nhưng 15 cells correlated
  → chỉ descriptive. Không hứa significance; đúng chuẩn roadmap.
- Cold-start power thấp hơn (entity-level folds → variance lớn);
  ghi exploratory nếu cần.

## 7. Rủi ro quy trình

- "Same embedding inputs as U1" cho new arm tái dùng artifacts —
  chấp nhận, nhưng phải tái xác nhận embeddings là train-only.
- Metric bugs đã có tiền lệ (REV4.1) → yêu cầu: mọi metric mới có
  fixture perfect/worst trước khi dùng (đã ghi vào protocol).

## Verdict

Protocol đủ chặt để trình owner duyệt. Ba caveat phải giữ nguyên
trong mọi báo cáo sau này: (i) label trust cap; (ii) 15 cells
correlated — descriptive only; (iii) retrospective flag trên
T-pair reuse.
